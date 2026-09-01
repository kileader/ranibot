"""Discord transport for Ranibot, including opt-in per-server memory."""

import asyncio
import json
import logging
import math
import time
from datetime import datetime
from typing import Literal

import discord
from discord import app_commands

from agents import AGENTS, consider, consider_selected_message, consult_agents, synthesize
from config import Settings
from llm import LLM, create_llm
from memory import (
    MEMORY_BATCH_SIZE,
    DisabledMemoryStore,
    MemoryStore,
    create_memory_store,
    extract_server_memories,
)

logger = logging.getLogger(__name__)
HISTORY_LIMIT = 30
MAX_MESSAGE_CHARS = 1500
MAX_QUESTION_CHARS = 1500


async def read_context(channel: discord.TextChannel | discord.Thread, before: datetime) -> str | None:
    """Read only the last 30 messages at invocation, then retain human text."""
    messages = []
    async for message in channel.history(limit=HISTORY_LIMIT, before=before, oldest_first=False):
        if message.author.bot or message.webhook_id is not None or message.is_system():
            continue
        text = message.clean_content.strip()
        if not text:
            continue
        if len(text) > MAX_MESSAGE_CHARS:
            text = text[:MAX_MESSAGE_CHARS] + " [truncated]"
        messages.append({
            "username": message.author.name,
            "display_name": message.author.display_name,
            "text": text,
        })
    if not messages:
        return None
    messages.reverse()
    return json.dumps(messages, ensure_ascii=False)


class RaniBot(discord.Client):
    def __init__(self, settings: Settings, llm: LLM, memory: MemoryStore | None = None):
        intents = discord.Intents.none()
        intents.guilds = True
        # Message events are used only in servers that explicitly enable memory.
        intents.messages = True
        intents.message_content = True
        super().__init__(intents=intents, max_messages=None,
                         allowed_mentions=discord.AllowedMentions.none())
        self.settings = settings
        self.llm = llm
        self.memory = memory or DisabledMemoryStore()
        self.started_at = time.monotonic()
        self.active_channels: set[int] = set()
        self.memory_locks: dict[int, asyncio.Lock] = {}
        self.tree = app_commands.CommandTree(self)
        for name, description, callback in (
            ("agents", "Let Mira, Hex, and Moss consider joining the conversation.", self.run_agents),
            ("ask", "Ask one agent a question; enabled server memory may also be supplied.", self.run_ask),
            ("status", "Show uptime and configured model without making an AI request.", self.run_status),
            ("chesslab", "Share the Chess Lab app link and introduction; no AI request.", self.run_chesslab),
            ("synthesize", "Map recent common ground, tensions, and open questions with one AI request.", self.run_synthesize),
            ("consent", "Explain exactly what Ranibot reads, sends, stores, and costs.", self.run_consent),
            ("help", "Explain Ranibot's commands and what gets sent to AI.", self.run_help),
        ):
            command = app_commands.Command(name=name, description=description, callback=callback)
            command.guild_only = True
            self.tree.add_command(command)
        memory_group = app_commands.Group(
            name="memory", description="Inspect and control this server's persistent memory.",
            guild_only=True,
        )
        for name, description, callback in (
            ("status", "Show whether memory is configured and enabled.", self.run_memory_status),
            ("enable", "Let Ranibot learn from human messages in accessible channels.", self.run_memory_enable),
            ("pause", "Stop learning from messages and stop applying saved memory.", self.run_memory_pause),
            ("list", "Privately inspect recent shared and agent memories.", self.run_memory_list),
            ("forget", "Delete one memory by its numeric ID (Manage Server required).", self.run_memory_forget),
            ("clear", "Delete this server's memories and buffered messages.", self.run_memory_clear),
        ):
            memory_group.add_command(app_commands.Command(name=name, description=description, callback=callback))
        self.tree.add_command(memory_group)
        for name, callback in (
            ("Ask Mira about this", self.message_mira),
            ("Analyze with Hex", self.message_hex),
            ("Connect with Moss", self.message_moss),
        ):
            self.tree.add_command(app_commands.ContextMenu(name=name, callback=callback))
        self.tree.on_error = self.on_command_error

    async def setup_hook(self) -> None:
        await self.memory.start()
        if self.settings.guild_id:
            guild = discord.Object(id=self.settings.guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            logger.info("Synced eight slash command roots and three message actions to test server %s", guild.id)
        else:
            await self.tree.sync()
            logger.info("Synced eight slash command roots and three message actions globally")

    async def on_ready(self) -> None:
        logger.info("Ranibot connected as bot ID %s", self.user.id)

    async def close(self) -> None:
        try:
            await super().close()
        finally:
            try:
                await self.memory.close()
            finally:
                await self.llm.close()

    async def on_message(self, message: discord.Message) -> None:
        """Buffer human text only when an administrator enabled memory for its server."""
        if (message.guild is None or message.author.bot or message.webhook_id is not None
                or message.is_system()):
            return
        text = message.clean_content.strip()
        if not text:
            return
        guild_id = message.guild.id
        try:
            if not await self.memory.is_enabled(guild_id):
                return
            pending = await self.memory.append_message(
                guild_id, message.channel.id, message.id, message.author.display_name, text,
            )
            if pending >= MEMORY_BATCH_SIZE:
                await self._extract_pending_memory(guild_id)
        except Exception as exc:
            logger.warning("Could not buffer server memory in guild %s (%s)", guild_id, type(exc).__name__)

    async def _extract_pending_memory(self, guild_id: int) -> None:
        lock = self.memory_locks.setdefault(guild_id, asyncio.Lock())
        async with lock:
            messages = await self.memory.pending_messages(guild_id, MEMORY_BATCH_SIZE)
            if len(messages) < MEMORY_BATCH_SIZE:
                return
            memories = await extract_server_memories(self.llm, messages)
            if memories is None:
                return
            await self.memory.save_extraction(guild_id, messages, memories)
            logger.info(
                "Processed %s buffered messages into %s server memories for guild %s",
                len(messages), len(memories), guild_id,
            )

    async def _memory_context(self, guild_id: int, agent_name: str) -> tuple[list[str], list[str]]:
        try:
            if await self.memory.is_enabled(guild_id):
                return await self.memory.get_context(guild_id, agent_name)
        except Exception as exc:
            logger.warning("Could not load memory for guild %s (%s)", guild_id, type(exc).__name__)
        return [], []

    async def _remember_agent_turn(
        self, guild_id: int, channel_id: int, agent_name: str, text: str, message_id: int | None = None,
    ) -> None:
        try:
            if await self.memory.is_enabled(guild_id):
                await self.memory.add_agent_memory(guild_id, agent_name, text, channel_id, message_id)
        except Exception as exc:
            logger.warning("Could not save %s journal in guild %s (%s)", agent_name, guild_id, type(exc).__name__)

    async def run_agents(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        channel = interaction.channel
        if interaction.guild is None or not isinstance(channel, (discord.TextChannel, discord.Thread)):
            await interaction.edit_original_response(content="Use /agents in a server text channel or thread.")
            return
        permissions = interaction.app_permissions
        can_send = permissions.send_messages_in_threads if isinstance(channel, discord.Thread) else permissions.send_messages
        if not (permissions.view_channel and permissions.read_message_history and can_send):
            await interaction.edit_original_response(
                content="I need View Channel, Read Message History, and Send Messages "
                        "(Send Messages in Threads for a thread) here."
            )
            return
        if not interaction.permissions.read_message_history:
            await interaction.edit_original_response(content="You need Read Message History to invoke /agents here.")
            return
        if channel.id in self.active_channels:
            await interaction.edit_original_response(content="The agents are already considering this channel. Try again when they finish.")
            return

        self.active_channels.add(channel.id)
        posted = 0
        try:
            context = await read_context(channel, before=interaction.created_at)
            if context is None:
                await interaction.edit_original_response(
                    content="No readable human text in the last 30 messages. "
                            "If you expected text, check the Message Content Intent in the developer portal."
                )
                return
            logger.info("Considering /agents in channel %s", channel.id)
            memory_contexts = await asyncio.gather(*(
                self._memory_context(interaction.guild.id, agent.name) for agent in AGENTS
            ))
            shared_memory = next((shared for shared, _ in memory_contexts if shared), [])
            journals = {
                agent.name: journal for agent, (_, journal) in zip(AGENTS, memory_contexts)
            }
            results = await consult_agents(self.llm, context, shared_memory, journals)
            for result in results:
                if result.text:
                    await channel.send(
                        f"**{result.agent.name}:** {result.text}",
                        allowed_mentions=discord.AllowedMentions.none(), suppress_embeds=True,
                    )
                    await self._remember_agent_turn(
                        interaction.guild.id, channel.id, result.agent.name, result.text
                    )
                    posted += 1
            failed = [result.agent.name for result in results if result.failed]
            if failed:
                status = f"Posted {posted} response(s). Could not get a decision from {', '.join(failed)}; check the console and API configuration."
            elif posted:
                status = f"Posted {posted} response(s); the remaining agents stayed silent."
            else:
                status = "All three agents chose silence."
            await interaction.edit_original_response(content=status)
        except discord.Forbidden:
            await interaction.edit_original_response(
                content=f"Discord denied access while reading or posting ({posted} responses posted). Check channel permissions."
            )
            logger.warning("Discord access denied in channel %s", channel.id)
        except discord.HTTPException as exc:
            logger.warning("Discord request failed in channel %s (HTTP %s)", channel.id, exc.status)
            await interaction.edit_original_response(
                content=f"Discord request failed ({posted} responses posted). Try /agents again later."
            )
        finally:
            self.active_channels.discard(channel.id)

    @app_commands.describe(agent="Choose Mira, Hex, or Moss", question="Your question (1-1500 characters; sent to AI)")
    async def run_ask(
        self, interaction: discord.Interaction, agent: Literal["Mira", "Hex", "Moss"],
        question: app_commands.Range[str, 1, MAX_QUESTION_CHARS],
    ) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        channel = interaction.channel
        if interaction.guild is None or not isinstance(channel, (discord.TextChannel, discord.Thread)):
            await interaction.edit_original_response(content="Use /ask in a server text channel or thread.")
            return
        question = question.strip()
        selected = next((item for item in AGENTS if item.name == agent), None)
        if selected is None or not 1 <= len(question) <= MAX_QUESTION_CHARS:
            await interaction.edit_original_response(content="Choose Mira, Hex, or Moss and enter a question of 1-1500 characters.")
            return
        permissions = interaction.app_permissions
        can_send = permissions.send_messages_in_threads if isinstance(channel, discord.Thread) else permissions.send_messages
        if not (permissions.view_channel and can_send):
            await interaction.edit_original_response(content="I need View Channel and Send Messages (Send Messages in Threads for a thread) here.")
            return
        if channel.id in self.active_channels:
            await interaction.edit_original_response(content="An AI request is already running in this channel. Try again when it finishes.")
            return

        self.active_channels.add(channel.id)
        try:
            logger.info("Handling /ask for %s in channel %s", selected.name, channel.id)
            shared_memory, journal = await self._memory_context(interaction.guild.id, selected.name)
            result = await consider(
                selected, self.llm, json.dumps({"question": question}, ensure_ascii=False),
                direct=True, shared_memory=shared_memory, agent_journal=journal,
            )
            if result.failed:
                await interaction.edit_original_response(content=f"Could not get an answer from {selected.name}. Try again later; check the bot logs if this persists.")
            elif not result.text:
                await interaction.edit_original_response(content=f"{selected.name} did not return an answer. Try rephrasing your question.")
            else:
                await channel.send(f"**{selected.name}:** {result.text}",
                                   allowed_mentions=discord.AllowedMentions.none(), suppress_embeds=True)
                await self._remember_agent_turn(
                    interaction.guild.id, channel.id, selected.name, result.text
                )
                await interaction.edit_original_response(content=f"{selected.name} replied in the channel.")
        except discord.HTTPException as exc:
            logger.warning("Discord /ask request failed in channel %s (HTTP %s)", channel.id, exc.status)
            await interaction.edit_original_response(content="Discord could not complete the reply. Check the channel before retrying, and check my send permissions.")
        finally:
            self.active_channels.discard(channel.id)

    async def message_mira(self, interaction: discord.Interaction, message: discord.Message) -> None:
        await self._run_message_agent(interaction, message, "Mira")

    async def message_hex(self, interaction: discord.Interaction, message: discord.Message) -> None:
        await self._run_message_agent(interaction, message, "Hex")

    async def message_moss(self, interaction: discord.Interaction, message: discord.Message) -> None:
        await self._run_message_agent(interaction, message, "Moss")

    async def _run_message_agent(
        self, interaction: discord.Interaction, message: discord.Message, agent_name: str,
    ) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        channel = interaction.channel
        if interaction.guild is None or not isinstance(channel, (discord.TextChannel, discord.Thread)):
            await interaction.edit_original_response(content="Use this message action in a server text channel or thread.")
            return
        permissions = interaction.app_permissions
        can_send = permissions.send_messages_in_threads if isinstance(channel, discord.Thread) else permissions.send_messages
        if not (permissions.view_channel and can_send):
            await interaction.edit_original_response(content="I need View Channel and Send Messages (Send Messages in Threads for a thread) here.")
            return
        text = message.clean_content.strip()
        if not text:
            await interaction.edit_original_response(content="That message has no readable text. Attachments and embeds are not sent to AI.")
            return
        if len(text) > MAX_MESSAGE_CHARS:
            text = text[:MAX_MESSAGE_CHARS] + " [truncated]"
        if channel.id in self.active_channels:
            await interaction.edit_original_response(content="An AI request is already running in this channel. Try again when it finishes.")
            return
        selected = next(agent for agent in AGENTS if agent.name == agent_name)

        self.active_channels.add(channel.id)
        try:
            logger.info("Handling selected-message action for %s in channel %s", selected.name, channel.id)
            context = json.dumps({"selected_message": text}, ensure_ascii=False)
            shared_memory, journal = await self._memory_context(interaction.guild.id, selected.name)
            result = await consider_selected_message(
                selected, self.llm, context, shared_memory=shared_memory, agent_journal=journal,
            )
            if result.failed:
                await interaction.edit_original_response(content=f"Could not get a response from {selected.name}. Try again later.")
            elif not result.text:
                await interaction.edit_original_response(content=f"{selected.name} did not return a response for that message.")
            else:
                await message.reply(
                    f"**{selected.name}:** {result.text}", mention_author=False,
                    allowed_mentions=discord.AllowedMentions.none(), suppress_embeds=True,
                )
                await self._remember_agent_turn(
                    interaction.guild.id, channel.id, selected.name, result.text, message.id
                )
                await interaction.edit_original_response(content=f"{selected.name} replied to the selected message.")
        except discord.HTTPException as exc:
            logger.warning("Discord message action failed in channel %s (HTTP %s)", channel.id, exc.status)
            await interaction.edit_original_response(content="Discord could not post the reply. Check the channel before retrying.")
        finally:
            self.active_channels.discard(channel.id)

    async def run_status(self, interaction: discord.Interaction) -> None:
        seconds = max(0, int(time.monotonic() - self.started_at))
        days, seconds = divmod(seconds, 86400)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        latency = f"{self.latency * 1000:.0f} ms" if math.isfinite(self.latency) else "not measured yet"
        model = discord.utils.escape_markdown(self.settings.llm_model)
        await interaction.response.send_message(
            f"**Ranibot is responding.**\nUptime: {days}d {hours}h {minutes}m {seconds}s\n"
            f"Discord heartbeat latency: {latency}\n"
            f"Configured AI: {self.settings.llm_provider} / {model}\n"
            "No AI request was made. This does not check API billing or model availability.",
            ephemeral=True, allowed_mentions=discord.AllowedMentions.none(), suppress_embeds=True,
        )

    async def run_chesslab(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_message(
            "**Chess Lab** — Explore your opening results and find positions to study.\n"
            "Import games from Lichess, Chess.com, or PGN. Sign in with Google; your library stays private.\n"
            "**[Open Chess Lab](https://chess-lab-zeta.vercel.app)**\n\n"
            "This command only shares the link. It doesn't access your games or make an AI request.",
            ephemeral=False, allowed_mentions=discord.AllowedMentions.none(), suppress_embeds=True,
        )

    async def _send_memory_private(self, interaction: discord.Interaction, content: str) -> None:
        await interaction.response.send_message(
            content, ephemeral=True, allowed_mentions=discord.AllowedMentions.none(),
            suppress_embeds=True,
        )

    def _can_manage_memory(self, interaction: discord.Interaction) -> bool:
        return bool(interaction.permissions.manage_guild)

    async def run_memory_status(self, interaction: discord.Interaction) -> None:
        if interaction.guild is None:
            await self._send_memory_private(interaction, "Use this command in a server.")
            return
        if not self.memory.available:
            await self._send_memory_private(
                interaction, "Persistent memory is unavailable because this deployment has no DATABASE_URL."
            )
            return
        try:
            enabled = await self.memory.is_enabled(interaction.guild.id)
            memories = await self.memory.list_memories(interaction.guild.id, limit=15)
        except Exception as exc:
            logger.warning("Could not read memory status for guild %s (%s)", interaction.guild.id, type(exc).__name__)
            await self._send_memory_private(interaction, "Persistent storage is temporarily unavailable.")
            return
        state = "enabled" if enabled else "paused"
        await self._send_memory_private(
            interaction,
            f"**Server memory is {state}.**\nStored memories shown by the list command: {len(memories)}"
            + ("\nRanibot observes future human text in channels it can access and learns in 20-message batches."
               if enabled else "\nSaved memories remain stored but are not applied while memory is paused."),
        )

    async def run_memory_enable(self, interaction: discord.Interaction) -> None:
        if interaction.guild is None:
            await self._send_memory_private(interaction, "Use this command in a server.")
            return
        if not self._can_manage_memory(interaction):
            await self._send_memory_private(interaction, "You need Manage Server to enable persistent memory.")
            return
        if not self.memory.available:
            await self._send_memory_private(
                interaction, "Persistent memory is unavailable because this deployment has no DATABASE_URL."
            )
            return
        try:
            await self.memory.set_enabled(interaction.guild.id, True)
        except Exception as exc:
            logger.warning("Could not enable memory for guild %s (%s)", interaction.guild.id, type(exc).__name__)
            await self._send_memory_private(interaction, "Persistent storage is temporarily unavailable.")
            return
        await self._send_memory_private(
            interaction,
            "**Server memory enabled.** Ranibot will observe future human text in every channel it can access, "
            "extract server-level context in 20-message batches, and retain agent contributions. Tell members "
            "before leaving this enabled. Use `/memory pause` to stop observation and memory use.",
        )

    async def run_memory_pause(self, interaction: discord.Interaction) -> None:
        if interaction.guild is None:
            await self._send_memory_private(interaction, "Use this command in a server.")
            return
        if not self._can_manage_memory(interaction):
            await self._send_memory_private(interaction, "You need Manage Server to pause persistent memory.")
            return
        if not self.memory.available:
            await self._send_memory_private(interaction, "Persistent memory is not configured.")
            return
        try:
            await self.memory.set_enabled(interaction.guild.id, False)
        except Exception as exc:
            logger.warning("Could not pause memory for guild %s (%s)", interaction.guild.id, type(exc).__name__)
            await self._send_memory_private(interaction, "Persistent storage is temporarily unavailable.")
            return
        await self._send_memory_private(
            interaction,
            "**Server memory paused.** Ranibot will not buffer new messages or apply saved memory. "
            "Existing memories remain available for inspection and deletion.",
        )

    async def run_memory_list(self, interaction: discord.Interaction) -> None:
        if interaction.guild is None:
            await self._send_memory_private(interaction, "Use this command in a server.")
            return
        if not self.memory.available:
            await self._send_memory_private(interaction, "Persistent memory is not configured.")
            return
        try:
            memories = await self.memory.list_memories(interaction.guild.id, limit=15)
        except Exception as exc:
            logger.warning("Could not list memories for guild %s (%s)", interaction.guild.id, type(exc).__name__)
            await self._send_memory_private(interaction, "Persistent storage is temporarily unavailable.")
            return
        if not memories:
            await self._send_memory_private(interaction, "This server has no extracted or agent memories yet.")
            return
        lines = ["**Recent server and agent memories**"]
        for memory in memories:
            content = discord.utils.escape_markdown(memory.content.replace("\n", " "))
            if len(content) > 100:
                content = content[:99].rstrip() + "…"
            lines.append(f"`{memory.id}` **{memory.scope}:** {content}")
        lines.append("A member with Manage Server can remove an entry with `/memory forget`.")
        await self._send_memory_private(interaction, "\n".join(lines))

    @app_commands.describe(memory_id="Numeric ID shown by /memory list")
    async def run_memory_forget(
        self, interaction: discord.Interaction, memory_id: app_commands.Range[int, 1],
    ) -> None:
        if interaction.guild is None:
            await self._send_memory_private(interaction, "Use this command in a server.")
            return
        if not self._can_manage_memory(interaction):
            await self._send_memory_private(interaction, "You need Manage Server to delete persistent memory.")
            return
        if not self.memory.available:
            await self._send_memory_private(interaction, "Persistent memory is not configured.")
            return
        try:
            deleted = await self.memory.delete_memory(interaction.guild.id, memory_id)
        except Exception as exc:
            logger.warning("Could not delete memory in guild %s (%s)", interaction.guild.id, type(exc).__name__)
            await self._send_memory_private(interaction, "Persistent storage is temporarily unavailable.")
            return
        await self._send_memory_private(
            interaction, f"Deleted memory `{memory_id}`." if deleted else "That memory ID does not exist in this server."
        )

    @app_commands.describe(confirm="Set true to permanently delete this server's memories and buffered text")
    async def run_memory_clear(self, interaction: discord.Interaction, confirm: bool = False) -> None:
        if interaction.guild is None:
            await self._send_memory_private(interaction, "Use this command in a server.")
            return
        if not self._can_manage_memory(interaction):
            await self._send_memory_private(interaction, "You need Manage Server to clear persistent memory.")
            return
        if not confirm:
            await self._send_memory_private(
                interaction, "Nothing was deleted. Run `/memory clear` with `confirm:True` to continue."
            )
            return
        if not self.memory.available:
            await self._send_memory_private(interaction, "Persistent memory is not configured.")
            return
        try:
            deleted = await self.memory.clear_guild(interaction.guild.id)
        except Exception as exc:
            logger.warning("Could not clear memory for guild %s (%s)", interaction.guild.id, type(exc).__name__)
            await self._send_memory_private(interaction, "Persistent storage is temporarily unavailable.")
            return
        await self._send_memory_private(
            interaction, f"Deleted {deleted} stored memories and cleared any buffered messages for this server."
        )

    async def run_synthesize(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        channel = interaction.channel
        if interaction.guild is None or not isinstance(channel, (discord.TextChannel, discord.Thread)):
            await interaction.edit_original_response(content="Use /synthesize in a server text channel or thread.")
            return
        permissions = interaction.app_permissions
        can_send = permissions.send_messages_in_threads if isinstance(channel, discord.Thread) else permissions.send_messages
        if not (permissions.view_channel and permissions.read_message_history and can_send):
            await interaction.edit_original_response(
                content="I need View Channel, Read Message History, and Send Messages "
                        "(Send Messages in Threads for a thread) here."
            )
            return
        if not interaction.permissions.read_message_history:
            await interaction.edit_original_response(content="You need Read Message History to invoke /synthesize here.")
            return
        if channel.id in self.active_channels:
            await interaction.edit_original_response(content="An AI request is already running in this channel. Try again when it finishes.")
            return

        self.active_channels.add(channel.id)
        try:
            context = await read_context(channel, before=interaction.created_at)
            if context is None:
                await interaction.edit_original_response(content="No readable human text in the last 30 messages to synthesize.")
                return
            logger.info("Synthesizing conversation in channel %s", channel.id)
            result = await synthesize(self.llm, context)
            if result.failed:
                await interaction.edit_original_response(content="Could not synthesize the conversation. Try again later; check the bot logs if this persists.")
            elif not result.text:
                await interaction.edit_original_response(content="The recent conversation is too thin or casual to synthesize honestly.")
            else:
                await channel.send(
                    f"**Synthesis:**\n{result.text}",
                    allowed_mentions=discord.AllowedMentions.none(), suppress_embeds=True,
                )
                await interaction.edit_original_response(content="Posted one synthesis of the recent human conversation.")
        except discord.Forbidden:
            logger.warning("Discord access denied during /synthesize in channel %s", channel.id)
            await interaction.edit_original_response(content="Discord denied access while reading or posting. Check channel permissions.")
        except discord.HTTPException as exc:
            logger.warning("Discord /synthesize request failed in channel %s (HTTP %s)", channel.id, exc.status)
            await interaction.edit_original_response(content="Discord could not complete the synthesis. Check the channel before retrying.")
        finally:
            self.active_channels.discard(channel.id)

    async def run_consent(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_message(
            "**Ranibot data and cost guide**\n"
            "**/agents:** reads up to 30 recent messages, removes bot/webhook/system and empty messages, then sends human usernames, display names, and text to OpenAI in **3 requests**.\n"
            "**/synthesize:** sends that same filtered recent context to OpenAI in **1 request**.\n"
            "**/ask:** sends the typed question in **1 request**; it does not read channel history. **Message actions:** send only selected text in **1 request**. If server memory is enabled, both also send saved server context and that agent's journal.\n"
            "**Memory:** a Manage Server user can enable ambient observation. Ranibot then buffers human text from accessible server channels for at most 7 days. Each 20-message batch is sent to OpenAI in **1 request** to extract durable server-level context; processed raw text is deleted. Agent replies are retained in separate Mira, Hex, and Moss journals. Memories stay in this server's PostgreSQL scope until pruned or deleted. `/memory pause` stops observation and use; list, forget, and clear provide controls.\n"
            "**/status, /help, /consent, /chesslab, and memory controls:** make **0 AI requests**.\n\n"
            "Ranibot never downloads attachments, opens links, browses, or accesses your computer. Generated replies are public; controls and status are private. Provider requests use `store=False`, but OpenAI may retain data under its policies. AI output and extracted memories can be wrong. Tell members before enabling memory, avoid secrets, and remember that AI features use the bot owner's paid API account.",
            ephemeral=True, allowed_mentions=discord.AllowedMentions.none(), suppress_embeds=True,
        )

    async def run_help(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_message(
            "**Ranibot commands**\n"
            "**/agents** — Reads the latest 30 channel messages, filters for human text, and asks Mira, Hex, and Moss independently whether to join in. Any of them may stay silent.\n"
            "**/ask agent question** — Sends your question to one agent and posts a short answer. It reads no channel history.\n"
            "**/status** — Shows uptime and the configured model; makes no AI request.\n"
            "**/chesslab** — Publicly shares the Chess Lab link and introduction; no game access or AI request.\n"
            "**/synthesize** — Sends recent filtered human chat in one AI request and publicly maps common ground, tensions, open questions, and a possible next step.\n"
            "**/memory status|enable|pause|list|forget|clear** — Controls transparent per-server memory. Enable, pause, forget, and clear require Manage Server.\n"
            "**/consent** — Privately explains exactly what Ranibot reads, sends, stores, and costs; no AI request.\n"
            "**/help** — Shows this private guide; makes no AI request.\n"
            "**Message actions** — Right-click a message → Apps to ask one personality about that selected text.\n\n"
            "**Privacy and cost:** AI commands use the bot owner's paid OpenAI account. When memory is enabled, Ranibot observes human messages in channels it can access, sends each 20-message batch to OpenAI for extraction, stores concise server memories and agent journals, and supplies them to future agent requests. Raw processed batches are deleted. Use `/consent` for the complete data flow. Avoid secrets and tell members before enabling memory. Ranibot has no access to your computer and does not open links. AI replies and memories can be wrong.",
            ephemeral=True, allowed_mentions=discord.AllowedMentions.none(), suppress_embeds=True,
        )

    async def on_command_error(self, interaction: discord.Interaction, error: app_commands.AppCommandError) -> None:
        original = getattr(error, "original", error)
        logger.error("Slash command failed (%s)", type(original).__name__)
        try:
            if interaction.response.is_done():
                await interaction.edit_original_response(content="The command failed. Check the bot console and try again.")
            else:
                await interaction.response.send_message("The command failed. Try again later.", ephemeral=True)
        except discord.HTTPException:
            logger.warning("Could not deliver the private error message")


async def main() -> None:
    settings = Settings.from_env()
    async with RaniBot(settings, create_llm(settings), create_memory_store(settings.database_url)) as bot:
        await bot.start(settings.discord_token)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped")
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        raise SystemExit(1)
    except discord.LoginFailure:
        logger.error("Discord login failed. Check DISCORD_BOT_TOKEN in .env.")
        raise SystemExit(1)
    except discord.PrivilegedIntentsRequired:
        logger.error("Enable Message Content Intent on the application's Bot page, then restart.")
        raise SystemExit(1)
    except Exception as exc:
        logger.error("Bot stopped (%s). Check connectivity, credentials, and server installation.", type(exc).__name__)
        raise SystemExit(1)
