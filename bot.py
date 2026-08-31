"""Discord transport for Ranibot and its Agent Terrarium feature."""

import asyncio
import json
import logging
import math
import time
from datetime import datetime
from typing import Literal

import discord
from discord import app_commands

from agents import AGENTS, consider, consult_agents
from config import Settings
from llm import LLM, create_llm

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
    def __init__(self, settings: Settings, llm: LLM):
        # No message event subscription or message cache. History is fetched via REST
        # only inside /agents. Content access also requires the portal intent toggle.
        intents = discord.Intents.none()
        intents.guilds = True
        intents.message_content = True
        super().__init__(intents=intents, max_messages=None,
                         allowed_mentions=discord.AllowedMentions.none())
        self.settings = settings
        self.llm = llm
        self.started_at = time.monotonic()
        self.active_channels: set[int] = set()
        self.tree = app_commands.CommandTree(self)
        for name, description, callback in (
            ("agents", "Let Mira, Hex, and Moss consider joining the conversation.", self.run_agents),
            ("ask", "Ask one agent a question; only that question goes to AI. Reply is public.", self.run_ask),
            ("status", "Show uptime and configured model without making an AI request.", self.run_status),
            ("chesslab", "Share the Chess Lab app link and introduction; no AI request.", self.run_chesslab),
            ("help", "Explain Ranibot's commands and what gets sent to AI.", self.run_help),
        ):
            command = app_commands.Command(name=name, description=description, callback=callback)
            command.guild_only = True
            self.tree.add_command(command)
        self.tree.on_error = self.on_command_error

    async def setup_hook(self) -> None:
        if self.settings.guild_id:
            guild = discord.Object(id=self.settings.guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            logger.info("Synced /agents, /ask, /status, /chesslab, /help to test server %s", guild.id)
        else:
            await self.tree.sync()
            logger.info("Synced /agents, /ask, /status, /chesslab, /help globally")

    async def on_ready(self) -> None:
        logger.info("Ranibot connected as bot ID %s", self.user.id)

    async def close(self) -> None:
        try:
            await super().close()
        finally:
            await self.llm.close()

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
            results = await consult_agents(self.llm, context)
            for result in results:
                if result.text:
                    await channel.send(
                        f"**{result.agent.name}:** {result.text}",
                        allowed_mentions=discord.AllowedMentions.none(), suppress_embeds=True,
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
            result = await consider(selected, self.llm, json.dumps({"question": question}, ensure_ascii=False), direct=True)
            if result.failed:
                await interaction.edit_original_response(content=f"Could not get an answer from {selected.name}. Try again later; check the bot logs if this persists.")
            elif not result.text:
                await interaction.edit_original_response(content=f"{selected.name} did not return an answer. Try rephrasing your question.")
            else:
                await channel.send(f"**{selected.name}:** {result.text}",
                                   allowed_mentions=discord.AllowedMentions.none(), suppress_embeds=True)
                await interaction.edit_original_response(content=f"{selected.name} replied in the channel.")
        except discord.HTTPException as exc:
            logger.warning("Discord /ask request failed in channel %s (HTTP %s)", channel.id, exc.status)
            await interaction.edit_original_response(content="Discord could not complete the reply. Check the channel before retrying, and check my send permissions.")
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

    async def run_help(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_message(
            "**Ranibot commands**\n"
            "**/agents** — Reads the latest 30 channel messages, filters for human text, and asks Mira, Hex, and Moss independently whether to join in. Any of them may stay silent.\n"
            "**/ask agent question** — Sends only your question to one chosen agent and posts a short, labeled answer publicly. No channel history is read.\n"
            "**/status** — Shows uptime and the configured model; makes no AI request.\n"
            "**/chesslab** — Publicly shares the Chess Lab link and introduction; no game access or AI request.\n"
            "**/help** — Shows this private guide; makes no AI request.\n\n"
            "**Privacy and cost:** /agents sends recent usernames and text to OpenAI in three requests. /ask sends your question in one request. Both use paid API usage. Avoid sharing secrets and get participants' agreement before using /agents.\n"
            "The bot has no persistent memory or access to your computer, does not browse links, and reads chat only when /agents is invoked. AI replies can be wrong.",
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
    async with RaniBot(settings, create_llm(settings)) as bot:
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
