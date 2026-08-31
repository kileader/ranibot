"""Discord transport for Ranibot and its Agent Terrarium feature."""

import asyncio
import json
import logging
from datetime import datetime

import discord
from discord import app_commands

from agents import consult_agents
from config import Settings
from llm import LLM, create_llm

logger = logging.getLogger(__name__)
HISTORY_LIMIT = 30
MAX_MESSAGE_CHARS = 1500


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
        self.active_channels: set[int] = set()
        self.tree = app_commands.CommandTree(self)
        command = app_commands.Command(
            name="agents", description="Let Mira, Hex, and Moss consider joining the conversation.",
            callback=self.run_agents,
        )
        command.guild_only = True
        self.tree.add_command(command)
        self.tree.on_error = self.on_command_error

    async def setup_hook(self) -> None:
        if self.settings.guild_id:
            guild = discord.Object(id=self.settings.guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            logger.info("Synced /agents to test server %s", guild.id)
        else:
            await self.tree.sync()
            logger.info("Synced /agents globally")

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
