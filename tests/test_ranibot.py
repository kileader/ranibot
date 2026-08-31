import asyncio
import json
import os
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import discord

from agents import AGENTS, consult_agents, parse_contribution
from bot import RaniBot, read_context
from config import Settings
from llm import OpenAILLM


def message(text, username="kevin", *, bot=False, system=False, webhook_id=None):
    return SimpleNamespace(
        clean_content=text,
        author=SimpleNamespace(name=username, display_name=username.title(), bot=bot),
        webhook_id=webhook_id,
        is_system=lambda: system,
    )


def channel_with(messages):
    channel = MagicMock(spec=discord.TextChannel)
    channel.id = 42

    async def history(**kwargs):
        for item in messages:
            yield item

    channel.history.side_effect = history
    channel.send = AsyncMock()
    return channel


def interaction_for(channel):
    return SimpleNamespace(
        channel=channel,
        guild=SimpleNamespace(id=1),
        created_at=datetime.now(timezone.utc),
        app_permissions=discord.Permissions(view_channel=True, read_message_history=True, send_messages=True),
        permissions=discord.Permissions(read_message_history=True),
        response=SimpleNamespace(defer=AsyncMock()),
        edit_original_response=AsyncMock(),
    )


class FakeLLM:
    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    async def generate(self, system_prompt, context):
        name = next(agent.name for agent in AGENTS if system_prompt.startswith(f"You are {agent.name},"))
        self.calls.append((name, context))
        result = self.replies[name]
        if isinstance(result, Exception):
            raise result
        return result

    async def close(self):
        pass


class ContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_recent_snapshot_filters_and_orders_human_text(self):
        channel = channel_with([
            message("newest\nwith a second line", "lee"),
            message("bot text", bot=True),
            message("join event", system=True),
            message("webhook text", webhook_id=123),
            message("   "),
            message("oldest", "sam"),
        ])
        before = datetime.now(timezone.utc)
        context = json.loads(await read_context(channel, before))
        self.assertEqual([item["username"] for item in context], ["sam", "lee"])
        self.assertEqual(context[1]["text"], "newest\nwith a second line")
        self.assertEqual(context[0]["display_name"], "Sam")
        channel.history.assert_called_once_with(limit=30, before=before, oldest_first=False)


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_slash_command_registration_and_no_passive_message_subscription(self):
        for guild_id in (None, 123):
            with self.subTest(guild_id=guild_id):
                bot = RaniBot(Settings("unused", "unused", guild_id=guild_id), FakeLLM({}))
                command = bot.tree.get_command("agents")
                self.assertTrue(command.guild_only)
                self.assertEqual(command.parameters, [])
                self.assertFalse(bot.intents.messages)
                self.assertTrue(bot.intents.message_content)
                self.assertEqual(len(bot.cached_messages), 0)
                bot.tree.sync = AsyncMock()
                await bot.setup_hook()
                if guild_id:
                    guild = bot.tree.sync.call_args.kwargs["guild"]
                    self.assertEqual(guild.id, guild_id)
                    self.assertIsNotNone(bot.tree.get_command("agents", guild=guild))
                else:
                    bot.tree.sync.assert_awaited_once_with()
                await bot.close()

    async def run_workflow(self, replies, messages=None):
        llm = FakeLLM(replies)
        bot = RaniBot(Settings("unused", "unused"), llm)
        channel = channel_with(messages if messages is not None else [message("What would we measure?")])
        interaction = interaction_for(channel)
        await bot.run_agents(interaction)
        self.assertFalse(bot.active_channels)
        await bot.close()
        return llm, channel, interaction

    async def test_mixed_decisions_post_only_labeled_contributions(self):
        llm, channel, interaction = await self.run_workflow({
            "Mira": "What changes if we measure retention a week later?",
            "Hex": "SILENT", "Moss": "@everyone Treat the schedule like a garden experiment.",
        })
        self.assertEqual(len(llm.calls), 3)
        self.assertEqual(len({context for _, context in llm.calls}), 1)
        posts = channel.send.call_args_list
        self.assertEqual(len(posts), 2)
        self.assertTrue(posts[0].args[0].startswith("**Mira:** "))
        self.assertTrue(posts[1].args[0].startswith("**Moss:** "))
        for post in posts:
            self.assertEqual(post.kwargs["allowed_mentions"].to_dict()["parse"], [])
            self.assertTrue(post.kwargs["suppress_embeds"])
        interaction.response.defer.assert_awaited_once_with(ephemeral=True, thinking=True)

    async def test_all_silent_posts_nothing_publicly(self):
        _, channel, interaction = await self.run_workflow(dict.fromkeys((a.name for a in AGENTS), "SILENT"))
        channel.send.assert_not_awaited()
        self.assertIn("All three agents chose silence", interaction.edit_original_response.call_args.kwargs["content"])

    async def test_empty_context_never_calls_provider(self):
        llm, channel, _ = await self.run_workflow({}, [message("a bot", bot=True), message("")])
        self.assertEqual(llm.calls, [])
        channel.send.assert_not_awaited()

    async def test_failure_does_not_silence_other_agents_or_leak_error_body(self):
        with self.assertLogs("agents", level="WARNING") as logs:
            _, channel, interaction = await self.run_workflow({
                "Mira": "Try a delayed recall test.", "Hex": RuntimeError("secret-sensitive-body"), "Moss": "SILENT",
            })
        channel.send.assert_awaited_once()
        self.assertNotIn("secret-sensitive-body", " ".join(logs.output))
        status = interaction.edit_original_response.call_args.kwargs["content"]
        self.assertIn("Hex", status)
        self.assertNotIn("secret-sensitive-body", status)

    async def test_concurrent_command_is_rejected_without_second_context_read(self):
        started, release = asyncio.Event(), asyncio.Event()
        llm = FakeLLM({})

        async def generate(*args):
            started.set()
            await release.wait()
            return "SILENT"

        llm.generate = generate
        bot = RaniBot(Settings("unused", "unused"), llm)
        channel = channel_with([message("Is this useful?")])
        first = asyncio.create_task(bot.run_agents(interaction_for(channel)))
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            second = interaction_for(channel)
            await bot.run_agents(second)
            self.assertIn("already considering", second.edit_original_response.call_args.kwargs["content"])
            channel.history.assert_called_once()
        finally:
            release.set()
            await first
            await bot.close()
        self.assertFalse(bot.active_channels)

    async def test_missing_permissions_prevent_history_and_llm_calls(self):
        for target in ("app_permissions", "permissions"):
            with self.subTest(target=target):
                llm = FakeLLM({})
                bot = RaniBot(Settings("unused", "unused"), llm)
                channel = channel_with([])
                interaction = interaction_for(channel)
                getattr(interaction, target).read_message_history = False
                await bot.run_agents(interaction)
                channel.history.assert_not_called()
                self.assertEqual(llm.calls, [])
                await bot.close()

    async def test_discord_forbidden_releases_channel(self):
        llm = FakeLLM({})
        bot = RaniBot(Settings("unused", "unused"), llm)
        channel = channel_with([])
        channel.history.side_effect = discord.Forbidden(SimpleNamespace(status=403, reason="Forbidden"), "Denied")
        interaction = interaction_for(channel)
        with self.assertLogs("bot", level="WARNING"):
            await bot.run_agents(interaction)
        self.assertFalse(bot.active_channels)
        self.assertIn("denied access", interaction.edit_original_response.call_args.kwargs["content"])
        await bot.close()

    async def test_timeout_does_not_cancel_successful_agent(self):
        async def generate(system_prompt, context):
            if system_prompt.startswith("You are Hex,"):
                await asyncio.Event().wait()
            return "SILENT"

        with patch("agents.AGENT_TIMEOUT_SECONDS", 0.02), self.assertLogs("agents", level="WARNING"):
            results = await consult_agents(SimpleNamespace(generate=generate), "[]")
        self.assertEqual([r.agent.name for r in results if r.failed], ["Hex"])


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_responses_request_is_stateless_and_has_no_tools(self):
        with patch("llm.AsyncOpenAI") as sdk:
            sdk.return_value.responses.create = AsyncMock(return_value=SimpleNamespace(status="completed", output_text="SILENT"))
            sdk.return_value.close = AsyncMock()
            llm = OpenAILLM("test-key", "test-model")
            self.assertEqual(await llm.generate("system", "context"), "SILENT")
            request = sdk.return_value.responses.create.call_args.kwargs
            self.assertEqual(request["instructions"], "system")
            self.assertEqual(request["input"], "context")
            self.assertEqual(request["model"], "test-model")
            self.assertFalse(request["store"])
            self.assertNotIn("tools", request)
            self.assertNotIn("previous_response_id", request)
            sdk.return_value.responses.create.return_value.status = "incomplete"
            with self.assertRaises(RuntimeError):
                await llm.generate("system", "context")
            sdk.return_value.responses.create.return_value = SimpleNamespace(status="completed", output_text="")
            with self.assertRaises(RuntimeError):
                await llm.generate("system", "context")
            await llm.close()
            sdk.return_value.close.assert_awaited_once()


class ConfigurationAndOutputTests(unittest.TestCase):
    def test_silence_and_safe_output_length(self):
        for value in ("SILENT", " silent. ", "`SILENT`", "**SILENT**", ""):
            self.assertIsNone(parse_contribution(value))
        self.assertEqual(parse_contribution("The silent treatment isn't a measurement."), "The silent treatment isn't a measurement.")
        text = parse_contribution("🌱" * 2000)
        self.assertLessEqual(len(text), 800)
        self.assertLess(len(f"**Mira:** {text}".encode("utf-16-le")) // 2, 2000)

    @patch("config.load_dotenv")
    def test_configuration_validation_does_not_expose_credentials(self, _):
        env = {"DISCORD_BOT_TOKEN": "private-token", "LLM_API_KEY": "private-key"}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()
            self.assertNotIn("private-token", repr(settings))
            self.assertNotIn("private-key", repr(settings))
            os.environ["DISCORD_GUILD_ID"] = "bad-id"
            with self.assertRaisesRegex(ValueError, "DISCORD_GUILD_ID"):
                Settings.from_env()
            os.environ["DISCORD_GUILD_ID"] = ""
            os.environ["LLM_PROVIDER"] = "unsupported"
            with self.assertRaisesRegex(ValueError, "LLM_PROVIDER"):
                Settings.from_env()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "DISCORD_BOT_TOKEN"):
                Settings.from_env()


if __name__ == "__main__":
    unittest.main()
