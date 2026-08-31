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
        response=SimpleNamespace(defer=AsyncMock(), send_message=AsyncMock()),
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
                self.assertEqual({c.name for c in bot.tree.get_commands()}, {"agents", "ask", "status", "help"})
                self.assertTrue(all(c.guild_only for c in bot.tree.get_commands()))
                ask_options = bot.tree.get_command("ask").to_dict(bot.tree)["options"]
                self.assertEqual([c["value"] for c in ask_options[0]["choices"]], ["Mira", "Hex", "Moss"])
                self.assertEqual((ask_options[1]["min_length"], ask_options[1]["max_length"]), (1, 1500))
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
                    self.assertEqual({c.name for c in bot.tree.get_commands(guild=guild)}, {"agents", "ask", "status", "help"})
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


class UtilityCommandTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.llm = FakeLLM({"Hex": "@everyone Compare retention after a week."})
        self.bot = RaniBot(Settings("private-token", "private-key", llm_model="test-model"), self.llm)
        self.channel = channel_with([message("private channel history")])
        self.interaction = interaction_for(self.channel)

    async def asyncTearDown(self):
        await self.bot.close()

    async def test_ask_sends_only_question_to_selected_agent_and_posts_safely(self):
        self.interaction.app_permissions.read_message_history = False
        self.interaction.permissions.read_message_history = False
        self.llm.generate = AsyncMock(wraps=self.llm.generate)
        await self.bot.run_ask(self.interaction, "Hex", "  How should I test retention?  ")
        self.assertEqual(self.llm.calls, [("Hex", '{"question": "How should I test retention?"}')])
        prompt = self.llm.generate.call_args.args[0]
        self.assertIn("address you directly", prompt)
        self.assertNotIn("permission to consider joining", prompt)
        self.channel.history.assert_not_called()
        self.channel.send.assert_awaited_once()
        post = self.channel.send.call_args
        self.assertTrue(post.args[0].startswith("**Hex:** "))
        self.assertEqual(post.kwargs["allowed_mentions"].to_dict()["parse"], [])
        self.assertTrue(post.kwargs["suppress_embeds"])
        self.interaction.response.defer.assert_awaited_once_with(ephemeral=True, thinking=True)
        self.assertFalse(self.bot.active_channels)

    async def test_ask_rejects_bad_input_without_ai_or_history(self):
        for agent, question in [("Hex", "   "), ("Hex", "x" * 1501), ("Unknown", "Hi")]:
            with self.subTest(agent=agent, length=len(question)):
                await self.bot.run_ask(self.interaction, agent, question)
        self.assertEqual(self.llm.calls, [])
        self.channel.history.assert_not_called()
        self.channel.send.assert_not_awaited()
        self.assertFalse(self.bot.active_channels)

    async def test_ask_checks_channel_and_send_permissions_before_ai(self):
        for permission in ["view_channel", "send_messages"]:
            with self.subTest(permission=permission):
                interaction = interaction_for(self.channel)
                setattr(interaction.app_permissions, permission, False)
                await self.bot.run_ask(interaction, "Hex", "Question?")
        interaction = interaction_for(self.channel)
        interaction.guild = None
        await self.bot.run_ask(interaction, "Hex", "Question?")
        self.assertEqual(self.llm.calls, [])
        self.channel.history.assert_not_called()
        self.channel.send.assert_not_awaited()

    async def test_ask_thread_uses_thread_send_permission(self):
        channel = MagicMock(spec=discord.Thread)
        channel.id = 51
        channel.send = AsyncMock()
        interaction = interaction_for(channel)
        interaction.app_permissions.send_messages = False
        await self.bot.run_ask(interaction, "Hex", "Question?")
        self.assertEqual(self.llm.calls, [])
        interaction.app_permissions.send_messages_in_threads = True
        await self.bot.run_ask(interaction, "Hex", "Question?")
        channel.send.assert_awaited_once()
        channel.history.assert_not_called()
        self.assertEqual(len(self.llm.calls), 1)

    async def test_ask_failure_is_private_and_releases_channel(self):
        self.llm.replies["Hex"] = RuntimeError("secret-sensitive-body")
        with self.assertLogs("agents", level="WARNING") as logs:
            await self.bot.run_ask(self.interaction, "Hex", "Question?")
        self.channel.send.assert_not_awaited()
        status = self.interaction.edit_original_response.call_args.kwargs["content"]
        self.assertIn("Could not get an answer", status)
        self.assertNotIn("secret-sensitive-body", status + str(logs.output))
        self.assertFalse(self.bot.active_channels)

    async def test_ask_timeout_and_silence_never_post_public_fallbacks(self):
        async def hanging(*args):
            await asyncio.Event().wait()
        self.llm.generate = hanging
        with patch("agents.AGENT_TIMEOUT_SECONDS", 0.02), self.assertLogs("agents", level="WARNING"):
            await self.bot.run_ask(self.interaction, "Hex", "Question?")
        self.assertFalse(self.bot.active_channels)
        self.llm.generate = AsyncMock(return_value="SILENT")
        await self.bot.run_ask(self.interaction, "Hex", "Question?")
        self.channel.send.assert_not_awaited()
        self.assertIn("did not return an answer", self.interaction.edit_original_response.call_args.kwargs["content"])
        self.assertFalse(self.bot.active_channels)

    async def test_ask_discord_failure_releases_channel_without_retrying_ai(self):
        self.channel.send.side_effect = discord.Forbidden(SimpleNamespace(status=403, reason="Forbidden"), "Denied")
        with self.assertLogs("bot", level="WARNING"):
            await self.bot.run_ask(self.interaction, "Hex", "Question?")
        self.assertFalse(self.bot.active_channels)
        self.assertEqual(len(self.llm.calls), 1)
        self.assertIn("permissions", self.interaction.edit_original_response.call_args.kwargs["content"])

    async def test_ask_and_agents_share_channel_guard(self):
        started, release = asyncio.Event(), asyncio.Event()
        async def generate(*args):
            started.set()
            await release.wait()
            return "SILENT"
        self.llm.generate = AsyncMock(side_effect=generate)
        first = asyncio.create_task(self.bot.run_ask(self.interaction, "Hex", "Question?"))
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            second = interaction_for(self.channel)
            await self.bot.run_ask(second, "Mira", "Another question?")
            self.assertIn("already running", second.edit_original_response.call_args.kwargs["content"])
            await self.bot.run_agents(second)
            self.assertIn("already considering", second.edit_original_response.call_args.kwargs["content"])
            self.channel.history.assert_not_called()
            self.assertEqual(self.llm.generate.await_count, 1)
        finally:
            release.set()
            await first
        self.assertFalse(self.bot.active_channels)

    async def test_status_and_help_are_private_without_ai_or_history(self):
        self.bot.started_at = 100
        with patch("bot.time.monotonic", return_value=90161):
            await self.bot.run_status(self.interaction)
        status = self.interaction.response.send_message.call_args.args[0]
        self.assertIn("1d 1h 1m 1s", status)
        self.assertIn("test-model", status)
        self.assertIn("not measured yet", status)
        self.assertIn("does not check API billing", status)
        await self.bot.run_help(self.interaction)
        for call in self.interaction.response.send_message.call_args_list:
            self.assertTrue(call.kwargs["ephemeral"])
            self.assertEqual(call.kwargs["allowed_mentions"].to_dict()["parse"], [])
            self.assertNotIn("private-token", call.args[0])
            self.assertNotIn("private-key", call.args[0])
            self.assertLess(len(call.args[0]), 2000)
        self.assertEqual(self.llm.calls, [])
        self.channel.history.assert_not_called()
        self.channel.send.assert_not_awaited()


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
