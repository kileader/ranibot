import asyncio
import json
import os
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import discord

from agents import AGENTS, consult_agents, parse_contribution, parse_synthesis, synthesize
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
        if system_prompt.startswith("You are Ranibot, an AI facilitator"):
            name = "Synthesis"
        elif "selected one Discord message" in system_prompt:
            name = next(agent.name for agent in AGENTS if system_prompt.startswith(f"You are {agent.name},"))
        else:
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
                slash = bot.tree.get_commands(type=discord.AppCommandType.chat_input)
                messages = bot.tree.get_commands(type=discord.AppCommandType.message)
                self.assertEqual({c.name for c in slash}, {"agents", "ask", "status", "chesslab", "synthesize", "consent", "help"})
                self.assertEqual({c.name for c in messages}, {"Ask Mira about this", "Analyze with Hex", "Connect with Moss"})
                self.assertEqual(bot.tree.get_command("chesslab").parameters, [])
                self.assertEqual(bot.tree.get_command("synthesize").parameters, [])
                self.assertEqual(bot.tree.get_command("consent").parameters, [])
                self.assertTrue(all(c.guild_only for c in slash))
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
                    self.assertEqual(
                        {c.name for c in bot.tree.get_commands(guild=guild, type=discord.AppCommandType.chat_input)},
                        {"agents", "ask", "status", "chesslab", "synthesize", "consent", "help"},
                    )
                    self.assertEqual(
                        {c.name for c in bot.tree.get_commands(guild=guild, type=discord.AppCommandType.message)},
                        {"Ask Mira about this", "Analyze with Hex", "Connect with Moss"},
                    )
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

    async def test_message_actions_send_only_selected_text_to_one_personality(self):
        self.llm.replies.update({"Mira": "Possibility.", "Hex": "Analysis.", "Moss": "Connection."})
        for method, expected in [
            (self.bot.message_mira, "Mira"),
            (self.bot.message_hex, "Hex"),
            (self.bot.message_moss, "Moss"),
        ]:
            with self.subTest(agent=expected):
                interaction = interaction_for(self.channel)
                target = MagicMock(spec=discord.Message)
                target.clean_content = "  selected text only  "
                target.reply = AsyncMock()
                await method(interaction, target)
                name, context = self.llm.calls[-1]
                self.assertEqual(name, expected)
                self.assertEqual(context, '{"selected_message": "selected text only"}')
                self.channel.history.assert_not_called()
                target.reply.assert_awaited_once()
                post = target.reply.call_args
                self.assertTrue(post.args[0].startswith(f"**{expected}:** "))
                self.assertFalse(post.kwargs["mention_author"])
                self.assertTrue(post.kwargs["suppress_embeds"])
                self.assertEqual(post.kwargs["allowed_mentions"].to_dict()["parse"], [])
                self.assertFalse(self.bot.active_channels)
        self.assertEqual(len(self.llm.calls), 3)

    async def test_message_action_rejects_empty_or_denied_message_without_ai(self):
        target = MagicMock(spec=discord.Message)
        target.clean_content = "   "
        target.reply = AsyncMock()
        await self.bot.message_mira(self.interaction, target)
        denied = interaction_for(self.channel)
        denied.app_permissions.send_messages = False
        target.clean_content = "selected"
        await self.bot.message_mira(denied, target)
        self.assertEqual(self.llm.calls, [])
        self.channel.history.assert_not_called()
        target.reply.assert_not_awaited()

    async def test_message_action_failure_and_guard_never_post_fallback(self):
        target = MagicMock(spec=discord.Message)
        target.clean_content = "selected"
        target.reply = AsyncMock()
        self.llm.replies["Hex"] = RuntimeError("secret-sensitive-body")
        with self.assertLogs("agents", level="WARNING") as logs:
            await self.bot.message_hex(self.interaction, target)
        status = self.interaction.edit_original_response.call_args.kwargs["content"]
        self.assertIn("Could not get a response", status)
        self.assertNotIn("secret-sensitive-body", status + str(logs.output))
        target.reply.assert_not_awaited()
        self.assertFalse(self.bot.active_channels)
        self.bot.active_channels.add(self.channel.id)
        second = interaction_for(self.channel)
        await self.bot.message_mira(second, target)
        self.assertIn("already running", second.edit_original_response.call_args.kwargs["content"])
        self.assertEqual(len(self.llm.calls), 1)
        self.assertEqual(self.bot.active_channels, {self.channel.id})

    async def test_message_action_truncates_selected_text(self):
        self.llm.replies["Mira"] = "Response."
        target = MagicMock(spec=discord.Message)
        target.clean_content = "x" * 2000
        target.reply = AsyncMock()
        await self.bot.message_mira(self.interaction, target)
        context = json.loads(self.llm.calls[0][1])
        self.assertEqual(len(context["selected_message"]), 1512)
        self.assertTrue(context["selected_message"].endswith(" [truncated]"))

    async def test_synthesize_uses_filtered_context_once_and_posts_safely(self):
        self.llm.replies["Synthesis"] = "**Common ground:** Measure delayed retention.\n**Open questions:** Which material?"
        await self.bot.run_synthesize(self.interaction)
        self.assertEqual(len(self.llm.calls), 1)
        name, context = self.llm.calls[0]
        self.assertEqual(name, "Synthesis")
        self.assertNotIn("private-token", context)
        self.channel.history.assert_called_once()
        self.channel.send.assert_awaited_once()
        post = self.channel.send.call_args
        self.assertTrue(post.args[0].startswith("**Synthesis:**\n"))
        self.assertEqual(post.kwargs["allowed_mentions"].to_dict()["parse"], [])
        self.assertTrue(post.kwargs["suppress_embeds"])
        self.interaction.response.defer.assert_awaited_once_with(ephemeral=True, thinking=True)
        self.assertFalse(self.bot.active_channels)

    async def test_synthesize_insufficient_context_posts_nothing(self):
        self.llm.replies["Synthesis"] = "INSUFFICIENT"
        await self.bot.run_synthesize(self.interaction)
        self.channel.send.assert_not_awaited()
        self.assertIn("too thin or casual", self.interaction.edit_original_response.call_args.kwargs["content"])
        self.assertFalse(self.bot.active_channels)

    async def test_synthesize_empty_or_denied_context_never_calls_ai(self):
        async def empty_history(**kwargs):
            if False:
                yield None
        self.channel.history.side_effect = empty_history
        await self.bot.run_synthesize(self.interaction)
        self.assertEqual(self.llm.calls, [])
        self.channel.history.reset_mock()
        denied = interaction_for(self.channel)
        denied.app_permissions.read_message_history = False
        await self.bot.run_synthesize(denied)
        self.channel.history.assert_not_called()
        self.assertEqual(self.llm.calls, [])

    async def test_synthesize_failure_is_private_and_releases_channel(self):
        self.llm.replies["Synthesis"] = RuntimeError("secret-sensitive-body")
        with self.assertLogs("agents", level="WARNING") as logs:
            await self.bot.run_synthesize(self.interaction)
        self.channel.send.assert_not_awaited()
        status = self.interaction.edit_original_response.call_args.kwargs["content"]
        self.assertIn("Could not synthesize", status)
        self.assertNotIn("secret-sensitive-body", status + str(logs.output))
        self.assertFalse(self.bot.active_channels)

    async def test_synthesize_timeout_returns_failure(self):
        async def generate(*args):
            await asyncio.Event().wait()
        with patch("agents.AGENT_TIMEOUT_SECONDS", 0.02), self.assertLogs("agents", level="WARNING"):
            result = await synthesize(SimpleNamespace(generate=generate), "[]")
        self.assertTrue(result.failed)
        self.assertIsNone(result.text)

    async def test_synthesize_shares_ai_channel_guard(self):
        self.bot.active_channels.add(self.channel.id)
        await self.bot.run_synthesize(self.interaction)
        self.assertIn("already running", self.interaction.edit_original_response.call_args.kwargs["content"])
        self.channel.history.assert_not_called()
        self.assertEqual(self.llm.calls, [])
        self.assertEqual(self.bot.active_channels, {self.channel.id})

    async def test_consent_is_private_complete_and_uses_no_ai_or_history(self):
        await self.bot.run_consent(self.interaction)
        post = self.interaction.response.send_message.call_args
        text = post.args[0]
        self.assertTrue(post.kwargs["ephemeral"])
        self.assertTrue(post.kwargs["suppress_embeds"])
        self.assertEqual(post.kwargs["allowed_mentions"].to_dict()["parse"], [])
        for phrase in ("**3 requests**", "**1 request**", "selected message text", "store=False", "paid API account", "does not download attachments"):
            self.assertIn(phrase, text)
        self.assertLess(len(text), 2000)
        self.assertEqual(self.llm.calls, [])
        self.channel.history.assert_not_called()
        self.channel.send.assert_not_awaited()

    async def test_chesslab_shares_public_link_without_ai_history_or_account_access(self):
        self.interaction.app_permissions.read_message_history = False
        self.interaction.permissions.read_message_history = False
        self.bot.active_channels.add(self.channel.id)
        await self.bot.run_chesslab(self.interaction)
        self.interaction.response.send_message.assert_awaited_once()
        post = self.interaction.response.send_message.call_args
        self.assertFalse(post.kwargs["ephemeral"])
        self.assertTrue(post.kwargs["suppress_embeds"])
        self.assertEqual(post.kwargs["allowed_mentions"].to_dict()["parse"], [])
        self.assertIn("https://chess-lab-zeta.vercel.app", post.args[0])
        self.assertIn("Sign in with Google", post.args[0])
        self.assertIn("library stays private", post.args[0])
        self.assertIn("doesn't access your games", post.args[0])
        self.assertNotIn("private-token", post.args[0])
        self.assertNotIn("private-key", post.args[0])
        self.assertLess(len(post.args[0]), 2000)
        self.assertEqual(self.llm.calls, [])
        self.channel.history.assert_not_called()
        self.channel.send.assert_not_awaited()
        self.interaction.response.defer.assert_not_awaited()
        self.assertEqual(self.bot.active_channels, {self.channel.id})

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
        help_text = self.interaction.response.send_message.call_args.args[0]
        self.assertIn("**/chesslab**", help_text)
        self.assertIn("**/synthesize**", help_text)
        self.assertIn("**/consent**", help_text)
        self.assertIn("**Message actions**", help_text)
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
        for value in ("INSUFFICIENT", " insufficient. ", "`INSUFFICIENT`", ""):
            self.assertIsNone(parse_synthesis(value))
        synthesis = parse_synthesis("🌱" * 3000)
        self.assertLessEqual(len(synthesis), 1500)
        self.assertLess(len(f"**Synthesis:**\n{synthesis}"), 2000)

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
