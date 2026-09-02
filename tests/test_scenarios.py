import json
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import discord

from bot import RaniBot, read_scenario_discussion
from config import Settings
from scenarios import (
    DEEPEN_PROMPT,
    SCENARIO_PROMPT,
    Feed,
    Scenario,
    Story,
    article_url_allowed,
    deepen_scenario,
    format_scenario,
    generate_scenario,
    parse_deepening,
    parse_feed,
    parse_scenario,
)


class StubLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    async def generate(self, prompt, context):
        self.calls.append((prompt, context))
        return self.reply

    async def close(self):
        pass


class ScenarioLogicTests(unittest.IsolatedAsyncioTestCase):
    def test_scenario_and_deepening_parsers_validate_and_bound_output(self):
        scenario = parse_scenario('```json\n{"title":"A", "premise":"P", "question":"Q?"}\n```')
        self.assertEqual(scenario, Scenario("A", "P", "Q?"))
        selected = parse_deepening(json.dumps({"agent": "Hex", "text": "Check the base rate."}))
        self.assertEqual((selected.agent_name, selected.text), ("Hex", "Check the base rate."))
        with self.assertRaises(ValueError):
            parse_deepening('{"agent":"Ranibot","text":"No."}')
        with self.assertRaises((ValueError, json.JSONDecodeError)):
            parse_scenario("not json")

    async def test_generation_uses_source_and_deepening_is_one_selected_call(self):
        source = Story(
            "Implant trial begins", "https://news.mit.edu/example", "A small safety trial began.",
            "MIT News", "Cybernetics", datetime(2026, 9, 1, tzinfo=timezone.utc),
        )
        llm = StubLLM('{"title":"Who controls the update?","premise":"A trial began. Suppose adoption expands.","question":"What happens next?"}')
        scenario = await generate_scenario(llm, source)
        self.assertEqual(scenario.title, "Who controls the update?")
        self.assertEqual(llm.calls[0][0], SCENARIO_PROMPT)
        sent = json.loads(llm.calls[0][1])
        self.assertEqual(sent["source_type"], "article")
        self.assertEqual(sent["source_summary"], "A small safety trial began.")
        self.assertEqual(sent["source_url"], source.url)

        llm.reply = '{"agent":"Mira","text":"What changes if updates become a public utility?"}'
        result = await deepen_scenario(
            llm, "Scenario text", json.dumps([{"username": "k", "text": "Regulate it."}]),
            ["The server values individual autonomy."],
        )
        self.assertEqual(result.agent_name, "Mira")
        self.assertEqual(llm.calls[1][0], DEEPEN_PROMPT)
        deepen_input = json.loads(llm.calls[1][1])
        self.assertEqual(deepen_input["human_discussion"][0]["text"], "Regulate it.")

    def test_format_keeps_source_visible_and_mentions_inert_at_transport(self):
        output = format_scenario(
            Scenario("Memory markets", "A real trial exists. Suppose it scales.", "Who benefits?"),
            Story("Study [results]", "https://news.mit.edu/example"),
        )
        self.assertTrue(output.startswith("**Future Scenario:"))
        self.assertIn("**Question:** Who benefits?", output)
        self.assertIn("https://news.mit.edu/example", output)
        self.assertLess(len(output), 2000)

    def test_curated_url_allowlist_rejects_ports_credentials_and_other_hosts(self):
        self.assertTrue(article_url_allowed("https://news.mit.edu/2026/example"))
        self.assertTrue(article_url_allowed("https://www.nature.com/articles/example"))
        self.assertFalse(article_url_allowed("http://news.mit.edu/example"))
        self.assertFalse(article_url_allowed("https://news.mit.edu:8443/example"))
        self.assertFalse(article_url_allowed("https://user@news.mit.edu/example"))
        self.assertFalse(article_url_allowed("https://example.com/news"))

    def test_rss_and_atom_entries_are_parsed_without_html(self):
        feed = Feed("AI", "MIT News", "https://news.mit.edu/feed")
        rss = b'''<rss><channel><item><title>New AI system &amp; useful</title>
        <link>https://news.mit.edu/2026/useful</link><description>&lt;b&gt;Summary&lt;/b&gt;</description>
        <pubDate>Tue, 01 Sep 2026 12:00:00 GMT</pubDate></item></channel></rss>'''
        stories = parse_feed(rss, feed)
        self.assertEqual(len(stories), 1)
        self.assertEqual(stories[0].title, "New AI system & useful")
        self.assertEqual(stories[0].summary, "Summary")
        self.assertEqual(stories[0].published.year, 2026)


class ScenarioDiscordTests(unittest.IsolatedAsyncioTestCase):
    async def test_discussion_reader_stops_at_latest_scenario_and_filters_humans(self):
        channel = MagicMock(spec=discord.TextChannel)
        channel.id = 42

        def item(text, author_id, *, is_bot=False, webhook=None):
            return SimpleNamespace(
                content=text, clean_content=text,
                author=SimpleNamespace(id=author_id, name=f"u{author_id}", display_name=f"U{author_id}", bot=is_bot),
                webhook_id=webhook, is_system=lambda: False,
            )

        messages = [
            item("newest point", 2),
            item("agent reply", 99, is_bot=True),
            item("first point", 1),
            item("**Future Scenario: Test**\nPremise", 99, is_bot=True),
            item("older unrelated chat", 3),
        ]

        async def history(**kwargs):
            for message in messages:
                yield message

        channel.history.side_effect = history
        scenario, discussion, count = await read_scenario_discussion(
            channel, datetime.now(timezone.utc), 99,
        )
        self.assertEqual(scenario, "**Future Scenario: Test**\nPremise")
        self.assertEqual(count, 2)
        self.assertEqual(
            [entry["text"] for entry in json.loads(discussion)],
            ["first point", "newest point"],
        )

    def make_interaction(self, channel, *, create_threads=True):
        return SimpleNamespace(
            channel=channel,
            guild=SimpleNamespace(id=1),
            user=SimpleNamespace(id=7),
            created_at=datetime.now(timezone.utc),
            app_permissions=discord.Permissions(
                view_channel=True, read_message_history=True, send_messages=True,
                create_public_threads=create_threads, send_messages_in_threads=True,
            ),
            permissions=discord.Permissions(read_message_history=True),
            response=SimpleNamespace(defer=AsyncMock(), send_message=AsyncMock()),
            edit_original_response=AsyncMock(),
        )

    async def test_default_scenario_posts_then_creates_public_thread(self):
        llm = StubLLM("unused")
        bot = RaniBot(Settings("unused", "unused"), llm)
        channel = MagicMock(spec=discord.TextChannel)
        channel.id = 42
        starter = SimpleNamespace(create_thread=AsyncMock(
            return_value=SimpleNamespace(mention="#future-thread")
        ))
        channel.send = AsyncMock(return_value=starter)
        interaction = self.make_interaction(channel)
        source = Story("An implant trial", "https://news.mit.edu/example", "A trial began.")
        generated = Scenario("Who owns the update?", "A trial began. Suppose access expands.", "Who decides?")

        with patch("bot.generate_scenario", AsyncMock(return_value=generated)):
            await bot._start_scenario(interaction, source, in_channel=False)

        channel.send.assert_awaited_once()
        starter.create_thread.assert_awaited_once_with(
            name="Who owns the update?", auto_archive_duration=1440,
        )
        post = channel.send.call_args
        self.assertIn("**Future Scenario:", post.args[0])
        self.assertEqual(post.kwargs["allowed_mentions"].to_dict()["parse"], [])
        self.assertIn("#future-thread", interaction.edit_original_response.call_args.kwargs["content"])
        self.assertFalse(bot.active_channels)
        await bot.close()

    async def test_missing_thread_permission_fails_before_ai(self):
        llm = StubLLM("unused")
        bot = RaniBot(Settings("unused", "unused"), llm)
        channel = MagicMock(spec=discord.TextChannel)
        channel.id = 42
        channel.send = AsyncMock()
        interaction = self.make_interaction(channel, create_threads=False)

        await bot._start_scenario(interaction, Story("A topic", summary="A topic"), in_channel=False)

        self.assertEqual(llm.calls, [])
        channel.send.assert_not_awaited()
        self.assertIn("Create Public Threads", interaction.edit_original_response.call_args.kwargs["content"])
        await bot.close()

    async def test_deepen_waits_for_humans_then_posts_one_selected_personality(self):
        llm = StubLLM("unused")
        bot = RaniBot(Settings("unused", "unused"), llm)
        bot._connection.user = SimpleNamespace(id=99)
        channel = MagicMock(spec=discord.TextChannel)
        channel.id = 42
        channel.send = AsyncMock()
        interaction = self.make_interaction(channel)
        transcript = json.dumps([
            {"username": "a", "display_name": "A", "text": "Public funding."},
            {"username": "b", "display_name": "B", "text": "Only with audits."},
        ])

        with patch("bot.read_scenario_discussion", AsyncMock(
            return_value=("**Future Scenario: X**", transcript, 2)
        )), patch("bot.deepen_scenario", AsyncMock(
            return_value=SimpleNamespace(agent_name="Hex", text="What evidence would make the audit credible?")
        )):
            await bot.run_scenario_deepen(interaction)

        channel.send.assert_awaited_once()
        self.assertTrue(channel.send.call_args.args[0].startswith("**Hex:**"))
        self.assertEqual(channel.send.call_args.kwargs["allowed_mentions"].to_dict()["parse"], [])
        self.assertFalse(bot.active_channels)
        await bot.close()
