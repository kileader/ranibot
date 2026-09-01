# Ranibot

A small Discord bot, starting with **Agent Terrarium**: a manually triggered
experiment where one bot account hosts three AI personalities:

- **Mira**: curious and imaginative; explores possibilities without automatic agreement.
- **Hex**: skeptical and analytical; asks what the evidence supports.
- **Moss**: playful and slightly weird; makes unexpected but relevant connections.

Normal conversation → `/agents` → three independent decisions → zero to three labeled replies.

The bot fetches the latest **30 channel messages before the command was invoked**, excludes bots, webhooks, system messages, and empty text, then gives each agent the same chronological human transcript. This can mean fewer than 30 human messages. It includes usernames, display names, and text, with each message limited to 1,500 characters. It doesn't download attachments or open links.

Each agent makes one LLM call that returns either a short contribution or `SILENT`. All three calls run concurrently; contributions are posted in Mira/Hex/Moss order. Silence produces no public message. Status and errors are private to whoever ran the command. AI slash commands and message actions permit only one active run per channel, within this bot process.

Ranibot also has optional per-server memory. A member with **Manage Server** must
explicitly enable it. Once enabled, Ranibot observes future human text in server
channels it can access, buffers 20 messages, and makes one LLM request to extract up
to three durable server-level memories. It then deletes that processed message
batch. Actual replies from Mira, Hex, and Moss are retained in separate agent
journals. Shared memories and the selected personality's journal are supplied to
future agent requests in that server. Unprocessed buffered messages expire after
seven days, and memory never crosses server boundaries.

Memory is deliberately limited: it does not build personal profiles, download
attachments, open links, execute tools, browse, or respond autonomously. The bot
keeps no Discord message cache. When memory is paused or unavailable, channel
history is fetched only inside `/agents` or `/synthesize` and the older memory-free
behavior remains intact.

## Commands

| Command | What it does | AI requests | Public output |
| --- | --- | --- | --- |
| `/agents` | Reads recent human chat; all three agents independently decide whether to contribute | 3 when there is readable human text | Zero to three labeled replies |
| `/ask agent question` | Asks one agent directly using the question plus enabled server/agent memory; it does not read channel history | 1 for a valid, permitted request | One short labeled answer if successful |
| `/status` | Shows process uptime, Discord heartbeat latency, and configured model | 0 | None; private response |
| `/chesslab` | Shares the Chess Lab app link and a short introduction | 0 | One message with the app link |
| `/synthesize` | Maps recent common ground, tensions, open questions, and a possible next step | 1 when there is readable human text | One compact synthesis if the discussion supports it |
| `/consent` | Explains what Ranibot reads, sends, stores, and costs | 0 | None; private response |
| `/help` | Explains these commands and their privacy/cost behavior | 0 | None; private response |
| `/memory ...` | Inspects, enables, pauses, forgets, or clears per-server memory | 0, except background extraction after each 20 buffered messages | None; private response |

### Persistent memory

`/memory` contains six subcommands:

- `/memory status` shows whether storage is configured and memory is enabled.
- `/memory enable` starts observation and memory use; **Manage Server** is required.
- `/memory pause` stops observation and stops applying saved memory without deleting it.
- `/memory list` privately shows the latest 15 shared and agent memories with IDs.
- `/memory forget memory_id` deletes one entry; **Manage Server** is required.
- `/memory clear confirm:True` deletes all stored memories and buffered text for this server; **Manage Server** is required.

Server memory holds up to 40 extracted notes. Each agent journal holds its 20 most
recent posted contributions. Memories are fallible context, not authoritative facts;
current conversation should override old or corrected material. If extraction fails,
the batch remains buffered for a later retry, subject to the seven-day buffer limit.
To bound prompt size and cost, an agent request receives at most the 12 newest server
notes and its 8 newest journal entries. Tell members before enabling memory.

### Message actions

Right-click a text message and choose **Apps** to use one of these actions:

- **Ask Mira about this**
- **Analyze with Hex**
- **Connect with Moss**

Each action sends the selected message text to one personality in one paid AI
request, then replies publicly to that message. It does not send the author name,
surrounding conversation, attachments, embeds, or linked-page contents. If memory is
enabled, saved server context and that personality's journal are included too.
Empty messages are rejected and selected text is capped at 1,500 characters.
Mentions and link previews are suppressed in the generated reply.

For example, use `/ask`, select **Hex**, and enter "How could I measure whether my
study schedule improves retention?" The question is limited to 1,500 characters;
the answer is limited to 800 characters. Unlike `/agents`, direct questions use an
answering prompt rather than asking whether to participate. If the provider fails
or returns silence, you get a private notice instead of a public fallback.

`/ask` never reads channel history or adds the questioner's username to the provider request. Enabled saved memory is included. Its
answer is public and may repeat parts of the question, so do not enter secrets.
`/ask` and `/agents` share a per-channel busy guard to prevent overlapping AI runs.
Both suppress mentions and link previews. No command gives agents tools or access
to your computer. `/status` reports configuration; it does not test OpenAI access,
billing, or available credit. Uptime resets when Railway restarts the process.

`/synthesize` uses the same 30-message human-only snapshot as `/agents`, but makes
one call with a neutral facilitator prompt. It avoids inventing consensus, labels
inference and uncertainty, and can decline when the discussion is too thin. Its
result is public and capped at 1,500 characters. `/consent` is a private, fixed
explanation of data flow and cost; it neither reads history nor calls OpenAI.

`/chesslab` posts a fixed introduction and a link to
[Chess Lab](https://chess-lab-zeta.vercel.app). It does not read channel history,
call AI, fetch the website, link accounts, or access anyone's games. The response
is public, with mentions and link previews suppressed. Chess Lab sign-in happens
on the website, and game libraries remain private. No new configuration or
permissions are required.

All eight command roots follow `DISCORD_GUILD_ID`: a configured test server receives the
commands immediately; leaving it blank registers them globally on startup.

## Windows PowerShell setup

Use Python **3.11 or newer**, a Discord test server you can manage, and an OpenAI API key with access to the configured model. The default is `gpt-5.6-luna` with reasoning disabled; no paid API calls are needed for the offline tests.

### 1. Create the Discord application and bot

1. Open the [Discord Developer Portal](https://discord.com/developers/applications), select **New Application**, and name it **Ranibot**. If you already created Agent Terrarium, keep that application and token; you can change its name without recreating it.
2. Open its **Bot** page. Create the bot if prompted, then use **Reset Token** to generate/copy its token. This is the bot token, not the application's public key or client secret.
3. Keep the token private. Put it only in your local `.env` in step 4. If exposed, reset it immediately.

### 2. Enable the required intent

On the **Bot** page, under **Privileged Gateway Intents**, enable **Message Content Intent** and save. Leave **Server Members Intent** and **Presence Intent** off; this bot doesn't need them. Code enables content access as well. Slash commands alone don't give access to ordinary conversation text. See the [discord.py intents guide](https://discordpy.readthedocs.io/en/stable/intents.html).

### 3. Invite it to a test server

1. On the application's **Installation** page, enable **Guild Install** (server installation); user installation isn't needed.
2. Under **Default Install Settings → Guild Install**, select scopes **`bot`** and **`applications.commands`**.
3. Grant **View Channels**, **Read Message History**, and **Send Messages**. Also grant **Send Messages in Threads** if you want to test in threads. Don't grant Administrator.
4. Open the generated install link and add the app to your test server. If the portal offers the OAuth2 URL Generator instead, select those same scopes and permissions there.
5. Check channel permission overrides: the bot needs access in the actual test channel. Your own account needs **Use Application Commands** and **Read Message History**. Start with a normal text channel; private or locked threads may require additional access.

The official [Discord bot setup guide](https://docs.discord.com/developers/quick-start/getting-started) shows application creation and installation. This Python bot connects through Discord's Gateway, so **leave the Interactions Endpoint URL unset**; you do not need a public web server or tunnel.

For quick test-server command registration, enable **User Settings → Advanced → Developer Mode** in Discord. Right-click your server and **Copy Server ID**.

### 4. Configure `.env`

In PowerShell, from this project folder:

```powershell
Set-Location C:\Users\kevin\projects\ranibot
Copy-Item .env.example .env
notepad .env
```

Only copy the example on first setup; don't overwrite an existing configured `.env`.

Fill in:

```dotenv
DISCORD_BOT_TOKEN=your_real_bot_token
LLM_API_KEY=your_real_openai_api_key
LLM_PROVIDER=openai
LLM_MODEL=gpt-5.6-luna
LLM_REASONING_EFFORT=none
DISCORD_GUILD_ID=your_numeric_test_server_id
DATABASE_URL=your_postgresql_connection_string
```

Create an API key in the [OpenAI API dashboard](https://platform.openai.com/api-keys). Configure API billing/usage limits as appropriate. Each nonempty `/agents` invocation sends three requests, even if every agent chooses silence; `/synthesize` sends one request with the same filtered context, and `/ask` sends one request to its selected agent. No automatic retries are configured.

`DISCORD_GUILD_ID` is optional: when set, all eight command roots are synced only to
that server. Blank means global registration. `DATABASE_URL` is optional for a local
memory-free run; `/memory enable` requires PostgreSQL. Existing environment variables
take precedence over `.env`. Restart the bot after configuration changes.

### 5. Install dependencies

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

These commands use the virtual environment directly, so you don't need to activate it or change PowerShell execution policy. Dependencies are bounded to compatible major versions, not a full lockfile.

### 6. Run

```powershell
.\.venv\Scripts\python.exe bot.py
```

Wait for the console to report command synchronization and the connected bot ID. Keep PowerShell open. Stop with **Ctrl+C**. No shell commands are exposed to agents or Discord users.

### 7. Test `/agents`

Have a short conversation in the test channel, for example:

> Kevin: I'm considering measuring whether changing my study schedule improves retention.
>
> Friend: Compare a week of morning study with a week at night?
>
> Kevin: How could we avoid confusing schedule effects with easier material?

Run `/agents` using Discord's slash-command picker. You should get a private status and potentially messages such as `Mira: ...`, `Hex: ...`, and `Moss: ...` (names are bold). Output is generated, so the content and number of speakers vary. All three may stay silent; that is a valid result, not a command failure.

Also test a fresh channel containing only a routine acknowledgment, and an empty channel. The first should encourage silence (not guarantee it); the second should make no LLM calls. Ordinary messages never trigger a public reply. With memory enabled they can be buffered and eventually included in one extraction request per 20 messages. Agent text is capped at 800 characters; Discord mentions are disabled and link previews suppressed.

## Railway hosting

The root `Dockerfile` installs Python 3.12 and starts `python bot.py` as a non-root
user. Railway [automatically detects the Dockerfile](https://docs.railway.com/builds/dockerfiles).
This is a continuously running Discord bot, not a website: it does not need a
domain, public port, HTTP healthcheck, or volume. Persistent memory uses a linked
Railway PostgreSQL service; a memory-free deployment can omit it. Keep **Serverless / App
Sleeping off** and use **one replica in one region**.

Railway hosting charges are separate from OpenAI API usage. If you already have
Hobby, this service shares your existing included resource allowance with your
other services; check your workspace usage and spending controls.

### Deploy from GitHub

1. Put the project files in a **private GitHub repository**. Never commit `.env`,
   tokens, or `.venv`. Keep `.gitignore` and `.dockerignore` in the repository.
2. In Railway, create a project with a GitHub-repository service, or add that service
   to an existing project. Select this repository. An initial deployment without
   credentials will exit with a configuration error; add variables before retrying.
3. In the service's **Variables** tab, enter the values from your local `.env`:

   | Variable | Value |
   | --- | --- |
   | `DISCORD_BOT_TOKEN` | Your existing Discord bot token |
   | `LLM_API_KEY` | Your funded OpenAI API key |
   | `LLM_PROVIDER` | `openai` |
   | `LLM_MODEL` | `gpt-5.6-luna` |
   | `LLM_REASONING_EFFORT` | `none` |
   | `DISCORD_GUILD_ID` | Your test server's numeric ID |
   | `DATABASE_URL` | A Railway reference such as `${{Postgres.DATABASE_URL}}` |

   Set these as runtime service variables, not Docker build arguments. Do not
   upload `.env` to the repository or add credentials to the Dockerfile.
   To use memory, add a PostgreSQL service to the same Railway project, then add a
   `DATABASE_URL` reference in the Ranibot service pointing to that database's
   `DATABASE_URL`. Do not copy the generated database password into source control.
4. In service settings, leave the start command unset to use the Dockerfile's
   `CMD` (or set it to `python bot.py`). Leave the HTTP healthcheck path unset,
   disable sleeping, and keep one replica. Use an **On Failure** restart policy
   with a bounded retry count, such as 3, if the plan permits it.
5. **Stop the local bot with Ctrl+C before launching the cloud copy.** Two
   processes using the same bot token can compete for slash-command interactions.
6. Deploy the staged changes. Check the runtime logs for command synchronization
   and `Ranibot connected as bot ID ...`, then run `/agents` in Discord.
   A successful image build alone does not verify Discord or OpenAI connectivity.

### Updating the GitHub-connected bot

The private repository is [kileader/ranibot](https://github.com/kileader/ranibot).
The existing Railway service deploys the `main` branch. Its credentials remain in
Railway service variables; never commit `.env` or copy credentials into GitHub.

After editing the local project, run the offline checks:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
git status
git diff
```

Stage only the files you intend to change, inspect `git diff --cached`, then commit
and push to `main`. Each push to that branch triggers Railway to build and deploy
with the existing Dockerfile and service variables. Local edits and commits alone
do not change the hosted bot. Check Railway's deployment status and runtime logs
after a push; do not start another local copy while the cloud bot is running.

The direct CLI upload method below remains an alternative, but use GitHub pushes
for normal updates so the hosted code matches the repository.

### Deploy the local folder without GitHub

With the [Railway CLI](https://docs.railway.com/cli) installed, run from this folder:

```powershell
railway login
railway link
```

Link an existing project, environment, and empty service you created in Railway.
Configure its variables and settings as above, stop the local bot, then run:

```powershell
railway up
```

Railway's uploader [respects `.gitignore` and `.railwayignore`](https://docs.railway.com/cli/up).
Our `.railwayignore` permits only the deployment files; `.env`, the local virtual
environment, and other workspace files are excluded. Do not use `--no-gitignore`.
You do not need Docker Desktop running for Railway to build the uploaded source.

Avoid invoking `/agents` during redeployments, when old and new instances may
briefly overlap. To return to local hosting, stop/remove the Railway deployment
before starting the local process. No re-invite or new Discord account is needed.
A Railway-hosted process cannot directly read your home PC's temperatures or usage.

## Offline verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use fake Discord messages/interactions and a fake LLM, plus a mocked OpenAI SDK call. They check context boundaries and ordering, silent decisions, labeled posting, mention suppression, empty channels, overlapping invocations, permission failures, partial API failure, timeouts, direct-question isolation, synthesis boundaries and failures, selected-message isolation, utility commands without AI calls, and the provider request contract. They do not contact Discord/OpenAI or require `.env`. Real credentials and a test server are still needed to verify live permissions, command visibility, model access, and response quality.

## Small code map

| File | Responsibility |
| --- | --- |
| `bot.py` | Discord connection, eight command roots, message observation, memory controls, and labeled posting |
| `agents.py` | Agent dataclass, personalities, silence parsing, independent calls |
| `llm.py` | Async provider interface and OpenAI adapter |
| `config.py` | Validated `.env` configuration |
| `memory.py` | PostgreSQL schema/store, bounded buffers, and server-memory extraction |
| `tests/test_ranibot.py` | Offline workflow checks |

To change personalities, edit `AGENTS` in `agents.py`. To add another provider later, implement `generate(system_prompt, context)` and `close()` in `llm.py`, then update its factory and configuration validation. Only `openai` works today; setting a different provider name does not magically add compatibility. The adapter uses the [OpenAI Responses API](https://developers.openai.com/api/docs/guides/text?api-mode=responses). The [default model's documentation](https://developers.openai.com/api/docs/models/gpt-5.6-luna) lists supported endpoints. Ranibot explicitly uses `reasoning.effort: none` so its 400-token output cap is reserved for the short visible answer; other models may support different reasoning values, and account availability can vary.

## Privacy and limitations

Tell participants before using conversation commands or enabling memory. `/agents`
sends recent usernames and text to OpenAI in three requests; `/synthesize` sends the
same filtered context in one. `/ask` and message actions make one request and include
enabled saved memory. When ambient memory is enabled, Ranibot buffers human text
from all server channels it can access and sends each 20-message batch to OpenAI in
one extraction request. Processed raw batches are deleted and unprocessed buffered
messages expire after seven days; extracted server notes
and bounded personality journals remain in PostgreSQL until pruned or deleted.
Memory controls themselves make no AI requests. The bot logs counts and error types,
not message bodies, database URLs, or API keys. `store=False` disables Responses
application-state storage, but does **not** guarantee zero provider retention; provider
policies still apply. See [OpenAI data controls](https://platform.openai.com/docs/guides/your-data).

This is a toy, not a moderation system or a secure prompt-injection defense.
Conversation and memory are separated from system instructions, and agents have no
tools or secrets in their prompts, but output and extracted memories can still be
mistaken or manipulated. Anyone who can invoke AI features, and anyone who posts in
an enabled server, can contribute to API usage. Restrict the bot to trusted servers;
there is no persistent rate limiter. The bot does not autonomously post, browse, or
use system tools.

## Troubleshooting

- **Bot won't log in:** check the bot token (not client secret/public key), and whether it was reset since configuring `.env`.
- **Privileged intent error or missing human text:** enable Message Content Intent on the same application, save, and restart. Media-only messages have no usable text in v0.
- **`/agents` doesn't appear:** check console sync output, server ID, installation scopes, and Use Application Commands permission. Reopen Discord; use a test guild ID to avoid waiting on global registration.
- **Forbidden / missing access:** check both server roles and channel overrides. Ensure the bot is actually installed in the configured server and has read/send permissions in that channel.
- **One or all agents failed:** console logs distinguish timeout, authentication, rate-limit, and other exception types without dumping sensitive API error bodies. Check API key, model access, billing, quota, and connectivity. Incomplete/empty provider output is reported as failure rather than posted as a partial answer.
- **All agents chose silence:** try a substantive open question in the human discussion, or adjust the personality prompts after observing several trials. Don't add a forced fallback reply; silence is part of the experiment.
- **Memory says it is unavailable:** attach PostgreSQL and set `DATABASE_URL` on the Ranibot service, then redeploy.
- **Memory is enabled but still empty:** the first extraction happens after 20 qualifying human messages. Check that the bot can view those channels and that Message Content Intent remains enabled.
