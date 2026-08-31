# Ranibot

A small Discord bot, starting with **Agent Terrarium**: a manually triggered
experiment where one bot account hosts three AI personalities:

- **Mira**: curious and imaginative; explores possibilities without automatic agreement.
- **Hex**: skeptical and analytical; asks what the evidence supports.
- **Moss**: playful and slightly weird; makes unexpected but relevant connections.

Normal conversation → `/agents` → three independent decisions → zero to three labeled replies.

The bot fetches the latest **30 channel messages before the command was invoked**, excludes bots, webhooks, system messages, and empty text, then gives each agent the same chronological human transcript. This can mean fewer than 30 human messages. It includes usernames, display names, and text, with each message limited to 1,500 characters. It doesn't download attachments or open links.

Each agent makes one LLM call that returns either a short contribution or `SILENT`. All three calls run concurrently; contributions are posted in Mira/Hex/Moss order. Silence produces no public message. Status and errors are private to whoever ran the command. The command permits only one active run per channel, within this bot process.

There is no database, persistent memory, background participation, tool execution, browsing, Moltbook integration, webhook identity, or multi-bot setup. No message events are subscribed to and no Discord message cache is kept. History is fetched only inside `/agents`. Earlier bot replies are excluded too: v0 does not conduct agent-to-agent debates or remember its previous contributions. Repeated commands on unchanged chat may produce similar replies.

## Windows PowerShell setup

Use Python **3.11 or newer**, a Discord test server you can manage, and an OpenAI API key with access to the configured model. The default is `gpt-4.1-mini`; no paid API calls are needed for the offline tests.

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
LLM_MODEL=gpt-4.1-mini
DISCORD_GUILD_ID=your_numeric_test_server_id
```

Create an API key in the [OpenAI API dashboard](https://platform.openai.com/api-keys). Configure API billing/usage limits as appropriate. Each nonempty invocation sends three requests, even if every agent chooses silence. No automatic retries are configured.

`DISCORD_GUILD_ID` is optional but recommended: when set, `/agents` is synced only to that server. Blank means global registration, which may take longer to appear. Use one registration mode consistently while testing; switching modes does not delete commands previously registered in the other scope. Existing environment variables take precedence over `.env`. Restart the bot after configuration changes.

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

Also test a fresh channel containing only a routine acknowledgment, and an empty channel. The first should encourage silence (not guarantee it); the second should make no LLM calls. Ordinary messages without `/agents` must never trigger a reply. Agent text is capped at 800 characters; Discord mentions are disabled and link previews suppressed.

## Railway hosting

The root `Dockerfile` installs Python 3.12 and starts `python bot.py` as a non-root
user. Railway [automatically detects the Dockerfile](https://docs.railway.com/builds/dockerfiles).
This is a continuously running Discord bot, not a website: it does not need a
domain, public port, HTTP healthcheck, database, or volume. Keep **Serverless / App
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
   | `LLM_MODEL` | `gpt-4.1-mini` (or your current model) |
   | `DISCORD_GUILD_ID` | Your test server's numeric ID |

   Set these as runtime service variables, not Docker build arguments. Do not
   upload `.env` to the repository or add credentials to the Dockerfile.
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

Tests use fake Discord messages/interactions and a fake LLM, plus a mocked OpenAI SDK call. They check context boundaries and ordering, silent decisions, labeled posting, mention suppression, empty channels, overlapping invocations, permission failures, partial API failure, timeouts, and the provider request contract. They do not contact Discord/OpenAI or require `.env`. Real credentials and a test server are still needed to verify live permissions, command visibility, model access, and response quality.

## Small code map

| File | Responsibility |
| --- | --- |
| `bot.py` | Discord connection, `/agents`, recent human context, labeled posting |
| `agents.py` | Agent dataclass, personalities, silence parsing, independent calls |
| `llm.py` | Async provider interface and OpenAI adapter |
| `config.py` | Validated `.env` configuration |
| `tests/test_ranibot.py` | Offline workflow checks |

To change personalities, edit `AGENTS` in `agents.py`. To add another provider later, implement `generate(system_prompt, context)` and `close()` in `llm.py`, then update its factory and configuration validation. Only `openai` works today; setting a different provider name does not magically add compatibility. The adapter uses the [OpenAI Responses API](https://developers.openai.com/api/docs/guides/text?api-mode=responses). The [default model's documentation](https://developers.openai.com/api/docs/models/gpt-4.1-mini) lists supported endpoints; choose a model available to your API project. Reasoning-heavy models may need a larger output budget than this toy's 400-token cap.

## Privacy and limitations

Use an opt-in test channel: tell participants that invoking `/agents` sends recent usernames and message text to OpenAI in three separate requests. The bot keeps no transcripts on disk and logs decisions/error types rather than message bodies or API keys. `store=False` disables Responses application-state storage, but does **not** guarantee zero provider retention; provider abuse-monitoring policies still apply. See [OpenAI data controls](https://platform.openai.com/docs/guides/your-data).

This is a toy, not a moderation system or a secure prompt-injection defense. Conversation content is separated from system instructions, and agents have no tools or secrets in their prompts, but model output can still be mistaken or manipulated. Anyone who can use the command can incur API costs. Restrict installation and command access to trusted testers; there is no persistent rate limiter. This v0 intentionally stops at the requested workflow.

## Troubleshooting

- **Bot won't log in:** check the bot token (not client secret/public key), and whether it was reset since configuring `.env`.
- **Privileged intent error or missing human text:** enable Message Content Intent on the same application, save, and restart. Media-only messages have no usable text in v0.
- **`/agents` doesn't appear:** check console sync output, server ID, installation scopes, and Use Application Commands permission. Reopen Discord; use a test guild ID to avoid waiting on global registration.
- **Forbidden / missing access:** check both server roles and channel overrides. Ensure the bot is actually installed in the configured server and has read/send permissions in that channel.
- **One or all agents failed:** console logs distinguish timeout, authentication, rate-limit, and other exception types without dumping sensitive API error bodies. Check API key, model access, billing, quota, and connectivity. Incomplete/empty provider output is reported as failure rather than posted as a partial answer.
- **All agents chose silence:** try a substantive open question in the human discussion, or adjust the personality prompts after observing several trials. Don't add a forced fallback reply; silence is part of the experiment.
