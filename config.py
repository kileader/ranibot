"""Load and validate local configuration without exposing credentials."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    discord_token: str = field(repr=False)
    llm_api_key: str = field(repr=False)
    llm_provider: str = "openai"
    llm_model: str = "gpt-4.1-mini"
    guild_id: int | None = None
    database_url: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(Path(__file__).with_name(".env"))

        def required(name: str) -> str:
            value = os.getenv(name, "").strip()
            if not value or value.startswith("replace_with_"):
                raise ValueError(f"Set {name} in .env before running the bot.")
            return value

        token = required("DISCORD_BOT_TOKEN")
        api_key = required("LLM_API_KEY")
        provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        if provider != "openai":
            raise ValueError("LLM_PROVIDER must be openai; v0 has no other adapter.")
        model = os.getenv("LLM_MODEL", "gpt-4.1-mini").strip()
        if not model:
            raise ValueError("LLM_MODEL must not be blank.")
        guild_text = os.getenv("DISCORD_GUILD_ID", "").strip()
        guild_id = None
        if guild_text:
            try:
                guild_id = int(guild_text)
            except ValueError:
                raise ValueError("DISCORD_GUILD_ID must be a positive server ID or blank.") from None
            if not 0 < guild_id < 2**64:
                raise ValueError("DISCORD_GUILD_ID must be a positive server ID or blank.")
        database_url = os.getenv("DATABASE_URL", "").strip() or None
        return cls(token, api_key, provider, model, guild_id, database_url)
