"""Configuration for the YouTube Learning Companion.

All configuration is loaded via environment variables or a local .env file.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CompanionSettings(BaseSettings):
    """Runtime configuration for YouTube Learning Companion."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────
    app_name: str = Field(
        default="YouTube Learning Companion", description="Application name"
    )
    version: str = Field(default="0.1.0", description="Application version")

    # ── YouTube Data API v3 ───────────────────────────────────
    youtube_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("youtube_api_key", "videx_youtube_api_key"),
        description="Google/YouTube Data API v3 key (optional; public web mode used when empty)",
    )
    youtube_api_base_url: str = Field(
        default="https://www.googleapis.com/youtube/v3",
        description="YouTube Data API v3 base endpoint URL",
    )
    youtube_request_timeout_seconds: float = Field(
        default=15.0,
        ge=1.0,
        description="Timeout in seconds for YouTube HTTP requests",
    )
    youtube_max_results_per_page: int = Field(
        default=50,
        ge=1,
        le=50,
        description="Maximum items per page for YouTube playlist discovery",
    )
    notes_export_dir: str = Field(
        default="data/notes",
        description="Default directory for exported study notes and manifests",
    )

    # ── LLM Synthesis Configuration ───────────────────────────
    llm_provider: str = Field(
        default="openai",
        validation_alias=AliasChoices(
            "llm_provider", "vlm_provider", "videx_llm_provider", "VLM_PROVIDER"
        ),
        description="LLM provider for companion note synthesis: 'openai', 'ollama', 'mock'",
    )
    llm_model: str = Field(
        default="gpt-4o",
        validation_alias=AliasChoices("llm_model", "vlm_model", "videx_llm_model", "VLM_MODEL"),
        description="Model checkpoint or deployment identifier for study note synthesis",
    )
    llm_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "llm_api_key", "openai_api_key", "videx_llm_api_key", "OPENAI_API_KEY"
        ),
        description="API key for LLM provider (never logged)",
    )
    llm_api_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices("llm_api_base_url", "videx_llm_api_base_url"),
        description="Base URL for OpenAI-compatible chat completions endpoint",
    )
    llm_timeout_seconds: float = Field(
        default=45.0,
        ge=5.0,
        description="Timeout in seconds for LLM synthesis calls",
    )
    llm_max_retries: int = Field(
        default=2,
        ge=0,
        description="Maximum retry attempts on transient network or rate-limit errors",
    )


@lru_cache(maxsize=1)
def get_settings() -> CompanionSettings:
    """Return cached CompanionSettings instance."""
    return CompanionSettings()
