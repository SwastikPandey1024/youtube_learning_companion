"""Shared test fixtures and configuration for YouTube Learning Companion."""

from __future__ import annotations

import pytest

from youtube_learning_companion.config import CompanionSettings, get_settings


@pytest.fixture()
def settings() -> CompanionSettings:
    """Return a CompanionSettings instance configured for tests."""
    get_settings.cache_clear()
    cfg = CompanionSettings(
        app_name="YouTube-Learning-Companion-Test",
        version="0.1.0",
        youtube_api_key="",
        llm_provider="mock",
    )
    yield cfg
    get_settings.cache_clear()
