from __future__ import annotations

import importlib

import pytest


class TestVertexAIMode:
    def test_default_is_real(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VERTEX_AI_MODE", raising=False)
        from src.core import config as config_module

        importlib.reload(config_module)
        assert config_module.GlobalConfig.VERTEX_AI_MODE == "real"

    def test_stub_mode_set_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VERTEX_AI_MODE", "stub")
        from src.core import config as config_module

        importlib.reload(config_module)
        assert config_module.GlobalConfig.VERTEX_AI_MODE == "stub"

    def test_invalid_value_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VERTEX_AI_MODE", "bogus")
        from src.core import config as config_module

        with pytest.raises(ValueError, match="VERTEX_AI_MODE"):
            importlib.reload(config_module)
