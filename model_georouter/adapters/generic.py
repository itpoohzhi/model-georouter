"""Generic-адаптер: `/messages`, `/chat/completions` → стандартные `/v1/...` пути OpenAI/Anthropic-совместимых API."""

from __future__ import annotations

from .base import ClientAdapter, clean_path


class GenericAdapter(ClientAdapter):
    name = "generic"

    def normalize(self, path: str) -> str:
        return "/v1" + clean_path(path)
