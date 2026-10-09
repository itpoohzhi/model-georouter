"""OpenCode-адаптер: `/v1/*`, `/go/v1/*` → inference-пути opencode.ai."""

from __future__ import annotations

from .base import ClientAdapter, clean_path

EXACT_PATHS = {
    "/v1/responses": "/inference/openai/v1/responses",
    "/v1/chat/completions": "/inference/openai/v1/chat/completions",
    "/v1/messages": "/inference/anthropic/v1/messages",
    "/v1/models": "/inference/openai/v1/models",
    "/go/v1/responses": "/inference/go/openai/v1/responses",
    "/go/v1/chat/completions": "/inference/go/openai/v1/chat/completions",
    "/go/v1/messages": "/inference/go/anthropic/v1/messages",
    "/go/v1/models": "/inference/go/openai/v1/models",
}
PREFIX_PATHS = (
    ("/go/v1/v1beta/models/", "/inference/go/google/v1beta/models/"),
    ("/go/v1beta/models/", "/inference/go/google/v1beta/models/"),
    ("/v1/v1beta/models/", "/inference/google/v1beta/models/"),
    ("/v1beta/models/", "/inference/google/v1beta/models/"),
)


class OpenCodeAdapter(ClientAdapter):
    name = "opencode"

    def normalize(self, path: str) -> str:
        path = clean_path(path)
        if path in EXACT_PATHS:
            return EXACT_PATHS[path]
        for prefix, target in PREFIX_PATHS:
            if path.startswith(prefix):
                return target + path[len(prefix) :]
        return path
