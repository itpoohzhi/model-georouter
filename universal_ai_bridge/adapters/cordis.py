"""Cordis-адаптер (DSH Cordis / OpenCode Zen): `/zen/v1/*`, `/zen/go/v1/*` пробрасываются без переписывания."""

from __future__ import annotations

from .base import ClientAdapter


class CordisAdapter(ClientAdapter):
    name = "cordis"
