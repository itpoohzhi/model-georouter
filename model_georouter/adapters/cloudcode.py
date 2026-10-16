"""CloudCode-адаптер: проксирование к daily-cloudcode-pa.googleapis.com с санитизацией loadCodeAssist."""

from __future__ import annotations

import gzip
import json
import zlib

from ..http_wire import ResponseHead
from .base import AdapterRoute, ClientAdapter, clean_path


class CloudCodeAdapter(ClientAdapter):
    """Адаптер для Google Cloud Code (Antigravity CLI / agy).

    Нормализует префикс /cloudcode/* -> /*, перехватывает ответ v1internal:loadCodeAssist
    и удаляет элементы ineligibleTiers с причиной UNSUPPORTED_LOCATION при наличии paidTier/currentTier,
    чтобы agy не блокировал запуск из-за неверной проверки региона.
    """

    name = "cloudcode"

    def normalize(self, path: str) -> str:
        cleaned = clean_path(path)
        for prefix in self.prefixes:
            if cleaned == prefix:
                return "/"
            if cleaned.startswith(prefix + "/"):
                return clean_path(cleaned[len(prefix) :])
        return cleaned

    def wants_transform(self, route: AdapterRoute, head: ResponseHead) -> bool:
        """Перехватываем только успешные ответы (200) для эндпоинта loadCodeAssist."""
        if head.status != 200:
            return False
        return "loadCodeAssist" in route.path

    def transform_response(self, route: AdapterRoute, head: ResponseHead, body: bytes) -> tuple[ResponseHead, bytes]:
        """Удаляет ineligibleTiers с причиной UNSUPPORTED_LOCATION из ответа loadCodeAssist при наличии tier."""
        if not body:
            return head, body

        # Декомпрессия gzip при необходимости
        encoding = (head.get("content-encoding") or "").lower()
        raw_data = body
        was_compressed = False
        if encoding == "gzip":
            try:
                raw_data = gzip.decompress(body)
                was_compressed = True
            except (gzip.BadGzipFile, zlib.error, OSError, EOFError, ValueError):
                return head, body
        elif encoding and encoding != "identity":
            return head, body

        try:
            payload = json.loads(raw_data)
        except (ValueError, RecursionError):
            return head, body

        if not isinstance(payload, dict):
            return head, body

        # Очищаем ineligibleTiers только при наличии активного или оплаченного тарифа (paidTier / currentTier)
        has_valid_tier = bool(
            payload.get("currentTier")
            or payload.get("current_tier")
            or payload.get("paidTier")
            or payload.get("paid_tier")
        )
        if not has_valid_tier:
            return head, body

        mutated = False

        # Очистка ineligibleTiers в camelCase (Google API) и snake_case (на случай вариаций)
        for key in ("ineligibleTiers", "ineligible_tiers"):
            tiers = payload.get(key)
            if isinstance(tiers, list) and tiers:
                filtered = [
                    tier
                    for tier in tiers
                    if not (
                        isinstance(tier, dict)
                        and (
                            tier.get("reasonCode") == "UNSUPPORTED_LOCATION"
                            or tier.get("reason_code") == "UNSUPPORTED_LOCATION"
                        )
                    )
                ]
                if len(filtered) != len(tiers):
                    payload[key] = filtered
                    mutated = True

        if not mutated:
            return head, body

        try:
            new_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        except (ValueError, TypeError, UnicodeEncodeError):
            return head, body

        new_headers = [
            (k, v)
            for k, v in head.headers
            if k.lower() not in {"content-length", "transfer-encoding"}
            and not (was_compressed and k.lower() == "content-encoding")
        ]
        new_headers.append(("Content-Length", str(len(new_body))))

        new_head = ResponseHead(
            version=head.version,
            status=head.status,
            reason=head.reason,
            headers=tuple(new_headers),
        )
        return new_head, new_body
