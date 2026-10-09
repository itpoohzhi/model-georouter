"""Иерархия исключений моста; маппинг в HTTP-ответы — в `error_handler.py`."""

from __future__ import annotations


class BridgeError(Exception):
    """Базовая ошибка моста."""


class ConfigError(BridgeError, ValueError):
    """Конфигурация не прошла валидацию."""


class BadRequestError(BridgeError):
    """Клиентский запрос некорректен (400)."""


class NotFoundError(BridgeError):
    """Путь не обслуживается ни одним адаптером (404)."""


class ClientTimeoutError(BridgeError):
    """Клиент не прислал голову/тело запроса вовремя (408)."""


class BodyTooLargeError(BridgeError):
    """Тело запроса превышает `body_buffer_max_bytes` (413)."""

    def __init__(self, limit: int):
        super().__init__(f"request body exceeds {limit} bytes")
        self.limit = limit


class FramingError(BridgeError):
    """Нарушено HTTP-обрамление тела (chunked / Content-Length)."""


class SlotsExhaustedError(BridgeError):
    """Все слоты соответствующего пула заняты (503)."""

    def __init__(self, kind: str):
        super().__init__(f"all {kind} slots are busy")
        self.kind = kind


class AllProxiesPenalizedError(BridgeError):
    """Все прокси пула оштрафованы: circuit breaker размыкается мгновенно."""

    def __init__(self, pool: str):
        super().__init__(f"all proxies of pool {pool!r} are temporarily penalized")
        self.pool = pool


class EgressConnectError(BridgeError):
    """Не удалось установить соединение с upstream (напрямую или через прокси)."""


class EgressTimeoutError(EgressConnectError):
    """Таймаут установления соединения (connect или общий дедлайн egress)."""


class UpstreamTimeoutError(BridgeError):
    """Upstream не прислал заголовки ответа за `headers_timeout`."""


class UpstreamProtocolError(BridgeError):
    """Upstream оборвал соединение или прислал некорректный ответ до начала ответа клиенту."""
