"""Слой 2: инспекция тела (поле `model`), правила маршрутизации и классификатор 403 гео-блокировок."""

from __future__ import annotations

import json
import re
import zlib
from collections.abc import Iterable
from dataclasses import dataclass

from .config import BridgeConfig, RuleConfig
from .errors import BadRequestError, BodyTooLargeError
from .geo_cache import GeoCache

# ───────────────────────────── BodyInspector ─────────────────────────────

_WS = frozenset(b" \t\r\n")
_QUOTE, _BACKSLASH = 0x22, 0x5C
_STRING_SPECIAL = re.compile(rb'["\\]')
_CONTAINER_SPECIAL = re.compile(rb'["{}\[\]]')
_SCALAR_END = re.compile(rb"[,}\]\s]")
_START, _KEY_OR_END, _KEY, _COLON, _VALUE, _SKIP, _AFTER_VALUE, _DONE = range(8)
_MODEL_PATH_RE = re.compile(r"/models/([^/:?]+)")


def model_from_path(path: str) -> str | None:
    """Модель из пути вида `/v1beta/models/<model>:generateContent` (Google-совместимые API)."""
    match = _MODEL_PATH_RE.search(path)
    return match.group(1) if match else None


def _decode_json_string(raw: bytes) -> str:
    if b"\\" not in raw:
        return raw.decode("utf-8", "replace")
    return json.loads(b'"' + raw + b'"')


class BodyInspector:
    """Буферизует тело (до `max_bytes`) и инкрементально извлекает `model` из JSON-объекта верхнего уровня.

    Сканер оперирует токенами: вложенные объекты/массивы/строки пропускаются без разбора,
    поэтому `model`, вложенная глубже уровня 1, игнорируется; дубли ключа допустимы только с одинаковым
    строковым значением, любые иные (другая строка, null, число, массив, объект) — BadRequestError (иначе прокси
    и upstream могли бы выбрать разные модели). Незавершённый токен дочитывается при следующем `feed` (позиция в строке запоминается — без квадратичной деградации).
    """

    def __init__(self, max_bytes: int):
        self._max = max_bytes
        self._buf = bytearray()
        self._pos = 0
        self._state = _START
        self._key: str | None = None
        self._depth = 0
        self._model: str | None = None
        self._model_seen = False
        self._model_first: str | None = None
        self._str_quote = -1
        self._str_scan = -1

    @property
    def model(self) -> str | None:
        return self._model

    @property
    def size(self) -> int:
        return len(self._buf)

    @property
    def scan_finished(self) -> bool:
        return self._state == _DONE

    def view(self) -> memoryview:
        """Буферизованное тело без копирования (после этого `feed` вызывать нельзя)."""
        return memoryview(self._buf)

    @property
    def body(self) -> bytes:
        return bytes(self._buf)

    def feed(self, data: bytes) -> None:
        if len(self._buf) + len(data) > self._max:
            raise BodyTooLargeError(self._max)
        self._buf += data
        if self._state != _DONE:
            self._scan()

    def _note_model(self, value: str | None) -> None:
        """Учесть вхождение ключа `model` (`None` — значение не строка); дубли допустимы только как равные строки."""
        if self._model_seen and (value is None or self._model_first is None or value != self._model_first):
            raise BadRequestError("Conflicting duplicate model keys in request body")
        if not self._model_seen:
            self._model_seen, self._model_first = True, value
        if value:
            self._model = value

    def _string_end(self, quote_pos: int) -> int:
        """Индекс после закрывающей кавычки или -1, если строка ещё не завершена."""
        buf = self._buf
        pos = self._str_scan if self._str_quote == quote_pos else quote_pos + 1
        while True:
            match = _STRING_SPECIAL.search(buf, pos)
            if match is None:
                self._str_quote, self._str_scan = quote_pos, len(buf)
                return -1
            i = match.start()
            if buf[i] == _QUOTE:
                self._str_quote = -1
                return i + 1
            if i + 1 >= len(buf):
                self._str_quote, self._str_scan = quote_pos, i
                return -1
            pos = i + 2

    def _scan(self) -> None:
        buf = self._buf
        while True:
            state, pos, size = self._state, self._pos, len(buf)
            ch = 0
            if state in (_START, _KEY_OR_END, _KEY, _COLON, _VALUE, _AFTER_VALUE):
                while pos < size and buf[pos] in _WS:
                    pos += 1
                self._pos = pos
                if pos >= size:
                    return
                ch = buf[pos]
            if state == _START:
                if ch != 0x7B:
                    self._state = _DONE
                    return
                self._pos, self._state = pos + 1, _KEY_OR_END
            elif state in (_KEY_OR_END, _KEY):
                if ch != _QUOTE:  # `}` сразу после `{` или любой мусор — разбор окончен
                    self._state = _DONE
                    return
                end = self._string_end(pos)
                if end < 0:
                    return
                try:
                    self._key = _decode_json_string(bytes(buf[pos + 1 : end - 1]))
                except ValueError:
                    self._state = _DONE
                    return
                self._pos, self._state = end, _COLON
            elif state == _COLON:
                if ch != 0x3A:
                    self._state = _DONE
                    return
                self._pos, self._state = pos + 1, _VALUE
            elif state == _VALUE:
                if ch == _QUOTE:
                    end = self._string_end(pos)
                    if end < 0:
                        return
                    if self._key == "model":
                        try:
                            value = _decode_json_string(bytes(buf[pos + 1 : end - 1]))
                        except ValueError:
                            value = None
                        self._note_model(value)
                    self._pos, self._state = end, _AFTER_VALUE
                elif ch in (0x7B, 0x5B):
                    if self._key == "model":
                        self._note_model(None)
                    self._depth = 1
                    self._pos, self._state = pos + 1, _SKIP
                else:
                    match = _SCALAR_END.search(buf, pos)
                    if match is None:
                        return
                    if self._key == "model":
                        self._note_model(None)
                    self._pos, self._state = match.start(), _AFTER_VALUE
            elif state == _SKIP:
                match = _CONTAINER_SPECIAL.search(buf, pos)
                if match is None:
                    self._pos = size
                    return
                i = match.start()
                found = buf[i]
                if found == _QUOTE:
                    end = self._string_end(i)
                    if end < 0:
                        self._pos = i
                        return
                    self._pos = end
                elif found in (0x7B, 0x5B):
                    self._depth += 1
                    self._pos = i + 1
                else:
                    self._depth -= 1
                    self._pos = i + 1
                    if self._depth == 0:
                        self._state = _AFTER_VALUE
            else:  # _AFTER_VALUE
                if ch == 0x2C:
                    self._pos, self._state = pos + 1, _KEY
                else:
                    self._state = _DONE
                    return


# ───────────────────────────── ModelRouter ─────────────────────────────


@dataclass(frozen=True)
class RouteDecision:
    pool: str
    source: str  # rule | default | geo_cache
    rule_index: int | None = None


class ModelRouter:
    """Сопоставление имени модели с пулом: правила по порядку (префикс ИЛИ regex), затем `default_pool`."""

    def __init__(self, config: BridgeConfig):
        self._rules: list[tuple[RuleConfig, re.Pattern[str] | None]] = [
            (rule, re.compile(rule.match_regex) if rule.match_regex else None) for rule in config.rules
        ]
        self._default = config.default_pool
        self._fallback = config.geo_fallback_pool
        self._pools = config.pools

    @classmethod
    def from_config(cls, config: BridgeConfig) -> ModelRouter:
        return cls(config)

    def match(self, model: str | None) -> RouteDecision:
        if model:
            for index, (rule, pattern) in enumerate(self._rules):
                if any(model.startswith(prefix) for prefix in rule.match_prefix) or (
                    pattern is not None and pattern.search(model)
                ):
                    return RouteDecision(rule.pool, "rule", index)
        return RouteDecision(self._default, "default")

    def decide(self, model: str | None, upstream: str, geo_cache: GeoCache | None = None) -> RouteDecision:
        """Правила + Geo-Cache: закэшированная гео-блокировка прямого маршрута уводит в `geo_fallback_pool`."""
        decision = self.match(model)
        if (
            geo_cache is not None
            and self._fallback
            and self._fallback != decision.pool
            and self._pools[decision.pool].type == "direct"
            and geo_cache.is_blocked(upstream, model)
        ):
            return RouteDecision(self._fallback, "geo_cache")
        return decision


# ───────────────────────────── RegionErrorClassifier ─────────────────────────────


@dataclass(frozen=True)
class Classification:
    is_region_error: bool
    reason: str


class RegionErrorClassifier:
    """Отличает гео-блокировку 403 от auth/quota/WAF по сигнатурам в теле ответа."""

    def __init__(self, signatures: Iterable[str], max_decoded_bytes: int = 65536):
        self._signatures = tuple(s.lower() for s in signatures if s)
        self._max_decoded = max_decoded_bytes

    def classify(self, status: int, body: bytes, content_encoding: str | None = None) -> Classification:
        if status != 403:
            return Classification(False, "not_403")
        if not body:
            return Classification(False, "empty_body")
        encoding = (content_encoding or "identity").strip().lower()
        if encoding in ("gzip", "x-gzip"):
            try:
                body = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(body, self._max_decoded)
            except zlib.error:
                return Classification(False, "bad_encoding")
        elif encoding not in ("", "identity"):
            return Classification(False, "unsupported_encoding")
        text = body.decode("utf-8", "replace").lower()
        for signature in self._signatures:
            if signature in text:
                return Classification(True, f"region_signature:{signature}")
        return Classification(False, "no_signature")
