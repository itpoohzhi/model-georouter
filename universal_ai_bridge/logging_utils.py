"""Логгер с маскированием секретов (Bearer-токены, Basic-пароли, userinfo прокси)."""

from __future__ import annotations

import logging
import logging.handlers
import os
import re
from pathlib import Path

MASK = "***"
_TOKEN_PREFIXES = ("sk-", "sk_", "st-", "st_")

_USERINFO_RE = re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://[^/\s:@]+:)[^/\s@]+@")
_BEARER_RE = re.compile(r"(?i)\b(bearer)(\s+)([^\s\"',;]+)")
_BASIC_RE = re.compile(r"(?i)\b(basic)(\s+)([A-Za-z0-9+/=._~-]{4,})")
_STANDALONE_RE = re.compile(r"(?i)\b(sk|st)([-_])[A-Za-z0-9][A-Za-z0-9_\-]{7,}")
_KEYVALUE_RE = re.compile(
    r"(?i)\b(password|passwd|pwd|api[_-]?key|x-api-key|secret|token)([\"']?\s*[:=]\s*[\"']?)(?!\*\*\*)[^\s\"',;&]+"
)
_QUERY_RE = re.compile(r"(?i)([?&](?:key|api_key|token|access_token)=)[^&#\s]*")


def _mask_bearer(match: re.Match[str]) -> str:
    token = match.group(3)
    lowered = token.lower()
    for prefix in _TOKEN_PREFIXES:
        if lowered.startswith(prefix):
            return f"{match.group(1)}{match.group(2)}{token[: len(prefix)]}{MASK}"
    return f"{match.group(1)}{match.group(2)}{MASK}"


def _mask_basic(match: re.Match[str]) -> str:
    token = match.group(3)
    looks_encoded = bool(re.search(r"[0-9+/=]", token)) or (
        any(c.isupper() for c in token) and any(c.islower() for c in token) and not token.istitle()
    )
    if not looks_encoded:
        return match.group(0)
    return f"{match.group(1)}{match.group(2)}{MASK}"


def mask_secrets(text: str) -> str:
    """Скрыть секреты в произвольной строке; операция идемпотентна."""
    text = _USERINFO_RE.sub(lambda m: f"{m.group(1)}{MASK}@", text)
    text = _BEARER_RE.sub(_mask_bearer, text)
    text = _BASIC_RE.sub(_mask_basic, text)
    text = _STANDALONE_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{MASK}", text)
    text = _KEYVALUE_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{MASK}", text)
    return _QUERY_RE.sub(lambda m: f"{m.group(1)}{MASK}", text)


class SecretMaskingFilter(logging.Filter):
    """Фильтр логгера: подставляет уже замаскированное сообщение."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = mask_secrets(record.getMessage())
        record.args = ()
        return True


def get_logger(name: str) -> logging.Logger:
    """Вернуть логгер с установленным фильтром маскирования (ровно один раз)."""
    logger = logging.getLogger(name)
    if not any(isinstance(f, SecretMaskingFilter) for f in logger.filters):
        logger.addFilter(SecretMaskingFilter())
    return logger


def add_file_handler(logger: logging.Logger, log_dir: str | Path) -> Path:
    """Подключить ротируемый файловый лог с правами 0600 в каталоге `log_dir`."""
    directory = Path(log_dir).expanduser()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / "bridge.log"
    handler = logging.handlers.RotatingFileHandler(path, maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path
