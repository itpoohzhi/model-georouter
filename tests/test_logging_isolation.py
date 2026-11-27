"""Unit tests for model_georouter stderr logging isolation (Wave 5)."""

from __future__ import annotations

import io
import logging
from pathlib import Path

from model_georouter import __main__ as main_mod
from model_georouter.logging_utils import get_logger


def _cleanup_logger(logger: logging.Logger) -> None:
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()


def test_info_not_propagated_to_stderr_by_default(monkeypatch, tmp_path: Path) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text('{"routes": [], "proxies": {}, "server": {"port": 0}}', encoding="utf-8")
    monkeypatch.delenv("GEOROUTER_STDERR_PROPAGATE", raising=False)
    monkeypatch.setattr(
        main_mod,
        "ConfigManager",
        lambda _p: type(
            "M",
            (),
            {"get": staticmethod(lambda: type("C", (), {"server": type("S", (), {"log_dir": str(tmp_path)})()})())},
        )(),
    )

    class _FakeServer:
        server_address = ("127.0.0.1", 19000)

        def serve_forever(self, **_kw):
            logging.getLogger("model_georouter").info("INFO_SHOULD_NOT_BE_ON_STDERR")
            logging.getLogger("external_lib").warning("WARN_ON_ROOT_STDERR")

        def server_close(self):
            pass

        def shutdown(self):
            pass

    monkeypatch.setattr(main_mod, "BridgeServer", lambda _m: _FakeServer())
    root = logging.getLogger()
    buf = io.StringIO()
    stderr_probe = logging.StreamHandler(buf)
    stderr_probe.setLevel(logging.INFO)
    root.addHandler(stderr_probe)
    try:
        rc = main_mod.main(["--config", str(cfg)])
        assert rc == 0
        assert get_logger("model_georouter").propagate is False
        err_text = buf.getvalue()
        assert "INFO_SHOULD_NOT_BE_ON_STDERR" not in err_text
        assert "WARN_ON_ROOT_STDERR" in err_text
        log_files = list(tmp_path.glob("*.log"))
        assert log_files and "INFO_SHOULD_NOT_BE_ON_STDERR" in log_files[0].read_text(encoding="utf-8")
    finally:
        root.removeHandler(stderr_probe)
        _cleanup_logger(get_logger("model_georouter"))


def test_env_rollback_restores_stderr_propagation(monkeypatch, tmp_path: Path) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text('{"routes": []}', encoding="utf-8")
    monkeypatch.setenv("GEOROUTER_STDERR_PROPAGATE", "1")
    monkeypatch.setattr(
        main_mod,
        "ConfigManager",
        lambda _p: type(
            "M",
            (),
            {"get": staticmethod(lambda: type("C", (), {"server": type("S", (), {"log_dir": str(tmp_path)})()})())},
        )(),
    )

    class _FakeServer:
        server_address = ("127.0.0.1", 19001)

        def serve_forever(self, **_kw):
            pass

        def server_close(self):
            pass

        def shutdown(self):
            pass

    monkeypatch.setattr(main_mod, "BridgeServer", lambda _m: _FakeServer())
    root = logging.getLogger()
    buf = io.StringIO()
    stderr_probe = logging.StreamHandler(buf)
    stderr_probe.setLevel(logging.INFO)
    root.addHandler(stderr_probe)
    try:
        rc = main_mod.main(["--config", str(cfg)])
        assert rc == 0
        assert get_logger("model_georouter").propagate is True
        assert "listening on 127.0.0.1:19001" in buf.getvalue()
    finally:
        root.removeHandler(stderr_probe)
        _cleanup_logger(get_logger("model_georouter"))
