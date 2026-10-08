"""CLI: `python -m universal_ai_bridge --config ~/.config/model-georouter/config.json`."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path

from .bridge_server import BridgeServer
from .config import ConfigManager, default_config_path
from .errors import ConfigError
from .logging_utils import add_file_handler, get_logger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Universal AI Smart Bridge")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="путь к config.json (по умолчанию ~/.config/model-georouter/config.json, иначе legacy universal-ai-bridge)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logger = get_logger("universal_ai_bridge")
    try:
        manager = ConfigManager(args.config)
    except (ConfigError, OSError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    add_file_handler(logger, manager.get().server.log_dir)
    server = BridgeServer(manager)

    def stop(signum, frame):  # noqa: ARG001
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    logger.info("listening on %s:%d", *server.server_address[:2])
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
