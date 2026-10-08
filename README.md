# Universal AI Smart Bridge

Локальный L7-шлюз `127.0.0.1:10830` на чистой стандартной библиотеке Python 3.11+: разбирает поле `model`
в теле запроса, выбирает egress-пул (`direct` / `http_connect` / `socks5h`), стримит SSE без буферизации и
один раз прозрачно повторяет запрос через `geo_fallback_pool`, если прямой маршрут вернул 403 `RegionError`.

Запуск: `python -m universal_ai_bridge --config ~/.config/universal-ai-bridge/config.json`
(пример — `config.example.json`). Конфиг перечитывается при смене mtime; невалидный файл игнорируется.

Эндпоинты: `/health`, `/metrics`, `POST /cache/flush`. Тесты: `pytest`.

Слои: `config` (5) · `proxy_pool` (3) · `model_router` + `geo_cache` (2) · `relay` + `error_handler` (4) ·
`adapters/` + `bridge_server` (1). Слоты direct/proxy раздельные; пул с полностью оштрафованными прокси
отвечает 502 мгновенно, не занимая слот. Сбои транспорта никогда не превращаются в 403.
