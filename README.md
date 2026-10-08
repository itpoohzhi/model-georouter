# model-georouter

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-0%20pip%20runtime-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/tests-306%20passed-success.svg)]()
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)]()

**model-georouter** — это автономный, высокопроизводительный локальный L7-шлюз и умный маршрутизатор сетевых запросов к AI-моделям, написанный на **чистой стандартной библиотеке Python 3.11+** (ноль сторонних runtime-зависимостей).

Проект решает фундаментальную проблему маршрутизации API-трафика современных AI-агентов: он позволяет **прозрачно заворачивать сетевые запросы (`baseURL`) любых клиентов и агентов в единую точку**, интеллектуально разделяя трафик между прямым каналом и пулами различных прокси по имени запрашиваемой модели, исключая гео-блокировки и задержки двойного RTT.

---

## 1. Зачем нужен этот проект? (Проблема и Решение)

### Проблема 1. Разрозненность API-эндпоинтов и несовместимость путей
Клиенты и агентские инструменты (OpenCode CLI, DeepSeek Harness Cordis, Cursor, Aider, LiteLLM) работают по протоколу HTTP/SSE, но используют специфичные схемы путей:
- OpenCode требует префиксы `/v1` или `/go/v1`, которые затем транслируются в `/inference/...`.
- DeepSeek Harness ожидает пути `/zen/v1` или `/zen/go/v1`.
- Обычные библиотеки (OpenAI SDK, LangChain, curl) отправляют запросы в `/v1/chat/completions`.
- Ряд моделей (например, `Muse Spark 1.3 Contributor`) требуют эндпоинт Responses API (`/responses`), возвращая ошибку 400 при отправке в стандартный `/chat/completions`.

**Решение:** `model-georouter` предоставляет единый локальный порт `127.0.0.1:10830` со встроенными Ingress-адаптерами для каждого клиента. Достаточно прописать `baseURL: http://127.0.0.1:10830/<prefix>` в конфиге клиента — шлюз сам согласует пути, заголовки сессий и протоколы.

### Проблема 2. Гео-блокировки (403 RegionError) и ложные баны ключей
Ряд передовых моделей (например, `Muse Spark 1.3 Contributor`, `Claude 3.5/5.5`, `Gemini 3.8 Flash`) блокируют прямые запросы из определенных регионов (включая РФ), возвращая `403 Forbidden` / `RegionError`.
- В таких клиентах, как DeepSeek Harness, получение 403 от шлюза приводит к ложному срабатыванию валидатора: агент считает, что токен невалиден (`API key is invalid [AUTH]`), и аварийно завершает сессию.
- Стандартные сетевые ошибки (таймаут рукопожатия, обрыв туннеля) часто маскируются мостами под исходную ошибку, усугубляя проблему.

**Решение:** 
1. Шлюз на лету анализирует тело запроса (первые байты JSON) и мгновенно отправляет гео-зависимые модели в заранее выделенный прокси-пул (**Proxy-First**).
2. Ошибки транспорта прокси возвращаются клиенту со статусами `502 Bad Gateway` / `504 Gateway Timeout` и флагом `retryable: true`, исключая ложную инвалидацию API-ключей.

### Проблема 3. Двойной RTT и медленный первый токен (TTFT)
Простые локальные прокси работают по принципу *«попробуем напрямую, упадем с 403, затем повторим через прокси»*. Это приводит к тому, что каждый запрос к модели тратит 4–6 секунд на холостой цикл ожидания перед началом генерации.

**Решение:** Интеллектуальный движок маршрутизации (`ModelRuleEngine`) с префиксными и регулярными правилами направляет вызов сразу в целевой прокси-выход. Адаптивный TTL-кэш (`GeoCache`) запоминает факт блокировки новых неизвестных моделей на 24 часа.

---

## 2. Ключевые возможности

- 🚀 **Zero Dependencies:** Чистый Python 3.11+ (`socketserver`, `socket`, `ssl`, `json`, `select`). Никаких тяжелых фреймворков (FastAPI, aiohttp, requests).
- ⚡ **Zero-Buffering SSE Stream Relay:** Прозрачная передача Server-Sent Events с флагом `TCP_NODELAY`. Полная поддержка потоковых блоков рассуждений (`reasoning_content`) и prompt caching (`cached_tokens`).
- 🛡️ **Раздельные пулы слотов (Resource Isolation):** Независимые семафоры слотов (по умолчанию 16 direct / 16 proxy). Задержки или перегрузка внешних прокси-каналов не блокируют прямой трафик к незаблокированным моделям (DeepSeek, Qwen и др.).
- 🔄 **Мульти-прокси egress-матрица:**
  - `direct` — прямой выход через системный сетевой стек (0 оверхеда).
  - `http_connect` — HTTP CONNECT туннелирование через локальные или удаленные узлы (Xray, ProxyMarket, BrightData).
  - `socks5` / `socks5h` — SOCKS5 прокси с удаленным разрешением DNS (Hysteria 2, Shadowsocks).
- ⏱️ **Устойчивость к сбоям:** Circuit Breaker по штрафам узлов, 12-секундный таймаут подключения к прокси, единый монотонный дедлайн тела запроса (защита от Slowloris), безопасный drain сокетов.
- 🔒 **Безопасность Production-уровня:** Принудительные права `0600` на файлы логов и `0700` на каталоги, маскирование Bearer/Basic токенов, защита от HTTP Smuggling (TE+CL) и Path Traversal (`%2e%2e%2f`).
- 🛠️ **Hot-Reload конфигурации:** Динамическое обновление правил маршрутизации и эндпоинтов из `config.json` по `mtime` без перезапуска сервиса.

---

## 3. Архитектура системы

```
                         [ ВХОДЯЩИЕ КЛИЕНТЫ ]
   OpenCode CLI         DeepSeek Harness Cordis        Claude / Generic / curl
  (:10830/v1, /go/v1)     (:10830/zen/go/v1)          (:10830/v1/chat/completions)
          │                        │                              │
          └────────────────────────┼──────────────────────────────┘
                                   ▼
             ┌───────────────────────────────────────────┐
             │      Layer 1: Ingress Route Adapters      │
             │   (opencode.py / cordis.py / generic.py)  │
             └─────────────────────┬─────────────────────┘
                                   │ path rewrite + header clean
                                   ▼
             ┌───────────────────────────────────────────┐
             │         Layer 2: Core Model Router        │
             │       - Fast JSON Stream Inspector        │
             │       - Prefix / Regex Rule Matching      │
             │       - Adaptive GeoCache (TTL 24h)       │
             └─────────────────────┬─────────────────────┘
                                   │ target pool decision
                                   ▼
             ┌───────────────────────────────────────────┐
             │       Layer 3: Proxy Pool Manager         │
             │       - Direct Slots Semaphore (16)       │
             │       - Proxy Slots Semaphore (16)        │
             │       - Health & Penalty Circuit Breaker  │
             └─────────────────────┬─────────────────────┘
                                   │ socket connection
                                   ▼
 ┌─────────────────────────────────┼─────────────────────────────────┐
 │                                 │                                 │
 ▼                                 ▼                                 ▼
[ direct ]                   [ route-de ]                      [ route-uz ]
(Прямой выход)           (Xray HTTP CONNECT)             (BrightData HTTP CONNECT)
 │                                 │                                 │
 └─────────────────────────────────┼─────────────────────────────────┘
                                   ▼
             ┌───────────────────────────────────────────┐
             │        Layer 4: Streaming SSE Relay       │
             │     - Token-by-token relay (TCP_NODELAY)  │
             │     - Reasoning stream transparency       │
             │     - 403 RegionError Detection & Replay  │
             │     - Client Abort -> Socket Shutdown     │
             └───────────────────────────────────────────┘
```

---

## 4. Быстрый старт (Quick Start)

### 4.1. Установка окружения

Требуется **Python 3.11** или новее.

```bash
# Клонирование репозитория
git clone https://github.com/<your-username>/model-georouter.git
cd model-georouter

# Создание виртуального окружения
python3 -m venv .venv
source .venv/bin/activate

# Установка зависимостей для тестов (для рантайма зависимости не требуются!)
pip install pytest pytest-xdist
```

### 4.2. Настройка конфигурации

Создайте конфигурационный каталог и скопируйте шаблон настроек:

```bash
mkdir -p ~/.config/universal-ai-bridge/logs
cp config.example.json ~/.config/universal-ai-bridge/config.json
chmod 600 ~/.config/universal-ai-bridge/config.json
```

Пример базового `~/.config/universal-ai-bridge/config.json`:

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 10830,
    "max_connections": 256,
    "ingress_wait_timeout": 5.0,
    "request_head_timeout": 15.0,
    "body_timeout": 60.0
  },
  "slots": {
    "direct_slots": 16,
    "proxy_slots": 16
  },
  "pools": {
    "direct": {
      "type": "direct"
    },
    "route-de": {
      "type": "http_connect",
      "urls": ["http://127.0.0.1:10820"],
      "connect_timeout": 12.0
    }
  },
  "routing": {
    "default_pool": "direct",
    "geo_fallback_pool": "route-de",
    "rules": [
      { "prefix": "muse-", "pool": "route-de" },
      { "prefix": "meta/", "pool": "route-de" },
      { "prefix": "gpt-", "pool": "route-de" },
      { "prefix": "claude-", "pool": "route-de" },
      { "prefix": "deepseek-", "pool": "direct" },
      { "prefix": "qwen-", "pool": "direct" }
    ]
  },
  "logging": {
    "log_dir": "~/.config/universal-ai-bridge/logs",
    "log_level": "INFO"
  }
}
```

### 4.3. Запуск шлюза

**Запуск в терминале:**
```bash
python -m universal_ai_bridge --config ~/.config/universal-ai-bridge/config.json
```

**Запуск в фоне как демон macOS (LaunchAgent):**
Создайте файл `~/Library/LaunchAgents/com.user.universal-ai-bridge.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.user.universal-ai-bridge</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>-m</string>
        <string>universal_ai_bridge</string>
        <string>--config</string>
        <string>/Users/YOUR_USER/.config/universal-ai-bridge/config.json</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/universal-ai-bridge</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

Загрузите агент:
```bash
launchctl load ~/Library/LaunchAgents/com.user.universal-ai-bridge.plist
```

---

## 5. Подключение клиентов и AI-агентов

### 5.1. OpenCode CLI
В файле конфигурации `~/.config/opencode/opencode.json`:
```json
{
  "provider": {
    "opencode": {
      "options": {
        "baseURL": "http://127.0.0.1:10830/v1"
      }
    },
    "opencode-go": {
      "options": {
        "baseURL": "http://127.0.0.1:10830/go/v1"
      }
    }
  }
}
```
Теперь команда:
```bash
opencode run -m opencode-go/muse-spark-1.3-contributor "Привет, как дела?"
```
автоматически пойдет через немецкий прокси (`route-de`) без единой ошибки и без двойного RTT!

### 5.2. DeepSeek Harness (Cordis Desktop)
В профиле `~/.dsh/profiles/desktop/cordis.patch.yml`:
```yaml
- id: agent-default-model
  name: "@deepseek-ai/dsh-agent-default-model"
  config:
    provider: opencode-go-responses
    model: muse-spark-1.3-contributor
    baseURL: "http://127.0.0.1:10830/zen/go/v1"
```

### 5.3. Произвольный вызов через cURL / OpenAI SDK
```bash
curl -X POST http://127.0.0.1:10830/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "deepseek-v4.1-flash",
    "messages": [{"role": "user", "content": "Ping"}],
    "stream": true
  }'
```

---

## 6. Мониторинг и администрирование

- **Проверка здоровья (Health Check):**
  ```bash
  curl -s http://127.0.0.1:10830/health | jq .
  ```
  *Возвращает статус слотов, пулов и кэша гео-блокировок.*

- **Метрики шлюза (Prometheus-совместимый формат):**
  ```bash
  curl -s http://127.0.0.1:10830/metrics
  ```
  *Выводит `bridge_requests_total`, `bridge_replayed_total`, `bridge_active_slots`, `classified_403_truncated` и др.*

- **Сброс кэша гео-блокировок:**
  ```bash
  curl -s -X POST http://127.0.0.1:10830/cache/flush | jq .
  ```

---

## 7. Тестирование и надежность

Проект протестирован с помощью исчерпывающего набора из **306 модульных, интеграционных и e2e тестов**:

```bash
pytest -v
```

```text
============================= test session starts ==============================
collected 306 items

tests/test_adapters.py ................................................. [ 16%]
tests/test_config.py ...........................................         [ 30%]
tests/test_e2e_bridge.py .....................................           [ 42%]
tests/test_model_router.py ............................................. [ 57%]
.....................................................                    [ 74%]
tests/test_proxy_pool.py ..........................                      [ 83%]
tests/test_relay_and_errors.py ......................................... [ 96%]
...........                                                              [100%]
============================= 306 passed in 11.17s =============================
```

Покрытие включает:
- Симуляцию Slowloris-атак и обрыва клиентов на разных фазах SSE-стриминга.
- Защиту от HTTP Request Smuggling (конфликты `Transfer-Encoding` и `Content-Length`).
- Защиту от Path Traversal через экранированные последовательности (`%2e%2e%2f`).
- Атомарность записи GeoCache под высокой параллельной нагрузкой.
- Circuit breaker и штрафные интервалы при падении промежуточных узлов.

---

## 8. Лицензия

Проект распространяется под открытой лицензией [MIT](LICENSE).
Разработано для сообщества открытого AI-инструментария.
