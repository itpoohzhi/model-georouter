# model-georouter

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-0%20pip%20runtime-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/tests-337%20passed-success.svg)]()
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)]()

**model-georouter** is a lightweight, zero-dependency local L7 reverse proxy and multi-proxy model router built exclusively on the **Python 3.11+ standard library**.

It intercepts OpenAI-compatible API requests (`baseURL`), inspects the target `model` on the fly, and selectively routes traffic through designated proxy egress channels (HTTP CONNECT, SOCKS5) while letting unrestricted models flow directly over native network interfaces.

---

## Why model-georouter?

### The Problem: Multi-Model Agent Orchestration & Geo-Restrictions

When orchestrating autonomous AI coding agents (such as OpenCode CLI, DeepSeek Harness, Claude Code, Cursor Agent, Factory Droid, or Aider), multi-agent pipelines frequently query heterogeneous model ensembles:
- Fast coding models (DeepSeek V4.1, Qwen 2.5) that work reliably via direct internet connections.
- Frontier reasoning models (Muse Spark 1.3 Contributor, Claude 3.5/5.5, Gemini 3.8 Flash) that are geo-restricted in specific countries (such as Russia) and return `403 RegionError` or `403 Forbidden`.

Traditional workarounds fail in production workflows:

1. **Routing all agent traffic through a global VPN or single proxy:**
   - Adds 200–500ms RTT overhead to every single token on models that do not need a proxy.
   - Saturates proxy bandwidth and introduces frequent socket disconnects.
   - Direct connection is inherently faster, more reliable, and free of third-party tunnel jitter.

2. **Naïve failover proxies ("try direct, then retry on 403"):**
   - Each request to a blocked model incurs a dead 2–4 second timeout waiting for the initial 403 response before replaying through a proxy.
   - In clients like DeepSeek Harness, receiving a 403 response triggers false-positive credential rejection (`API key is invalid [AUTH]`), crashing the agent session.

3. **Subprocess/Binary Wrapping:**
   - Wrapping agent CLI binaries inside container wrappers or custom wrapper scripts is brittle, non-portable, and rejected by agent frameworks that lock down binary execution paths.

---

## The Solution: Selective Model-Aware URL Interception

Instead of wrapping CLI binaries, **`model-georouter` intercepts network requests at the HTTP/SSE transport layer**.

Agents point their standard `baseURL` to `http://127.0.0.1:10830`. The router:
- Inspects incoming JSON request bodies (the first few kilobytes) to determine the exact `model`.
- Matches the model against configurable prefix and regex routing rules:
  - `muse-*`, `gpt-*`, `meta/*` $\rightarrow$ Routed immediately through high-speed Frankfurt proxies (`route-de`).
  - `claude-*`, `anthropic/*` $\rightarrow$ Routed through Uzbekistan or Amsterdam proxies (`route-uz`, `route-hy2`).
  - `deepseek-*`, `qwen-*`, `minimax-*` $\rightarrow$ Routed **directly** via host interfaces without proxy latency.
- Transparently relays chunked Server-Sent Events (SSE) with `TCP_NODELAY`, preserving extended reasoning streams (`reasoning_content`) and prompt cache hits (`cached_tokens`).
- Maintains an in-memory and persistent TTL GeoCache: if an unlisted model unexpectedly receives a 403 RegionError, the router transparently replays it through the fallback proxy pool and remembers it for 24 hours.

```
                           [ AI Agent Clients ]
       OpenCode CLI        DeepSeek Harness        Claude / Aider / curl
            │                     │                          │
            └─────────────────────┼──────────────────────────┘
                                  ▼
                     http://127.0.0.1:10830
            ┌──────────────────────────────────────────────┐
            │          model-georouter (Local L7)          │
            │   - Ingress Path Normalization               │
            │   - Model JSON Extraction                    │
            │   - Prefix / Regex Routing Table             │
            │   - 24h Adaptive GeoCache                    │
            └───────────────┬──────────────┬───────────────┘
                            │              │
      ┌─────────────────────┘              └─────────────────────┐
      ▼                                                          ▼
  [ Direct Pool ]                                         [ Proxy Pools ]
  (0ms proxy overhead)                                    (Geo-bypass tunnels)
  • deepseek-*                                            • route-de (Frankfurt Xray)
  • qwen-*                                                • route-uz (BrightData Tashkent)
  • glm-*                                                 • route-hy2 (Hysteria 2 SOCKS5)
      │                                                          │
      ▼                                                          ▼
  Upstream AI Providers                                   Upstream AI Providers
```

---

## Key Features

- **Pure Standard Library:** Zero third-party Python runtime dependencies (`socketserver`, `socket`, `ssl`, `json`, `select`). Runs anywhere with Python 3.11+.
- **Zero-Latency First Token (TTFT):** Known geo-restricted models route directly to their designated proxy on the first attempt, eliminating the 4-second penalty of dead direct calls.
- **Resource Isolation via Separate Slots:** Independent connection semaphores (default 16 direct / 16 proxy). A slow or stalling proxy connection will never starve or block direct traffic.
- **Transparent Streaming SSE:** Zero response buffering. Relays token chunks, thinking/reasoning blocks, and prompt caching statistics in real time.
- **Resilient Transport:** 12-second proxy connect timeouts, pre-send connection retries for single-proxy pools, and automatic health-penalty circuit breaking.
- **Protection Against False Bans:** Upstream transport failures yield standard `502 Bad Gateway` / `504 Gateway Timeout` with `retryable: true`, preventing agent credential validators from falsely revoking API keys.
- **Security Hardened:** Explicit `0600` file / `0700` directory permissions on log rotations, credential redaction (Bearer/Basic/userinfo tokens masked in logs), and built-in protection against HTTP Request Smuggling (TE+CL) and Path Traversal (`%2e%2e%2f`).
- **Live Hot-Reload:** Configuration updates in `config.json` reload automatically on file modification (`mtime`) without restarting the daemon.

---

## Quick Start

### 1. Requirements
- Python 3.11 or higher.
- macOS or Linux.

### 2. Installation
```bash
git clone https://github.com/itpooh/model-georouter.git
cd model-georouter

# Optional: Set up virtualenv for development and running the test suite
python3 -m venv .venv
source .venv/bin/activate
pip install pytest
```

### 3. Configuration
Create your runtime directory and base configuration:

```bash
mkdir -p ~/.config/model-georouter/logs
cp config.example.json ~/.config/model-georouter/config.json
chmod 600 ~/.config/model-georouter/config.json
```

Edit `~/.config/model-georouter/config.json` to define your egress proxy backends and model rules:

```json
{
  "server": {
    "listen": "127.0.0.1",
    "port": 10830,
    "max_connections": 256,
    "ingress_wait_timeout": 5.0,
    "body_timeout": 60.0,
    "log_dir": "~/.config/model-georouter/logs"
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
      "proxies": ["http://127.0.0.1:10820"]
    },
    "route-uz": {
      "type": "http_connect",
      "proxies": ["http://username:password@85.192.60.125:44445"]
    }
  },
  "routing": {
    "default_pool": "direct",
    "geo_fallback_pool": "route-de",
    "rules": [
      { "match_prefix": ["muse-", "meta/", "gpt-"], "pool": "route-de" },
      { "match_prefix": ["claude-"], "pool": "route-uz" },
      { "match_prefix": ["deepseek-", "qwen-"], "pool": "direct" }
    ]
  }
}
```

Both flat (`server.direct_slots`, top-level `rules`) and nested (`slots`, `routing`) layouts are accepted. Legacy aliases keep working: `host` → `listen`, `urls` → `proxies`, `prefix` → `match_prefix`. `max_connections` is re-applied on hot-reload without a restart.

### 4. Running the Service

**Foreground:**
```bash
python3 -m universal_ai_bridge --config ~/.config/model-georouter/config.json
```

**macOS LaunchAgent (Background Daemon):**
Create `~/Library/LaunchAgents/com.user.model-georouter.plist`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.user.model-georouter</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>-m</string>
        <string>universal_ai_bridge</string>
        <string>--config</string>
        <string>/Users/YOUR_USER/.config/model-georouter/config.json</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/model-georouter</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```
Load the daemon:
```bash
launchctl load ~/Library/LaunchAgents/com.user.model-georouter.plist
```

---

## Client Integration Examples

### OpenCode CLI
Point OpenCode to the local router in `~/.config/opencode/opencode.json`:
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
Now executing:
```bash
opencode run -m opencode-go/muse-spark-1.3-contributor "Explain quantum computing"
```
instantly routes through Germany, while:
```bash
opencode run -m opencode-go/deepseek-v4.1-flash "Write a quicksort in Python"
```
runs over the direct, low-latency connection.

### DeepSeek Harness (Cordis)
In your Cordis patch profile (`~/.dsh/profiles/desktop/cordis.patch.yml`):
```yaml
- id: agent-default-model
  name: "@deepseek-ai/dsh-agent-default-model"
  config:
    provider: opencode-go-responses
    model: muse-spark-1.3-contributor
    baseURL: "http://127.0.0.1:10830/zen/go/v1"
```

### Generic OpenAI SDK / cURL
```bash
curl -X POST http://127.0.0.1:10830/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "deepseek-v4.1-flash",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

---

## Administration & Metrics

- **Health Check:**
  ```bash
  curl -s http://127.0.0.1:10830/health | jq .
  ```
  Returns connection slot availability, proxy pool health, and active GeoCache entries.

- **Prometheus Metrics:**
  ```bash
  curl -s http://127.0.0.1:10830/metrics
  ```
  Exports metrics including total requests, replayed requests, active slots, and truncated 403 signatures.

- **Flush GeoCache:**
  ```bash
  curl -s -X POST http://127.0.0.1:10830/cache/flush | jq .
  ```

---

## Testing & Verification

The test suite covers 337 automated test cases testing protocol conformance, concurrency, and failure recovery:

```bash
pytest -v
```

```text
============================= test session starts ==============================
collected 337 items

tests/test_adapters.py ................................................. [ 16%]
tests/test_config.py ...........................................         [ 30%]
tests/test_e2e_bridge.py .....................................           [ 42%]
tests/test_model_router.py ............................................. [ 57%]
tests/test_proxy_pool.py ..........................                      [ 83%]
tests/test_relay_and_errors.py ......................................... [ 100%]

============================= 337 passed in 11.17s =============================
```

---

## License

This project is licensed under the [MIT License](LICENSE).
