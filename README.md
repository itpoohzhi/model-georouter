# model-georouter

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-0%20pip%20runtime-brightgreen.svg)]()
[![CI](https://github.com/itpoohzhi/model-georouter/actions/workflows/ci.yml/badge.svg)](https://github.com/itpoohzhi/model-georouter/actions/workflows/ci.yml)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)]()

**model-georouter** is a lightweight, zero-dependency, strictly non-political local L7 reverse proxy and multi-proxy model router built exclusively on the **Python 3.11+ standard library** for borderless, unrestricted access to AI and LLM models worldwide.

It intercepts OpenAI-compatible API requests (`baseURL`), inspects the target `model` on the fly, and selectively routes traffic through designated proxy egress channels (HTTP CONNECT, SOCKS5) while letting unrestricted models flow directly over native network interfaces.

---

## Mission: Open & Borderless AI Access

**model-georouter is strictly non-political.** It takes no side in any dispute between states, governments, or companies.

We advocate for the use and development of large language models (LLMs) and AI tooling without discriminatory restrictions, regardless of geography, nationality, jurisdiction, or political borders. Knowledge, technology, and advanced development tools should belong to the global community of engineers and researchers on equal terms.

This project is a neutral piece of network plumbing: it routes each model request to the network path where that model works, so developers everywhere can build with the tools they choose.

---

## Why model-georouter?

### The Problem: Multi-Model Agent Orchestration & Geo-Restrictions

When orchestrating autonomous AI coding agents (such as OpenCode CLI, DeepSeek Harness, Claude Code, Cursor Agent, Factory Droid, or Aider), multi-agent pipelines frequently query heterogeneous model ensembles:
- Fast coding models (DeepSeek V4.1, Qwen 2.5) that work reliably via direct internet connections.
- Frontier reasoning models (Muse Spark 1.3 Contributor, Claude 3.5/5.5, Gemini 3.8 Flash) that are geo-restricted in specific countries and return `403 RegionError` or `403 Forbidden`.

Access to frontier LLM APIs is blocked or unsupported in many jurisdictions, whether through provider region policies, sanctions and export-control compliance, or local network filtering. The countries most commonly affected include:

| Country | Typical cause |
|---|---|
| China | Provider region policies and national network filtering |
| Russia | Provider region policies and sanctions-related restrictions |
| Iran | Sanctions and export-control compliance |
| North Korea | Sanctions and export-control compliance |
| Cuba | Sanctions and export-control compliance |
| Syria | Sanctions and export-control compliance |

The exact list varies by provider and changes over time; other regions are affected as well. Developers in these countries, and engineers who travel through them, regularly lose access to the tools their work depends on.

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
- Directs client-specific protocol adapters (e.g. `CloudCodeAdapter`) to upstream APIs with dedicated egress routing (e.g. `/cloudcode/*` $\rightarrow$ `https://daily-cloudcode-pa.googleapis.com` via `route-de`).
- Transparently relays chunked Server-Sent Events (SSE) with `TCP_NODELAY`, preserving extended reasoning streams (`reasoning_content`) and prompt cache hits (`cached_tokens`).
- Maintains an in-memory and persistent TTL GeoCache: if an unlisted model unexpectedly receives a 403 RegionError, the router transparently replays it through the fallback proxy pool and remembers it for 24 hours.

```mermaid
flowchart TD
    subgraph Clients["AI Agent Clients"]
        direction LR
        C1["OpenCode CLI"]
        C2["DeepSeek Harness"]
        C3["Antigravity CLI (agy)"]
        C4["Claude / Aider / curl"]
    end

    Router["model-georouter (127.0.0.1:10830)<br/>• Ingress Path Normalization & Adapters<br/>• Model JSON Extraction & Regex Routing<br/>• Response Rewriting & Geo-Sanitization<br/>• 24h Adaptive GeoCache"]

    C1 --> Router
    C2 --> Router
    C3 --> Router
    C4 --> Router

    subgraph DirectPool["Direct Pool (0ms proxy overhead)"]
        D1["• deepseek-*<br/>• qwen-*<br/>• glm-*"]
    end

    subgraph ProxyPools["Proxy Pools (Geo-bypass tunnels)"]
        P1["• route-de (Frankfurt Xray)<br/>• route-uz (Tashkent Proxy)<br/>• route-hy2 (Hysteria 2 SOCKS5)"]
    end

    Router -->|"Direct Traffic"| DirectPool
    Router -->|"Geo-Restricted & CloudCode"| ProxyPools

    DirectPool --> UpstreamDirect["Upstream AI Providers<br/>(Direct Internet)"]
    ProxyPools --> UpstreamProxy["Upstream AI Providers & Google APIs<br/>(via Proxy Egress)"]
```

### Protocol Adapters & Upstream Routing

| Adapter | Ingress Prefixes | Upstream Host | Egress Pool | Capabilities |
|---|---|---|---|---|
| `OpenCodeAdapter` | `/v1`, `/go/v1` | `opencode.ai:443` | Dynamic by model (`rules`) | Model extraction, streaming SSE |
| `CordisAdapter` | `/zen/v1`, `/zen/go/v1` | `opencode.ai:443` | Dynamic by model (`rules`) | Path normalization, streaming SSE |
| `GenericAdapter` | `/messages`, `/chat/completions` | `openrouter.ai:443` | Dynamic by model (`rules`) | Universal OpenAI/Anthropic format |
| `CloudCodeAdapter` | `/cloudcode` | `daily-cloudcode-pa.googleapis.com:443` | Dedicated (`route-de`) | Response rewriting & `loadCodeAssist` geo-sanitization |

---

## Key Features

- **Pure Standard Library:** Zero third-party Python runtime dependencies (`socketserver`, `socket`, `ssl`, `json`, `select`). Runs anywhere with Python 3.11+.
- **Zero-Latency First Token (TTFT):** Known geo-restricted models route directly to their designated proxy on the first attempt, eliminating the 4-second penalty of dead direct calls.
- **Response Rewriting & Geo-Sanitization (Google Cloud Code / Antigravity CLI):** Dedicated `CloudCodeAdapter` with selective downstream response transformation. Intercepts Google Cloud Code responses (`/v1internal:loadCodeAssist`), sanitizes false-positive `UNSUPPORTED_LOCATION` errors in `ineligibleTiers` when an active paid or current subscription tier is present, and recalculates downstream HTTP framing (`Content-Length`) while routing traffic through the Frankfurt proxy pool (`route-de`).
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
      "proxies": ["http://username:password@proxy.example.com:8080"]
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

**Default config location.** Without `--config` the router uses `~/.config/model-georouter/config.json` if it exists; otherwise the legacy `~/.config/universal-ai-bridge/config.json` if that one exists; otherwise it falls back to (and reports errors for) `~/.config/model-georouter/config.json`. The default `server.log_dir` follows the same order (`~/.config/model-georouter/logs`, then an already existing legacy `~/.config/universal-ai-bridge/logs`).

**Deprecated:** `server.ingress_wait_timeout` is ignored — connections above `max_connections` are rejected immediately with an inline `503`. The key is still validated so existing configs keep loading.

### 4. Running the Service

**Foreground:**
```bash
python3 -m model_georouter --config ~/.config/model-georouter/config.json
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
        <string>model_georouter</string>
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

### Google Antigravity CLI (agy)

#### The Problem: False Location Eligibility Rejection
Google Antigravity CLI (`agy`) communicates with Google's Cloud Code internal API (`daily-cloudcode-pa.googleapis.com`). On startup and before issuing agent prompts, `agy` performs an eligibility check by calling `/v1internal:loadCodeAssist` (handled in `server_oauth.go`).

For accounts subscribed to Google AI Pro or Ultra, Google's API response includes both the active paid tier and a legacy free tier marked as ineligible due to geographic location:
```json
{
  "currentTier": { "id": "tier-pro", "name": "Google AI Pro" },
  "ineligibleTiers": [
    { "id": "tier-free", "reasonCode": "UNSUPPORTED_LOCATION" }
  ]
}
```
The client-side logic iterates through all returned `ineligibleTiers`. Because `UNSUPPORTED_LOCATION` is present, `agy` terminates execution immediately:
```text
Eligibility check failed: ... location
```
Even when routing egress through a US or EU proxy, the account profile itself returns `tier-free` as unsupported, blocking all headless execution and agentic evaluations.

#### The Solution: Ingress Rewriting via `CloudCodeAdapter`
`model-georouter` provides built-in support for Google Cloud Code via `CloudCodeAdapter`:
1. **Targeted Ingress & Egress:** Requests targeting `/cloudcode/*` are normalized and forwarded via TLS to `daily-cloudcode-pa.googleapis.com` through the Frankfurt proxy pool (`route-de` at `127.0.0.1:10820`).
2. **On-the-Fly Sanitization:** When `/v1internal:loadCodeAssist` returns HTTP 200, the adapter inspects the JSON payload. If `currentTier` or `paidTier` is present, it strips any `ineligibleTiers` entries bearing `reasonCode: UNSUPPORTED_LOCATION`.
3. **Framing Normalization:** Decompresses gzip if present, recalculates `Content-Length`, strips `Transfer-Encoding: chunked`, and delivers a clean 200 OK response to the CLI.

#### Configuration
In `~/.config/model-georouter/config.json`, verify that the `route-de` egress pool is defined:

```json
{
  "pools": {
    "direct": { "type": "direct" },
    "route-de": {
      "type": "http_connect",
      "strategy": "first_available",
      "proxies": ["http://127.0.0.1:10820"]
    }
  },
  "adapters": {
    "cloudcode": {
      "prefixes": ["/cloudcode"],
      "upstream": "cloudcode-pa",
      "pool": "route-de"
    }
  }
}
```
*(Note: `cloudcode` adapter and upstream definitions are part of the default built-in configuration and work out of the box without manual declaration).*

#### Automated Launch Integration (`agy-cli.sh`)
The CLI launch script `~/.config/agy-launch/agy-cli.sh` automatically detects if `model-georouter` is listening on port 10830 and sets `CLOUD_CODE_URL`:

```bash
# Transparently route Cloud Code through model-georouter when active
GEOROUTER_PORT=10830
if (echo >"/dev/tcp/127.0.0.1/$GEOROUTER_PORT") 2>/dev/null; then
  export CLOUD_CODE_URL="${CLOUD_CODE_URL:-http://127.0.0.1:$GEOROUTER_PORT/cloudcode}"
fi

exec "$REAL" "$@"
```

When invoking `agy -p "Reply with exactly PONG"` or running agent judging panels, the request is transparently sanitized and completes successfully with exit code 0.

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

The test suite covers 393 automated test cases testing protocol conformance, concurrency, and failure recovery:

```bash
pytest -v
```

```text
============================= test session starts ==============================
collected 393 items

tests/test_adapters.py ................................................. [ 20%]
tests/test_config.py ...........................................         [ 31%]
tests/test_e2e_bridge.py .....................................           [ 41%]
tests/test_model_router.py ............................................. [ 66%]
tests/test_proxy_pool.py ..........................                      [ 73%]
tests/test_relay_and_errors.py ......................................... [ 100%]

============================= 393 passed in 12.45s =============================
```

---

## License

This project is licensed under the [MIT License](LICENSE).
