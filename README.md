# Indian Market Terminal

A local-first desktop market terminal for Indian equity and F&O research.

Indian Market Terminal is a native Tauri app with a React frontend, a Rust core,
and a Python FastAPI compute service. It is designed to bring live NIFTY data,
option-chain analytics, Greeks, IV surface views, technical indicators,
screeners, alerts, strategy payoff diagrams, and futures basis analysis into
one local desktop application.

> Status: active Phase 1 scaffold. This is not production trading software yet.

## What This App Does

- Runs as a native desktop app through Tauri v2.
- Uses React 18 and TypeScript for the terminal UI.
- Uses Rust for app startup, Tauri IPC, Redis-backed live data reads, token-free
  WebSocket ingestion, option-chain polling, and alert evaluation.
- Uses Python FastAPI for analytical compute, NSE data proxying, historical
  OHLCV loading, indicators, screeners, payoff, IV surface, and futures basis.
- Stores short-lived live data in Redis.
- Stores historical market data in DuckDB.
- Stores alert definitions in SQLite.

The project is local-first: market data cache, compute, and storage all run on
the user's machine.

## Current Implementation Status

Implemented or partially implemented:

- Tauri app shell and command registration.
- React terminal layout with panels for price, charts, options chain, IV
  surface, screener, alerts, and strategy lab.
- Rust startup path that launches background ingester, poller, alert engine, and
  Python subprocess.
- Angel One SmartAPI authentication uses only Windows secure credential storage;
  raw credentials and session tokens never cross into the frontend or Tauri IPC.
- NIFTY WebSocket ingestion resolves the current NSE index token from the canonical instrument master before authentication/subscription.
- Redis cache for `tick:NIFTY`, option chain, and PCR.
- NSE option-chain polling through the local Python `/chain/NIFTY` endpoint.
- Greeks enrichment through Python `/greeks`.
- DuckDB schema initialization and NIFTY OHLCV seeding.
- Lightweight Charts candlestick chart with EMA, RSI, MACD, and Bollinger Band
  overlays.
- Plotly IV surface, skew, term structure, payoff, and futures basis charts.
- SQLite-backed alerts with in-app banners and desktop notifications.

Known incomplete areas:

- `src-tauri/src/historical.rs` is still a placeholder.
- `get_chain_greeks`, `get_market_status`, `get_instrument_list`, and
  `download_historical` are currently stubs.
- Frontend panels poll for data instead of using a complete event-push model.
- Redis must be available locally.
- The automatic historical seed focuses on NIFTY index data, not complete
  NIFTY50 equity history.
- F&O EOD history is required before futures basis can show historical data.
- The app is research-only. Live order placement is out of scope for Phase 1.

## Architecture

```text
External data
  |-- Angel One SmartAPI WebSocket
  |-- NSE public option-chain and bhav endpoints
  v
Rust / Tauri core
  |-- app startup
  |-- shared state
  |-- token-free WebSocket ingester
  |-- option-chain poller
  |-- alert engine
  |-- Tauri IPC commands
  |-- Python subprocess launcher
  v
Local services and storage
  |-- Redis live cache
  |-- DuckDB market history
  |-- SQLite alerts/config
  |-- FastAPI compute service + secure Angel authentication/feed bridge
  v
React UI
  |-- chart panel
  |-- options chain
  |-- IV surface
  |-- screener
  |-- alerts
  |-- strategy lab
```

React talks to Rust through Tauri IPC commands. Rust either reads local stores
directly or proxies analytical requests to Python at `127.0.0.1:8765`.

## Tech Stack

| Layer | Technology |
|---|---|
| Desktop shell | Tauri v2 |
| Rust runtime | Tokio |
| Rust HTTP/WebSocket | reqwest, tokio-tungstenite |
| Frontend | React 18, TypeScript, Vite |
| State | Zustand |
| Charts | lightweight-charts, Plotly.js |
| Python API | FastAPI, uvicorn |
| Options math | py_vollib, py_vollib_vectorized |
| Indicators | pandas, pandas-ta-openbb |
| Storage | Redis, DuckDB, SQLite |

## Project Layout

```text
.
|-- README.md
|-- package.json
|-- pnpm-lock.yaml
|-- vite.config.ts
|-- tsconfig.json
|-- tailwind.config.ts
|-- index.html
|-- src/
|   |-- App.tsx
|   |-- main.tsx
|   |-- components/
|   |-- hooks/
|   `-- store/
|-- src-tauri/
|   |-- Cargo.toml
|   |-- build.rs
|   |-- tauri.conf.json
|   `-- src/
`-- compute/
    |-- main.py
    |-- requirements.txt
    |-- db/
    |-- futures/
    |-- options/
    `-- technical/
```

## Prerequisites

- Node.js 20+
- pnpm
- Rust stable
- Tauri CLI
- Python 3.11+
- Redis server
- Visual Studio Build Tools with C++ workload on Windows
- WebView2 Runtime on Windows

Quick checks:

```powershell
node --version
pnpm --version
rustc --version
cargo --version
python --version
redis-cli ping
```

`redis-cli ping` should return:

```text
PONG
```

## Setup

Install frontend dependencies:

```powershell
pnpm install
```

Create the canonical Python environment from the frozen lock:

```powershell
cd compute
uv sync --frozen --all-groups
cd ..
```

Check Rust dependencies:

```powershell
cd src-tauri
cargo check
cd ..
```

## Running Locally

Start a loopback-only Redis configured with ACL/password authentication, then
provide its authenticated URL through the protected process environment as
`BHARATTERMINAL_REDIS_URL`:

```powershell
redis-server C:\path\to\loopback-authenticated-redis.conf
```

Secret-free deployment examples live in
`deploy/redis/redis.conf.template` and `deploy/redis/users.acl.template`.
Provision private runtime copies, replace every `__...__` placeholder with an
absolute private path or generated credential, and set the protected process
environment to `redis://bharatterminal:<credential>@127.0.0.1:6379/0`. Redis
startup must fail if an unresolved placeholder remains. Generated ACL files and
credentials are operator-owned runtime state and must never enter Git.

Run the full desktop app:

```powershell
cargo tauri dev
```

Tauri dev mode starts the Vite server with:

```powershell
pnpm dev
```

The Rust app then attempts to launch:

```powershell
uv run --project compute --frozen python -m uvicorn main:app --host 127.0.0.1 --port 8765
```

Run the Python service manually during backend development:

```powershell
cd compute
uv run --project . --frozen python -m uvicorn main:app --host 127.0.0.1 --port 8765 --reload
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/health/live
```

Frontend-only mode:

```powershell
pnpm dev
```

The Vite server listens on:

```text
http://127.0.0.1:5173
```

## Secure Credential Provisioning

BharatTerminal has no application Login or Demo workflow. It opens directly to
the local dashboard. Angel-dependent features fail closed until credentials are
available in Windows Credential Manager.

Provision once through the canonical interactive operator CLI. Values are read
without terminal echo and never accepted as command-line arguments:

```powershell
cd compute
uv run --frozen python main.py credentials provision
```

For an installed release, run the installed sidecar from the application
installation directory:

```powershell
.\bharatterminal-compute.exe credentials provision
```

Existing secure values are preserved. Replacing them requires deliberate use of
`credentials provision --replace`. Check bounded availability without revealing
values using `credentials status`. Normal startup never reads plaintext TXT or
environment credential files. Never commit real credentials.

## Runtime Storage

### Redis

Used for short-lived live market cache.

| Key | Value | TTL |
|---|---|---:|
| `tick:NIFTY` | JSON serialized `Tick` | 15-60s |
| `chain:NIFTY:{expiry}` | JSON serialized `OptionLeg[]` | 15s |
| `pcr:NIFTY:{expiry}` | PCR float string | 15s |

### DuckDB

DuckDB path:

```text
~/.indian-terminal/data/terminal.db
```

Tables:

- `ohlcv`
- `fo_eod`
- `iv_history`

### SQLite

SQLite alert/config path:

```text
~/.indian-terminal/config.db
```

Table:

- `alerts`

## Frontend Modules

| Component | Purpose |
|---|---|
| `src/App.tsx` | Main terminal layout and quote polling |
| `src/components/ChartPanel` | Candles, volume, and indicator overlays |
| `src/components/OptionsChain` | Option chain table and PCR |
| `src/components/IVSurface` | IV surface, skew, term structure |
| `src/components/Screener` | Daily NIFTY50 screener UI |
| `src/components/Alerts` | Alert creation, listing, deletion |
| `src/components/StrategyLab` | Payoff diagrams and futures basis |
| `src/store/quotes.ts` | Latest quote store |
| `src/store/alerts.ts` | Alert list store |
| `src/store/chain.ts` | Chain store scaffold |

## Rust Modules

| File | Purpose |
|---|---|
| `src-tauri/src/main.rs` | Tauri setup, background tasks, Python launch |
| `src-tauri/src/state.rs` | Shared app state |
| `src-tauri/src/models.rs` | Shared serializable structs |
| `src-tauri/src/commands.rs` | Tauri IPC commands |
| `src-tauri/src/auth.rs` | Angel One env/TOTP authentication |
| `src-tauri/src/ingester.rs` | NIFTY SmartAPI WebSocket ingestion |
| `src-tauri/src/poller.rs` | Option-chain polling and Greeks enrichment |
| `src-tauri/src/alert_engine.rs` | SQLite alerts and trigger evaluation |
| `src-tauri/src/historical.rs` | Placeholder historical module |

## Python Modules

| File | Purpose |
|---|---|
| `compute/main.py` | FastAPI app and endpoint registration |
| `compute/db/duckdb_client.py` | DuckDB path and schema |
| `compute/db/bhav_loader.py` | NSE bhav copy download/load helpers |
| `compute/options/greeks.py` | IV and Greeks computation |
| `compute/options/iv_surface.py` | IV surface/skew/term structure shaping |
| `compute/options/payoff.py` | Strategy payoff and net Greeks |
| `compute/technical/indicators.py` | EMA, SMA, RSI, MACD, Bollinger Bands |
| `compute/technical/screener.py` | NIFTY50 daily screener |
| `compute/futures/basis.py` | Futures basis and annualised carry |

## Tauri Commands

| Command | Status | Description |
|---|---|---|
| `get_quote` | Implemented | Reads latest tick from Redis |
| `get_chain` | Implemented | Reads option chain from Redis |
| `get_pcr` | Implemented | Reads PCR from Redis |
| `get_ohlcv` | Implemented | Proxies to Python `/ohlcv/{symbol}` |
| `get_expiries` | Implemented | Proxies to Python `/expiries/{symbol}` |
| `get_iv_surface` | Implemented | Proxies to Python `/iv_surface/{symbol}` |
| `get_payoff` | Implemented | Proxies to Python `/payoff` |
| `get_futures_basis` | Implemented | Proxies to Python futures endpoint |
| `get_screener_results` | Implemented | Proxies to Python `/screener` |
| `get_indicators` | Implemented | Proxies to Python `/indicators` |
| `set_alert` | Implemented | Saves alert in SQLite |
| `delete_alert` | Implemented | Deletes alert from SQLite |
| `get_alerts` | Implemented | Loads alerts from SQLite |
| `get_chain_greeks` | Stub | Returns empty list |
| `get_market_status` | Stub | Returns closed |
| `get_instrument_list` | Stub | Returns empty list |
| `download_historical` | Stub | Returns queued status |

## FastAPI Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service health |
| POST | `/greeks` | Compute chain Greeks |
| POST | `/indicators` | Compute technical indicators |
| POST | `/screener` | Run screener |
| POST | `/payoff` | Compute strategy payoff |
| GET | `/futures/basis/{symbol}` | Futures basis |
| GET | `/chain/{symbol}` | NSE chain proxy |
| GET | `/expiries/{symbol}` | Next expiries |
| GET | `/iv_surface/{symbol}` | IV surface from Redis chain |
| GET | `/ohlcv/{symbol}` | Historical OHLCV |
| POST | `/ingest/bhav` | Insert OHLCV rows |
| POST | `/download/historical` | Start bhav download |
| GET | `/download/status` | Download progress |

## Key Data Contracts

### Tick

```text
symbol, token, ltp, open, high, low, close, volume, oi, bid, ask, timestamp, exchange
```

### OptionLeg

```text
symbol, expiry, strike, option_type, ltp, bid, ask, volume, oi, oi_change,
iv, delta, gamma, theta, vega
```

### Alert

```text
id, symbol, alert_type, threshold, condition, active, triggered_at, created_at
```

### StrategyLeg

```text
symbol, strike, expiry, option_type, action, quantity, entry_price, lot_size
```

## Build and Validation Commands

Build frontend:

```powershell
pnpm build
```

Check Rust:

```powershell
cd src-tauri
cargo check
```

Run Rust tests:

```powershell
cd src-tauri
cargo test
```

Compile Python files:

```powershell
cd compute
uv run --project . --frozen python -m py_compile main.py options\greeks.py options\payoff.py
```

## Troubleshooting

### "Connecting..." stays on screen

Check Redis and `tick:NIFTY`:

```powershell
redis-cli --askpass ping
redis-cli --askpass get tick:NIFTY
```

Also verify Angel One environment variables are present if relying on WebSocket
ticks.

### Python service is unavailable

Run it manually:

```powershell
cd compute
uv run --project . --frozen python -m uvicorn main:app --host 127.0.0.1 --port 8765
```

### Chart has no data

Check:

```powershell
Invoke-RestMethod "http://127.0.0.1:8765/ohlcv/NIFTY"
Invoke-RestMethod "http://127.0.0.1:8765/download/status"
```

### Chain does not load

Check:

```powershell
Invoke-RestMethod "http://127.0.0.1:8765/chain/NIFTY"
redis-cli keys "chain:*"
```

### Greeks are blank

Verify Python packages:

```powershell
cd compute
uv run --project . --frozen python -c "import py_vollib; print('py_vollib ok')"
```

Greeks may also be blank for illiquid legs or values rejected by validation.

## Roadmap

Phase 1 focuses on research and analysis:

- Live quotes.
- Historical OHLCV.
- Options chain.
- Greeks and IV.
- IV surface.
- Indicators and screener.
- Strategy lab.
- Alerts.

Future phases may include paper trading, live trading connectivity, and
backtesting, but live order placement is intentionally not part of this phase.

## Caveats

- This is not financial advice.
- Do not use unverified calculations for live trading.
- NSE public endpoints can rate-limit or change behavior.
- Some modules still use placeholders or partial implementations.
- Keep credentials out of Git.
