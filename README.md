# BharatTerminal

> **Local-first market intelligence and analytics for Indian equities and derivatives.**

BharatTerminal is a native desktop research terminal for the Indian equity and F&O markets. It combines a **Tauri desktop shell**, **React + TypeScript interface**, **Rust runtime**, and a **Python/FastAPI analytics service** to provide market data, options analytics, technical studies, screening, risk views, and research workflows from one local application.

The architecture is deliberately local-first: market data, analytical compute, credentials, and persistent state remain on the user's machine wherever possible.

> **Scope:** research and analytical workflows. BharatTerminal does not place live broker orders and is not financial advice.

---

## Highlights

- Native Windows desktop application built with **Tauri v2**
- Live NIFTY market-data ingestion through **Angel One SmartAPI**
- NSE option-chain and historical-data integration
- Technical indicators: **EMA, SMA, RSI, MACD, Bollinger Bands**
- Options analytics: **Greeks, IV, IV surface, skew, term structure**
- Strategy payoff and net-Greeks analysis
- Equity/F&O screening and market research views
- Futures basis and carry analysis
- Historical storage with **DuckDB**
- Short-lived live cache with **Redis**
- Alert/config persistence with **SQLite**
- Secure credential provisioning through **Windows Credential Manager**
- Signed release-integrity checks and fail-closed runtime behavior

---

## Architecture

```text
External Market Sources
│
├── Angel One SmartAPI / WebSocket
└── NSE public market endpoints
        │
        ▼
┌──────────────────────────────┐
│       Rust / Tauri Core      │
│                              │
│ • application lifecycle      │
│ • live-feed ingestion        │
│ • Tauri IPC                  │
│ • option-chain polling       │
│ • alert processing           │
│ • Python supervision         │
└──────────────┬───────────────┘
               │
     ┌─────────┼──────────┐
     ▼         ▼          ▼
  Redis      DuckDB     SQLite
 live data   history    alerts/config
               │
               ▼
┌──────────────────────────────┐
│   Python / FastAPI Compute   │
│                              │
│ • indicators                 │
│ • screeners                  │
│ • options analytics          │
│ • IV surfaces                │
│ • payoff / futures basis     │
│ • historical data            │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│    React + TypeScript UI     │
│ charts • options • risk      │
│ signals • models • research  │
└──────────────────────────────┘
```

React communicates with the local Tauri runtime through IPC. Rust reads local stores directly where appropriate and proxies analytical requests to the loopback FastAPI service on `127.0.0.1:8765`.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Desktop shell | Tauri v2 |
| Frontend | React 18, TypeScript, Vite |
| Frontend state | Zustand |
| Charts | Lightweight Charts, Plotly.js |
| Rust runtime | Tokio |
| Rust networking | reqwest, tokio-tungstenite |
| Python API | FastAPI, uvicorn |
| Options analytics | py_vollib, py_vollib_vectorized |
| Technical analysis | pandas, pandas-ta-openbb |
| Historical storage | DuckDB |
| Live cache | Redis |
| Alert/config storage | SQLite |
| Credential storage | Windows Credential Manager |

---

## Core Capabilities

### Market Data

- Angel One SmartAPI authentication
- NIFTY WebSocket ingestion
- NSE option-chain polling
- Historical OHLCV access
- Instrument-token resolution
- Local cache and persistence

### Technical Analysis

- Candlestick and volume views
- EMA / SMA
- RSI
- MACD
- Bollinger Bands
- Indicator overlays
- Screening workflows

### Options & Volatility

- Option-chain inspection
- Put-call ratio
- IV and Greeks
- IV surface
- Volatility skew
- Term structure
- Strategy payoff diagrams
- Net strategy Greeks

### Research & Risk

The application includes research-oriented modules for screeners, signals, risk, backtesting controls, futures basis, market intelligence, and model-status views.

Unavailable or insufficient data is intended to be represented honestly rather than replaced with fabricated values.

---

## Security Model

BharatTerminal is designed around explicit local trust boundaries.

### Credential handling

Broker and API credentials are provisioned through the operator CLI and stored using the platform credential backend. On Windows this is backed by **Windows Credential Manager / DPAPI**.

Credentials are not intended to be:

- stored in plaintext configuration files,
- passed through frontend state,
- embedded in Tauri IPC messages,
- supplied as normal command-line arguments,
- committed to Git.

### Runtime protections

The application uses:

- loopback-only internal services,
- authenticated Redis deployment,
- local API/process boundaries,
- fail-closed credential handling,
- release-resource integrity checks,
- signed release manifests,
- coordinated ownership for persistent database writes.

> Never commit real credentials, private signing material, runtime databases, or generated authentication files.

---

## Repository Layout

```text
.
├── README.md
├── package.json
├── pnpm-lock.yaml
├── vite.config.ts
├── tsconfig.json
├── index.html
├── src/                        # React + TypeScript frontend
│   ├── App.tsx
│   ├── main.tsx
│   ├── components/
│   ├── hooks/
│   └── store/
├── src-tauri/                  # Rust / Tauri runtime
│   ├── Cargo.toml
│   ├── build.rs
│   ├── tauri.conf.json
│   └── src/
└── compute/                    # Python analytical backend
    ├── main.py
    ├── db/
    ├── futures/
    ├── options/
    ├── technical/
    ├── ml/
    └── security/
```

---

## Prerequisites

For local development on Windows:

- Node.js 20+
- pnpm
- Rust stable
- Tauri CLI
- Python 3.11+
- uv
- Redis
- Visual Studio Build Tools with the C++ workload
- Microsoft WebView2 Runtime

Quick verification:

```powershell
node --version
pnpm --version
rustc --version
cargo --version
python --version
uv --version
redis-cli ping
```

---

## Development Setup

### 1. Frontend dependencies

```powershell
pnpm install
```

### 2. Python environment

```powershell
cd compute
uv sync --frozen --all-groups
cd ..
```

### 3. Rust dependencies

```powershell
cd src-tauri
cargo check
cd ..
```

---

## Redis Configuration

BharatTerminal expects a loopback-only Redis instance with authentication enabled.

Secret-free deployment templates live under:

```text
deploy/redis/
```

Provision private runtime copies and replace placeholders with operator-owned values. Runtime ACL files and credentials must remain outside Git.

Example authenticated local URL:

```text
redis://bharatterminal:<credential>@127.0.0.1:6379/0
```

Expose it to the application through `BHARATTERMINAL_REDIS_URL`.

---

## Secure Credential Provisioning

BharatTerminal has no browser-style login or demo credential form. Angel-dependent functionality remains unavailable until valid credentials are provisioned securely.

From the repository:

```powershell
cd compute
uv run --frozen python main.py credentials provision
```

For an installed release:

```powershell
.\bharatterminal-compute.exe credentials provision
```

Check availability without revealing values:

```powershell
.\bharatterminal-compute.exe credentials status
```

---

## Running Locally

### Full desktop app

```powershell
cargo tauri dev
```

### Python service only

```powershell
cd compute
uv run --project . --frozen python -m uvicorn main:app --host 127.0.0.1 --port 8765 --reload
```

### Frontend only

```powershell
pnpm dev
```

Vite listens on:

```text
http://127.0.0.1:5173
```

---

## Health Checks

```powershell
Invoke-RestMethod http://127.0.0.1:8765/health/live
Invoke-RestMethod http://127.0.0.1:8765/ohlcv/NIFTY
Invoke-RestMethod http://127.0.0.1:8765/chain/NIFTY
```

---

## Local Storage

### Redis

Representative short-lived keys:

| Key | Purpose | Typical TTL |
|---|---|---:|
| `tick:NIFTY` | Latest NIFTY tick | 15–60 s |
| `chain:NIFTY:{expiry}` | Cached option chain | 15 s |
| `pcr:NIFTY:{expiry}` | Put-call ratio | 15 s |

### DuckDB

Used for historical market data and analytical state, including OHLCV, F&O EOD history, and IV-related history.

### SQLite

Used for lightweight local state such as alerts and supporting runtime coordination.

---

## Data Contracts

### Tick

```text
symbol, token, ltp, open, high, low, close, volume, oi,
bid, ask, timestamp, exchange
```

### Option Leg

```text
symbol, expiry, strike, option_type, ltp, bid, ask, volume,
oi, oi_change, iv, delta, gamma, theta, vega
```

### Strategy Leg

```text
symbol, strike, expiry, option_type, action, quantity,
entry_price, lot_size
```

---

## Validation

### Frontend

```powershell
pnpm build
```

### Rust

```powershell
cd src-tauri
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cd ..
```

### Python

```powershell
cd compute
uv run --frozen pytest -q
cd ..
```

### Release

Use the repository's canonical release pipeline and release-integrity gate rather than manually assembling production artifacts.

A production release should verify:

- source-fingerprint coherence,
- frozen Python runtime integrity,
- frontend build success,
- Rust/Tauri release success,
- resource-manifest integrity,
- release-signature validity,
- installer generation,
- absence of developer-machine paths and secrets in release artifacts.

---

## Engineering Principles

1. **Fail closed on unavailable data.** Missing or unverified market/model data must not become plausible-looking values.
2. **Keep reads observational.** GET/read paths should not mutate persistent state.
3. **Coordinate database writes.** DuckDB mutation is controlled through the application's writer boundary.
4. **Keep internal services local.** Internal APIs are bound to loopback by default.
5. **Keep secrets out of application content.** Credentials belong in the OS credential store, not source, logs, frontend state, or release bundles.
6. **Separate source from generated artifacts.** Build products, caches, test databases, and local runtime state are not production source.
7. **Verify releases cryptographically and structurally.** Production artifacts must pass release-integrity gates.

---

## Troubleshooting

### Application remains in `CONNECTING`

Verify:

- Redis is running and authenticated.
- The local compute service is healthy.
- Angel credentials are provisioned.
- NIFTY instrument resolution succeeds.
- Collector/watchdog processes are healthy.
- The exchange/feed is available for the current session.

### Python service unavailable

```powershell
cd compute
uv run --project . --frozen python -m uvicorn main:app --host 127.0.0.1 --port 8765
```

### Chart has no data

```powershell
Invoke-RestMethod "http://127.0.0.1:8765/ohlcv/NIFTY"
```

### Option chain does not load

```powershell
Invoke-RestMethod "http://127.0.0.1:8765/chain/NIFTY"
redis-cli --askpass keys "chain:*"
```

NSE endpoints may rate-limit or change behavior.

---

## Project Status

BharatTerminal has evolved beyond an initial desktop scaffold into an integrated local research terminal with live-data, analytics, storage, security, packaging, and release-validation paths.

The project remains intentionally **research-oriented**. Live broker order placement is outside the current scope.

Ongoing development may include broader instrument coverage, deeper historical datasets, additional model validation, richer event-driven UI updates, expanded research tooling, and sustained-load performance work.

---

## Disclaimer

BharatTerminal is an engineering and market-research project.

- It does **not** constitute investment advice.
- Market data may be delayed, incomplete, unavailable, or affected by upstream providers.
- Analytical outputs and model results should be independently validated before financial decisions.
- Broker credentials and API keys remain the operator's responsibility.
- Live order execution is outside the current application scope.

---

## License

No license is declared in this README. Add an explicit project license before public redistribution or accepting external contributions under defined licensing terms.
