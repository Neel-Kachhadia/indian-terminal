# Bandit medium-severity review

Review date: 2026-08-27. Command authority: Bandit 1.8.6 from `pyproject.toml`
and `uv.lock`. Scope: all production Python plus production build/export/release
scripts. Result: 0 HIGH, 44 MEDIUM, 28 LOW. Medium findings are 43 B608 and one
B310. No finding is suppressed, so changed queries remain visible on every run.

Classification totals:

- TRUE_POSITIVE: 0
- CONTROLLED_SAFE_PATTERN: 44
- FALSE_POSITIVE: 0
- UNRESOLVED_MEANINGFUL: 0

## Per-finding classification

Every entry below is `CONTROLLED_SAFE_PATTERN`. SQL values are bound separately;
interpolation constructs only placeholder counts or fragments selected from
closed local sets. B310 uses a fixed loopback scheme/host and locally selected
port.

| Location reported by Bandit | Control proof |
| --- | --- |
| `compute/auto_collector/supervision.py:510` | Assignment columns come only from the seven-name literal tuple; owner clause is a fixed literal; values are bound. |
| `compute/backtest/engine.py:83` | Expiry clause is empty or one fixed predicate; values are bound. |
| `compute/backtest/engine.py:118` | Both `IN` shapes contain generated `?` markers only; values are bound. |
| `compute/backtest/engine.py:137` | Selected column must pass `_bif_column`'s closed allowlist; values are bound. |
| `compute/backtest/engine.py:155` | Same `_bif_column` allowlist and bound values. |
| `compute/backtest/options_pricer.py:130` | Selected source/quality expressions come from fixed schema-presence branches; lookup values are bound. |
| `compute/backtest/options_pricer.py:284` | Expiry filter is empty or one fixed predicate; values are bound. |
| `compute/backtest/options_pricer.py:478` | `IN` shape is generated `?` markers; `LIMIT` is a bounded local integer. |
| `compute/backtest/options_pricer.py:520` | Both `IN` shapes are generated `?` markers; values are bound. |
| `compute/backtest/options_pricer.py:562` | Both `IN` shapes are generated `?` markers; values are bound. |
| `compute/data/error_logger.py:144` | `where` is empty or fixed `WHERE resolved = 0`; limit is bound. |
| `compute/data/stock_snapshot.py:232` | Query builder emits generated `?` markers only; symbols are bound. |
| `compute/db/migrations.py:667` | Table identifiers come only from migration validator's closed `required_tables` set and are quoted. |
| `compute/db/write_service.py:901` | Table/column pair comes only from `_READ_DATASETS`; payload chooses a map key, never SQL text. |
| `compute/db/write_service.py:904` | Same closed `_READ_DATASETS` map; symbol is bound. |
| `compute/db/write_service.py:910` | Same closed `_READ_DATASETS` map; date is bound. |
| `compute/db/write_service.py:929` | `IN` shape is generated `?` markers; candidates are bound. |
| `compute/db/write_service.py:1074` | `IN` shape is generated `?` markers; symbols/date are bound. |
| `compute/db/write_service.py:1090` | `IN` shape is generated `?` markers; symbols/date are bound. |
| `compute/db/write_service.py:1106` | `IN` shape is generated `?` markers; symbols/date are bound. |
| `compute/db/write_service.py:1122` | `IN` shape is generated `?` markers; symbols/dates are bound. |
| `compute/db/write_service.py:1136` | `IN` shape is generated `?` markers; symbols are bound. |
| `compute/db/write_service.py:1163` | `IN` shape is generated `?` markers; symbols and ordering identity are bound. |
| `compute/db/write_service.py:1229` | Optional symbol fragment is empty or generated `?` markers; limit and symbols are bound. |
| `compute/db/write_service.py:1339` | Symbol fragment uses generated `?` markers; condition SQL is built only by field/operator/logic allowlists; values are bound. |
| `compute/db/write_service.py:1484` | `IN` shape is generated `?` markers; symbols/date are bound. |
| `compute/db/write_service.py:1581` | Both `IN` shapes are generated `?` markers; values are bound. |
| `compute/db/write_service.py:1597` | Both `IN` shapes are generated `?` markers; values are bound. |
| `compute/db/write_service.py:1616` | `IN` shape is generated `?` markers; values are bound. |
| `compute/db/write_service.py:1636` | `IN` shape is generated `?` markers; values are bound. |
| `compute/db/write_service.py:1649` | `IN` shape is generated `?` markers; values are bound. |
| `compute/db/write_service.py:1663` | `IN` shape is generated `?` markers; values are bound. |
| `compute/db/write_service.py:1852` | `IN` shape is generated `?` markers; aliases/dates are bound. |
| `compute/db/write_service.py:1870` | Both `IN` shapes are generated `?` markers; values are bound. |
| `compute/db/write_service.py:1886` | `IN` shape is generated `?` markers; symbols are bound. |
| `compute/ml/models/m21_ohlc_forecast.py:684` | `IN` shape is generated `?` markers; symbol candidates are bound. |
| `compute/ml/registry.py:147` | Columns are keys from a fixed-schema locally constructed registry record; values are bound. |
| `compute/ml/registry.py:187` | Predicate is fixed validation literals plus generated `?` markers; model IDs are bound. |
| `compute/operations/health.py:270` | Table comes only from the four literal health-probe table names. |
| `compute/operations/retention.py:88` | Table, column, cast come only from fixed retention policy map; cutoff is bound. |
| `compute/operations/retention.py:93` | Same fixed retention policy map and bound cutoff. |
| `compute/operations/retention.py:120` | `IN` shape is generated markers from DB-returned job IDs; IDs are rebound. |
| `compute/operations/retention.py:124` | Same generated markers and rebound job IDs. |
| `scripts/release_gate.py:386` | URL is fixed `http://127.0.0.1:{port}/openapi.json`; no caller-controlled scheme or host. |

Any future MEDIUM/HIGH not represented here is unresolved until separately
reviewed. Line changes also require regenerating this review before release.
