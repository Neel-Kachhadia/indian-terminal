# Persistence ownership

The FastAPI sidecar is the single writable owner of `terminal.db`. It holds an
OS-backed writer lease for its lifetime. Desktop Rust, the frontend, monitors,
and maintenance tools must use the authenticated sidecar API for writes; direct
readers must open DuckDB with `read_only=True`.

Schema upgrades are staged in a sibling database, validated, and atomically
swapped only after a rollback backup exists. Bootstrap/reference data, mutable
analytics, model artifacts, and application binaries have independent versions.
