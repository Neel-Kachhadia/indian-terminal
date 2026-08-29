from __future__ import annotations

import argparse
import getpass
import hmac
import os
from dataclasses import dataclass
import sys
from typing import Any

try:
    import keyring
except ImportError as error:  # pragma: no cover - startup dependency guard
    keyring = None
    _KEYRING_IMPORT_ERROR = error
else:
    _KEYRING_IMPORT_ERROR = None

SERVICE_PREFIX = "BharatTerminal"
ANGEL_KEYS = ("ANGEL_CLIENT_ID", "ANGEL_PASSWORD", "ANGEL_TOTP_SECRET", "ANGEL_API_KEY")
OPTIONAL_KEYS = ("GROQ_API_KEY",)


@dataclass
@dataclass(frozen=True)
class CredentialStatus:
    has_angel_credentials: bool
    has_groq_key: bool


def provision_credentials(payload: dict[str, Any], *, replace: bool = False) -> bool:
    """Atomically provision OS-backed secrets; preserve existing values by default."""
    existing = {key: get_secret(key) for key in (*ANGEL_KEYS, *OPTIONAL_KEYS)}
    desired: dict[str, str] = {}
    for key in (*ANGEL_KEYS, *OPTIONAL_KEYS):
        supplied = str(payload.get(key) or payload.get(_camel_key(key)) or "").strip()
        if existing[key] and not replace:
            desired[key] = str(existing[key])
        elif supplied:
            desired[key] = supplied
        else:
            desired[key] = str(existing[key] or "")
    missing = [key for key in ANGEL_KEYS if not desired[key]]
    if missing:
        raise ValueError("Missing required Angel credential fields")

    values = {
        key: value
        for key, value in desired.items()
        if value and (existing[key] is None or (replace and existing[key] != value))
    }
    if not values:
        return False

    previous = {key: existing[key] for key in values}
    written: list[str] = []
    try:
        for key, value in values.items():
            set_secret(key, value)
            written.append(key)
            stored = get_secret(key)
            if stored is None or not hmac.compare_digest(stored.encode("utf-8"), value.encode("utf-8")):
                raise RuntimeError(f"OS credential store verification failed for {key}")
    except Exception:
        for key in reversed(written):
            old_value = previous[key]
            if old_value is None:
                delete_secret(key)
            else:
                set_secret(key, old_value)
        raise
    return True


def status() -> CredentialStatus:
    return CredentialStatus(
        has_angel_credentials=all(bool(get_secret(key)) for key in ANGEL_KEYS),
        has_groq_key=bool(get_secret("GROQ_API_KEY")),
    )


def get_angel_credentials() -> dict[str, str]:
    credentials = {key: get_secret(key) or "" for key in ANGEL_KEYS}
    missing = [key for key, value in credentials.items() if not value]
    if missing:
        raise ValueError(f"Missing credentials: {', '.join(missing)}")
    return credentials


def get_groq_api_key() -> str | None:
    value = get_secret("GROQ_API_KEY")
    return value if value else None


def set_secret(key: str, value: str) -> None:
    _require_keyring()
    keyring.set_password(SERVICE_PREFIX, key, value)


def get_secret(key: str) -> str | None:
    _require_keyring()
    return keyring.get_password(SERVICE_PREFIX, key)


def delete_secret(key: str) -> None:
    _require_keyring()
    try:
        keyring.delete_password(SERVICE_PREFIX, key)
    except Exception:
        return


def _require_keyring() -> None:
    if keyring is None:
        raise RuntimeError(f"keyring unavailable: {_KEYRING_IMPORT_ERROR}")
    backend = keyring.get_keyring()
    priority = getattr(backend, "priority", 0)
    if priority is None or float(priority) <= 0:
        raise RuntimeError("no secure OS credential backend is available")
    if os.name == "nt" and not backend.__class__.__module__.startswith("keyring.backends.Windows"):
        raise RuntimeError(
            f"refusing non-Windows credential backend on Windows: {backend.__class__.__module__}"
        )


def _camel_key(key: str) -> str:
    parts = key.lower().split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bharatterminal-compute credentials")
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("status", help="Report bounded credential availability")
    provision = subparsers.add_parser("provision", help="Write credentials to Windows secure storage")
    provision.add_argument(
        "--replace",
        action="store_true",
        help="Deliberately replace already-provisioned values",
    )
    args = parser.parse_args(argv)

    try:
        if args.action == "status":
            current = status()
            print(f"Angel credentials: {'AVAILABLE' if current.has_angel_credentials else 'UNAVAILABLE'}")
            print(f"Groq credential: {'AVAILABLE' if current.has_groq_key else 'UNAVAILABLE'}")
            return 0

        payload = {
            "ANGEL_CLIENT_ID": getpass.getpass("Angel client ID: "),
            "ANGEL_PASSWORD": getpass.getpass("Angel password: "),
            "ANGEL_TOTP_SECRET": getpass.getpass("Angel TOTP seed: "),
            "ANGEL_API_KEY": getpass.getpass("Angel API key: "),
            "GROQ_API_KEY": getpass.getpass("Groq API key (optional): "),
        }
        changed = provision_credentials(payload, replace=bool(args.replace))
        for key in payload:
            payload[key] = ""
        print(f"Secure credential provisioning: {'STORED' if changed else 'PRESERVED'}")
        return 0
    except Exception as error:
        print(f"Secure credential operation failed: {type(error).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
