from __future__ import annotations

import asyncio
import hashlib
import json
import struct
import sys
from pathlib import Path


COMPUTE_ROOT = Path(__file__).resolve().parents[1]
if str(COMPUTE_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPUTE_ROOT))

from security import credential_manager  # noqa: E402


TOKEN = "99926000"
URL = "wss://smartapisocket.angelone.in/smart-stream"


def _q(packet: bytes, offset: int) -> int:
    return struct.unpack_from("<q", packet, offset)[0]


def _d(packet: bytes, offset: int) -> float:
    return struct.unpack_from("<d", packet, offset)[0]


def parse_quote(packet: bytes) -> dict[str, object]:
    if len(packet) != 123:
        raise ValueError(f"expected 123-byte QUOTE packet, received {len(packet)}")
    token = packet[2:27].split(b"\0", 1)[0].decode("ascii", errors="strict")
    return {
        "mode": packet[0],
        "exchange_type": packet[1],
        "token": token,
        "sequence_number": _q(packet, 27),
        "exchange_timestamp": _q(packet, 35),
        "last_traded_price": _q(packet, 43),
        "last_traded_quantity": _q(packet, 51),
        "average_traded_price": _q(packet, 59),
        "volume_trade_for_the_day": _q(packet, 67),
        "total_buy_quantity": _d(packet, 75),
        "total_sell_quantity": _d(packet, 83),
        "open_price_of_the_day": _q(packet, 91),
        "high_price_of_the_day": _q(packet, 99),
        "low_price_of_the_day": _q(packet, 107),
        "closed_price": _q(packet, 115),
    }


async def capture() -> int:
    credentials = credential_manager.get_angel_credentials()
    session: dict[str, object] = {}
    headers: dict[str, str] = {}
    try:
        import pyotp
        from SmartApi import SmartConnect
        import websockets

        smart = SmartConnect(credentials["ANGEL_API_KEY"])
        response = smart.generateSession(
            credentials["ANGEL_CLIENT_ID"],
            credentials["ANGEL_PASSWORD"],
            pyotp.TOTP(credentials["ANGEL_TOTP_SECRET"]).now(),
        )
        if not response or not response.get("status"):
            raise RuntimeError("Angel authentication rejected")
        session = dict(response.get("data") or {})
        jwt_token = str(session.get("jwtToken") or session.get("jwt") or "")
        feed_token = str(session.get("feedToken") or smart.getfeedToken() or "")
        if not jwt_token or not feed_token:
            raise RuntimeError("Angel session missing feed tokens")
        headers = {
            "Authorization": jwt_token if jwt_token.lower().startswith("bearer ") else f"Bearer {jwt_token}",
            "x-api-key": credentials["ANGEL_API_KEY"],
            "x-client-code": credentials["ANGEL_CLIENT_ID"],
            "x-feed-token": feed_token,
            "x-client-type": "WEB",
        }
        async with websockets.connect(
            URL,
            additional_headers=headers,
            open_timeout=15,
            close_timeout=5,
            ping_interval=20,
            ping_timeout=None,
            max_size=2 * 1024 * 1024,
        ) as socket:
            await socket.send(
                json.dumps(
                    {
                        "correlationID": "acceptance1",
                        "action": 1,
                        "params": {
                            "mode": 2,
                            "tokenList": [{"exchangeType": 1, "tokens": [TOKEN]}],
                        },
                    },
                    separators=(",", ":"),
                )
            )
            print("ANGEL_PROBE_CONNECTED=1", flush=True)
            valid = 0
            deadline = asyncio.get_running_loop().time() + 30
            while valid < 5:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                message = await asyncio.wait_for(socket.recv(), timeout=remaining)
                if isinstance(message, str):
                    print("ANGEL_CONTROL=" + message[:240], flush=True)
                    continue
                try:
                    parsed = parse_quote(message)
                except ValueError as error:
                    evidence = {
                        "length": len(message),
                        "sha256": hashlib.sha256(message).hexdigest(),
                        "packet_hex": message.hex(),
                    }
                    print(
                        "ANGEL_INVALID_PACKET="
                        + json.dumps(evidence, sort_keys=True)
                        + f" reason={error}",
                        flush=True,
                    )
                    continue
                evidence = {
                    "length": len(message),
                    "sha256": hashlib.sha256(message).hexdigest(),
                    "packet_hex": message.hex(),
                    "decoded": parsed,
                }
                print("ANGEL_PACKET=" + json.dumps(evidence, sort_keys=True), flush=True)
                if parsed["mode"] == 2 and parsed["exchange_type"] == 1 and parsed["token"] == TOKEN:
                    valid += 1
            print(f"ANGEL_VALID_PACKET_COUNT={valid}", flush=True)
            return 0 if valid >= 5 else 2
    except Exception as error:
        print(f"ANGEL_PROBE_ERROR={type(error).__name__}:{str(error)[:240]}", flush=True)
        return 1
    finally:
        credentials.clear()
        session.clear()
        headers.clear()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(capture()))
