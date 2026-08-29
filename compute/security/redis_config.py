from __future__ import annotations

import os
from urllib.parse import urlsplit

import redis


class RedisConfigurationError(RuntimeError):
    pass


def authenticated_redis_client(*, decode_responses: bool = False) -> redis.Redis:
    raw = os.environ.get("BHARATTERMINAL_REDIS_URL", "")
    harness_id = os.environ.get("BHARATTERMINAL_TEST_HARNESS_ID")
    harness_url = os.environ.get("BHARATTERMINAL_TEST_REDIS_URL")
    if harness_id and (not harness_url or raw != harness_url):
        raise RedisConfigurationError("test process rejected Redis endpoint outside its harness")
    parsed = urlsplit(raw)
    if (
        parsed.scheme != "redis"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or not parsed.password
    ):
        raise RedisConfigurationError("authenticated loopback Redis configuration required")
    return redis.Redis.from_url(
        raw,
        decode_responses=decode_responses,
        socket_connect_timeout=2,
        socket_timeout=2,
    )
