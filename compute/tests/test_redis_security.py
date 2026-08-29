import pytest

from security.redis_config import RedisConfigurationError, authenticated_redis_client


@pytest.mark.parametrize(
    "value",
    ["redis://127.0.0.1/0", "redis://:credential@192.0.2.1/0", "http://:credential@127.0.0.1/0"],
)
def test_redis_rejects_unauthenticated_or_non_loopback_configuration(monkeypatch, value) -> None:
    monkeypatch.setenv("BHARATTERMINAL_REDIS_URL", value)
    with pytest.raises(RedisConfigurationError):
        authenticated_redis_client()


def test_test_process_rejects_authenticated_loopback_outside_harness(monkeypatch) -> None:
    monkeypatch.setenv("BHARATTERMINAL_REDIS_URL", "redis://:other@127.0.0.1:6380/14")
    with pytest.raises(RedisConfigurationError, match="outside its harness"):
        authenticated_redis_client()
