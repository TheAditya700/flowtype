from app.routers.health import health_check


def test_health_check_returns_ok_status():
    assert health_check() == {"status": "ok"}
