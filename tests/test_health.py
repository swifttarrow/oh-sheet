import backend
from backend.contracts import SCHEMA_VERSION


def test_health_returns_ok_with_version_and_commit(client, monkeypatch):
    monkeypatch.setenv("COMMIT_SHA", "abc1234")
    import backend.api.routes.health as health_mod
    monkeypatch.setattr(health_mod, "_COMMIT_SHA", "abc1234")

    response = client.get("/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["schema_version"] == SCHEMA_VERSION
    assert body["version"] == backend.__version__
    assert body["commit"] == "abc1234"
