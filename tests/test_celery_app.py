"""Verify the Celery app can be imported and configured."""
from backend.workers.celery_app import celery_app


def test_celery_app_name():
    assert celery_app.main == "ohsheet"


def test_celery_app_broker_from_settings():
    """Broker URL should come from settings.redis_url."""
    # Default is redis://localhost:6379/0
    assert "redis" in celery_app.conf.broker_url
