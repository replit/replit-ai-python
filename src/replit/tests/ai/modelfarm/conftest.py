import pytest
from replit.ai.modelfarm import AsyncModelfarm, Modelfarm


@pytest.fixture
def client() -> Modelfarm:
    return Modelfarm()


@pytest.fixture
def async_client() -> AsyncModelfarm:
    return AsyncModelfarm()
