"""Fixtures e dados sintéticos para testes end-to-end do pipeline de churn."""

from tests.fixtures.mock_responses import (
    MockBedrockResponses,
    MockNPAWResponses,
    MockSageMakerResponses,
)
from tests.fixtures.synthetic_users import (
    generate_active_users,
    generate_churned_users,
    generate_all_users,
    CHURNED_USER_IDS,
    ACTIVE_USER_IDS,
)

__all__ = [
    "MockBedrockResponses",
    "MockNPAWResponses",
    "MockSageMakerResponses",
    "generate_active_users",
    "generate_churned_users",
    "generate_all_users",
    "CHURNED_USER_IDS",
    "ACTIVE_USER_IDS",
]
