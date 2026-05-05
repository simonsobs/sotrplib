"""
Fixtures for the dependency-injected completely simulated pipeline.
"""

import pytest


@pytest.fixture(params=["scipy", "lmfit"])
def fit_mode(request):
    return request.param
