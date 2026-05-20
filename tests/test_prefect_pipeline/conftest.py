"""
Fixtures for the dependency-injected completely simulated pipeline.
"""

import pytest


@pytest.fixture(params=["lmfit"])
def fit_mode(request):
    return request.param
