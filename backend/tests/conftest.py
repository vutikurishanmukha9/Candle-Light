"""
Test Configuration

Pytest configuration and fixtures for backend tests.
"""

import pytest
import asyncio
from typing import Generator

# Configure pytest-asyncio to use auto mode
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
