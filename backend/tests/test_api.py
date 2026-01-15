"""
Integration Tests for API Endpoints

Tests for authentication and analysis API workflows.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock
import io

from app.main import app
from app.database import get_db, init_db, Base, engine


@pytest_asyncio.fixture
async def setup_db():
    """Setup test database."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client(setup_db):
    """Create async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test basic health check."""
        response = await client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = await client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data


class TestAuthEndpoints:
    """Tests for authentication endpoints."""
    
    @pytest.mark.asyncio
    async def test_register_user(self, client):
        """Test user registration."""
        response = await client.post(
            "/auth/register",
            json={
                "email": "test@example.com",
                "password": "securepassword123",
                "full_name": "Test User"
            }
        )
        assert response.status_code == 201
        
        data = response.json()
        assert data["email"] == "test@example.com"
        assert "id" in data
    
    @pytest.mark.asyncio
    async def test_register_duplicate_email(self, client):
        """Test duplicate email registration fails."""
        # First registration
        await client.post(
            "/auth/register",
            json={
                "email": "duplicate@example.com",
                "password": "password123"
            }
        )
        
        # Second registration with same email
        response = await client.post(
            "/auth/register",
            json={
                "email": "duplicate@example.com",
                "password": "password456"
            }
        )
        assert response.status_code == 400
    
    @pytest.mark.asyncio
    async def test_login_success(self, client):
        """Test successful login."""
        # Register first
        await client.post(
            "/auth/register",
            json={
                "email": "login@example.com",
                "password": "password123"
            }
        )
        
        # Login
        response = await client.post(
            "/auth/login",
            json={
                "email": "login@example.com",
                "password": "password123"
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
    
    @pytest.mark.asyncio
    async def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials."""
        response = await client.post(
            "/auth/login",
            json={
                "email": "nonexistent@example.com",
                "password": "wrongpassword"
            }
        )
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_get_current_user(self, client):
        """Test getting current user info."""
        # Register and login
        await client.post(
            "/auth/register",
            json={
                "email": "me@example.com",
                "password": "password123",
                "full_name": "Current User"
            }
        )
        
        login_response = await client.post(
            "/auth/login",
            json={
                "email": "me@example.com",
                "password": "password123"
            }
        )
        token = login_response.json()["access_token"]
        
        # Get current user
        response = await client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["email"] == "me@example.com"
        assert data["full_name"] == "Current User"
    
    @pytest.mark.asyncio
    async def test_refresh_token(self, client):
        """Test token refresh."""
        # Register and login
        await client.post(
            "/auth/register",
            json={"email": "refresh@example.com", "password": "password123"}
        )
        
        login_response = await client.post(
            "/auth/login",
            json={"email": "refresh@example.com", "password": "password123"}
        )
        refresh_token = login_response.json()["refresh_token"]
        
        # Refresh
        response = await client.post(
            "/auth/refresh",
            json={"refresh_token": refresh_token}
        )
        assert response.status_code == 200
        assert "access_token" in response.json()


class TestAnalysisEndpoints:
    """Tests for analysis endpoints."""
    
    @pytest.fixture
    def create_test_image(self):
        """Create a test image for upload."""
        from PIL import Image
        
        img = Image.new('RGB', (100, 100), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        return buffer
    
    @pytest.mark.asyncio
    async def test_upload_requires_auth(self, client, create_test_image):
        """Test that upload requires authentication."""
        response = await client.post(
            "/analysis/upload",
            files={"file": ("test.png", create_test_image, "image/png")}
        )
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_history_requires_auth(self, client):
        """Test that history requires authentication."""
        response = await client.get("/analysis/history")
        assert response.status_code == 401


class TestRateLimiting:
    """Tests for rate limiting."""
    
    @pytest.mark.asyncio
    async def test_rate_limit_headers(self, client):
        """Test rate limit headers are present."""
        response = await client.get("/auth/me")
        # Even 401 responses should have rate limit headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
