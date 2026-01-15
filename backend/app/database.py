"""
Database Configuration and Session Management

Supports both SQLite (development) and PostgreSQL (production).
Uses async SQLAlchemy for non-blocking database operations.
"""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import MetaData
from typing import AsyncGenerator

from app.config import settings


# Naming conventions for constraints (helps with migrations)
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}

metadata = MetaData(naming_convention=convention)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    metadata = metadata


# Create async engine with appropriate settings
def create_engine():
    """Create database engine based on configuration."""
    connect_args = {}
    
    # SQLite specific settings
    if settings.is_sqlite:
        connect_args["check_same_thread"] = False
    
    engine = create_async_engine(
        settings.database_url,
        echo=settings.debug,
        future=True,
        connect_args=connect_args if settings.is_sqlite else {},
        # PostgreSQL specific - connection pool settings
        pool_pre_ping=True if settings.is_postgres else False,
    )
    
    return engine


# Create engine instance
engine = create_engine()

# Create async session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides a database session.
    
    Usage in FastAPI:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """
    Initialize database - create all tables.
    
    Called on application startup.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """
    Close database connections.
    
    Called on application shutdown.
    """
    await engine.dispose()
