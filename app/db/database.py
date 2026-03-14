"""
Database engine, session factory, init_db with seeding of default admin and providers.
"""

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from app.utils.logging_utils import get_logger

# central configuration values
from app.common import constants

logger = get_logger(__name__)

DATABASE_URL = constants.DATABASE_URL

# Read schema from env, default to public
DB_SCHEMA = constants.DB_SCHEMA
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base(metadata=MetaData(schema=DB_SCHEMA))


def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create schema, tables, and seed default data."""
    # Step 1: Create schema in its own committed transaction
    with engine.connect() as conn:
        if DB_SCHEMA != "public":
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{DB_SCHEMA}"'))
            conn.commit()

    # Step 2: Now create tables (schema already exists)
    Base.metadata.create_all(bind=engine)

    # Step 3: Seed default data
    _seed_defaults()


def _seed_defaults():
    """
    Ensure there is always at least one admin user and the base providers exist.
    Called on every startup — idempotent.
    """
    # Import here to avoid circular imports
    from app.db.models import User, UserRole, Provider
    from app.services.auth_service import get_password_hash

    db = SessionLocal()
    try:
        # --- Default admin user ---
        admin_exists = db.query(User).filter(User.role == UserRole.admin).first()
        if not admin_exists:
            logger.info("No admin user found — creating default admin (admin / admin)")
            default_admin = User(
                username="admin",
                password_hash=get_password_hash("admin"),
                role=UserRole.admin,
                is_default_admin=True,
                must_change_password=True,
                display_name="Default Admin",
            )
            db.add(default_admin)
            db.commit()
            logger.info("Default admin user created successfully")
        else:
            logger.info("Admin user already exists — skipping default admin creation")

        # --- Default providers ---
        default_providers = [
            {
                "name": "openai",
                "display_name": "OpenAI",
                "base_url": "https://api.openai.com/v1",
                "provider_type": "openai",
            },
            {
                "name": "gemini",
                "display_name": "Google Gemini",
                "base_url": "https://generativelanguage.googleapis.com",
                "provider_type": "gemini",
            },
            {
                "name": "ollama",
                "display_name": "Ollama (Local)",
                "base_url": "http://localhost:11434",
                "provider_type": "ollama",
            },
        ]
        for prov in default_providers:
            existing = db.query(Provider).filter(Provider.name == prov["name"]).first()
            if not existing:
                db.add(Provider(**prov))
                logger.info(f"Seeded provider: {prov['display_name']}")
        db.commit()

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
