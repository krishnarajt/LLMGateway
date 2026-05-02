"""baseline existing schema and add model multiplexing tables

Revision ID: 0001_multiplexing_schema
Revises:
Create Date: 2026-05-02 00:00:00.000000
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001_multiplexing_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    os.environ.setdefault("SECRET_KEY", "alembic-migration-placeholder")

    from app.db.database import Base  # noqa: WPS433
    from app.db import models  # noqa: F401,WPS433

    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        Base.metadata.schema = None
        for table in Base.metadata.tables.values():
            table.schema = None
    else:
        schema = os.getenv("DB_SCHEMA", "public")
        if schema != "public":
            bind.execute(sa.text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))

    Base.metadata.create_all(bind=bind, checkfirst=True)


def downgrade() -> None:
    # Intentionally non-destructive: this baseline may be run against live
    # databases that predate Alembic, so downgrade must not drop user data.
    pass
