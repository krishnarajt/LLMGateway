"""allow provider api key rows to reference environment variables

Revision ID: 0002_provider_keys_from_env_vars
Revises: 0001_multiplexing_schema
Create Date: 2026-05-02 00:00:01.000000
"""

from __future__ import annotations

import os
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0002_provider_keys_from_env_vars"
down_revision: Union[str, None] = "0001_multiplexing_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _schema() -> str | None:
    schema = os.getenv("DB_SCHEMA", "public")
    return None if schema == "public" else schema


def _columns(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return {column["name"] for column in inspector.get_columns(table_name, schema=_schema())}


def upgrade() -> None:
    schema = _schema()
    columns = _columns("provider_api_keys")

    if "env_var_id" not in columns:
        op.add_column(
            "provider_api_keys",
            sa.Column("env_var_id", sa.Integer(), nullable=True),
            schema=schema,
        )
        if op.get_bind().dialect.name != "sqlite":
            op.create_foreign_key(
                "fk_provider_api_keys_env_var_id_environment_variables",
                "provider_api_keys",
                "environment_variables",
                ["env_var_id"],
                ["id"],
                source_schema=schema,
                referent_schema=schema,
                ondelete="SET NULL",
            )

    if "order_index" not in columns:
        op.add_column(
            "provider_api_keys",
            sa.Column(
                "order_index",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
            schema=schema,
        )

    with op.batch_alter_table("provider_api_keys", schema=schema) as batch_op:
        batch_op.alter_column(
            "encrypted_key",
            existing_type=sa.Text(),
            nullable=True,
        )


def downgrade() -> None:
    # Non-destructive for live databases that may now depend on Env Var backed keys.
    pass
