"""rewrite snippet schema

Revision ID: 504b3839ec13
Revises: rewrite_snippets
Create Date: 2025-12-01 03:43:56.845617

"""
from alembic import op
import sqlalchemy as sa
from app.models.db_models import GUID

# revision identifiers
revision = 'rewrite_snippets_v2'
down_revision = 'rewrite_snippets'
branch_labels = None
depends_on = None


def upgrade():
    op.drop_table("snippets")

    op.create_table(
        "snippets",
        sa.Column("id", GUID(), primary_key=True),
        sa.Column("text", sa.String(), nullable=False),
        sa.Column("words", sa.JSON(), nullable=False),
        sa.Column("word_count", sa.Integer(), nullable=False),
        sa.Column("features", sa.JSON(), nullable=False),
        sa.Column("difficulty_score", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )


def downgrade():
    op.drop_table("snippets")
