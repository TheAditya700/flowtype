"""create telemetry_snippet_raw table

Revision ID: 0001_create_telemetry_raw
Revises: 
Create Date: 2025-11-30 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision = '0001_create_telemetry_raw'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create telemetry_snippet_raw table
    op.create_table(
        'telemetry_snippet_raw',
        sa.Column('id', sa.String(length=36), primary_key=True, nullable=False),
        sa.Column('received_at', sa.DateTime(), nullable=False),
        sa.Column('payload', sa.JSON(), nullable=False),
        sa.Column('user_id', sa.String(length=36), nullable=True),
        sa.Column('session_id', sa.String(length=36), nullable=True),
        sa.Column('source', sa.String(), nullable=True),
    )


def downgrade():
    op.drop_table('telemetry_snippet_raw')
