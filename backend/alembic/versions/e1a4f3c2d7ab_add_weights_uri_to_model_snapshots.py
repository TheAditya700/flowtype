"""add weights uri to model snapshots

Revision ID: e1a4f3c2d7ab
Revises: d25472802491
Create Date: 2025-12-22 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e1a4f3c2d7ab"
down_revision: Union[str, Sequence[str], None] = "f2d3c4b1a9ce"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add weights_uri to model_snapshots."""
    op.add_column(
        "model_snapshots",
        sa.Column("weights_uri", sa.String(), nullable=True),
    )


def downgrade() -> None:
    """Remove weights_uri from model_snapshots."""
    op.drop_column("model_snapshots", "weights_uri")
