"""remove legacy interactions from model_snapshots

Revision ID: cafe1234abcd
Revises: a7b8c9d0e1f2
Create Date: 2025-12-24 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "cafe1234abcd"
down_revision: Union[str, Sequence[str], None] = "a7b8c9d0e1f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Drop legacy interaction columns from model_snapshots."""
    with op.batch_alter_table("model_snapshots") as batch_op:
        batch_op.drop_column("top_positive_interactions")
        batch_op.drop_column("top_negative_interactions")


def downgrade() -> None:
    """Recreate legacy interaction columns on model_snapshots (nullable)."""
    with op.batch_alter_table("model_snapshots") as batch_op:
        batch_op.add_column(
            sa.Column("top_positive_interactions", sa.JSON(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("top_negative_interactions", sa.JSON(), nullable=True)
        )
