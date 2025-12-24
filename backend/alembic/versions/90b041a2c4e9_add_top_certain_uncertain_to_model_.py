"""add_top_certain_uncertain_to_model_snapshots

Revision ID: 90b041a2c4e9
Revises: a7b8c9d0e1f2
Create Date: 2025-12-22 11:20:46.518822

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "90b041a2c4e9"
down_revision: Union[str, Sequence[str], None] = "a7b8c9d0e1f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
