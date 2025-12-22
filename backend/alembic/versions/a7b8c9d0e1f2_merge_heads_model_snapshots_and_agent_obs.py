"""merge heads: model snapshots chain with agent observability chain

Revision ID: a7b8c9d0e1f2
Revises: 91d4c588b03b, e1a4f3c2d7ab
Create Date: 2025-12-22 06:45:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a7b8c9d0e1f2"
down_revision: Union[str, Sequence[str], None] = ("91d4c588b03b", "e1a4f3c2d7ab")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """No-op merge revision: consolidates branches into single head."""
    pass


def downgrade() -> None:
    """Downgrade not supported for merge revisions."""
    pass
