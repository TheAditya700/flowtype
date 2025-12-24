"""merge heads

Revision ID: 5cbee9bf7b00
Revises: 90b041a2c4e9, cafe1234abcd
Create Date: 2025-12-24 07:42:47.512859

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5cbee9bf7b00'
down_revision: Union[str, Sequence[str], None] = ('90b041a2c4e9', 'cafe1234abcd')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
