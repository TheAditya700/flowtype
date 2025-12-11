"""add_anonymous_user_fields

Revision ID: eb70214b29f3
Revises: d25472802491
Create Date: 2025-12-11 08:12:00.280609

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'eb70214b29f3'
down_revision: Union[str, Sequence[str], None] = 'd25472802491'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add anonymous user tracking fields
    op.add_column('users', sa.Column('is_anonymous', sa.Boolean(), nullable=False, server_default='true'))
    op.add_column('users', sa.Column('merged_into', sa.String(36), nullable=True))
    
    # Set existing users (with username) as non-anonymous
    op.execute("UPDATE users SET is_anonymous = false WHERE username IS NOT NULL")


def downgrade() -> None:
    """Downgrade schema."""
    # Remove anonymous user tracking fields
    op.drop_column('users', 'merged_into')
    op.drop_column('users', 'is_anonymous')
