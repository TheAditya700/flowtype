"""
Template used by Alembic when generating new revisions.
This is a minimal file to satisfy alembic usage in this repo.
"""
% from alembic import util
%
"""Auto-generated migration script."""
from alembic import op
import sqlalchemy as sa

revision = '${revision}'
down_revision = ${down_revision}
branch_labels = None
depends_on = None

def upgrade():
    pass

def downgrade():
    pass
