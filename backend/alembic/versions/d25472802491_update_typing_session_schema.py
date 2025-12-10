"""update_typing_session_schema

Revision ID: d25472802491
Revises: b49cf5793c8f
Create Date: 2025-12-10 04:59:39.080690

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd25472802491'
down_revision: Union[str, Sequence[str], None] = 'b49cf5793c8f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Drop obsolete tables
    op.drop_table('keystroke_events')
    op.drop_table('snippet_usage')
    
    # Drop old columns that no longer exist in the model
    op.drop_column('typing_sessions', 'words_typed')
    op.drop_column('typing_sessions', 'backspaces')
    op.drop_column('typing_sessions', 'final_wpm')
    op.drop_column('typing_sessions', 'accuracy')
    op.drop_column('typing_sessions', 'starting_difficulty')
    op.drop_column('typing_sessions', 'ending_difficulty')
    op.drop_column('typing_sessions', 'avg_difficulty')
    op.drop_column('typing_sessions', 'flow_score')
    
    # Add new columns
    op.add_column('typing_sessions', sa.Column('user_embedding', sa.JSON(), nullable=True))
    op.add_column('typing_sessions', sa.Column('snippet_ids', sa.JSON(), nullable=False, server_default='[]'))
    op.add_column('typing_sessions', sa.Column('snippet_embeddings', sa.JSON(), nullable=True))
    op.add_column('typing_sessions', sa.Column('keystroke_events', sa.JSON(), nullable=False, server_default='[]'))
    op.add_column('typing_sessions', sa.Column('actual_wpm', sa.Float(), nullable=True))
    op.add_column('typing_sessions', sa.Column('actual_accuracy', sa.Float(), nullable=True))
    op.add_column('typing_sessions', sa.Column('actual_consistency', sa.Float(), nullable=True))
    op.add_column('typing_sessions', sa.Column('predicted_wpm', sa.Float(), nullable=True))
    op.add_column('typing_sessions', sa.Column('predicted_accuracy', sa.Float(), nullable=True))
    op.add_column('typing_sessions', sa.Column('predicted_consistency', sa.Float(), nullable=True))
    op.add_column('typing_sessions', sa.Column('raw_wpm', sa.Float(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    # Remove new columns
    op.drop_column('typing_sessions', 'raw_wpm')
    op.drop_column('typing_sessions', 'predicted_consistency')
    op.drop_column('typing_sessions', 'predicted_accuracy')
    op.drop_column('typing_sessions', 'predicted_wpm')
    op.drop_column('typing_sessions', 'actual_consistency')
    op.drop_column('typing_sessions', 'actual_accuracy')
    op.drop_column('typing_sessions', 'actual_wpm')
    op.drop_column('typing_sessions', 'keystroke_events')
    op.drop_column('typing_sessions', 'snippet_embeddings')
    op.drop_column('typing_sessions', 'snippet_ids')
    op.drop_column('typing_sessions', 'user_embedding')
    
    # Restore old columns
    op.add_column('typing_sessions', sa.Column('flow_score', sa.Float(), nullable=True))
    op.add_column('typing_sessions', sa.Column('avg_difficulty', sa.Float(), nullable=True))
    op.add_column('typing_sessions', sa.Column('ending_difficulty', sa.Float(), nullable=True))
    op.add_column('typing_sessions', sa.Column('starting_difficulty', sa.Float(), nullable=True))
    op.add_column('typing_sessions', sa.Column('accuracy', sa.Float(), nullable=True))
    op.add_column('typing_sessions', sa.Column('final_wpm', sa.Float(), nullable=True))
    op.add_column('typing_sessions', sa.Column('backspaces', sa.Integer(), nullable=True))
    op.add_column('typing_sessions', sa.Column('words_typed', sa.Integer(), nullable=True))
