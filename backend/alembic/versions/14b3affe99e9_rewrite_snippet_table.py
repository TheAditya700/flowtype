"""rewrite snippet table

Revision ID: 14b3affe99e9
Revises: 
Create Date: 2025-11-30 16:43:15.966069

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'rewrite_snippets'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Drop old snippet table
    op.drop_table('snippets')

    # Recreate new snippet table
    op.create_table(
        'snippets',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('text', sa.String, nullable=False),
        sa.Column('words', sa.JSON, nullable=False),
        sa.Column('word_count', sa.Integer, nullable=False),
        sa.Column('features', sa.JSON, nullable=False),
        sa.Column('difficulty_score', sa.Float, nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
    )

    op.create_index('idx_word_count', 'snippets', ['word_count'])
    op.create_index('idx_difficulty_score', 'snippets', ['difficulty_score'])


def downgrade():
    # Drop new table
    op.drop_table('snippets')

    # Recreate old snippet structure (if you care)
    op.create_table(
        'snippets',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('words', sa.String, nullable=False),
        sa.Column('word_count', sa.Integer, nullable=False),
        sa.Column('difficulty_score', sa.Float, nullable=False),
        sa.Column('avg_word_length', sa.Float),
        sa.Column('punctuation_density', sa.Float),
        sa.Column('rare_letter_count', sa.Integer),
        sa.Column('bigram_rarity', sa.Float),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now())
    )
