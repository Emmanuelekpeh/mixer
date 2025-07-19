"""initial

Revision ID: 0001_initial
Revises: 
Create Date: 2025-07-19
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0001_initial'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Tables already created via SQLAlchemy Base metadata.
    pass

def downgrade():
    # Downgrade not supported for initial revision.
    pass 