#!/bin/sh
set -e

echo "Starting application..."
# Migrations are handled by app lifespan (create_all + alembic upgrade/stamp)
exec "$@"
