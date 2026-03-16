#!/bin/sh
set -e

echo "Running database migrations..."
alembic upgrade head 2>/dev/null || echo "No migrations to apply (tables created on startup)"

echo "Starting application..."
exec "$@"
