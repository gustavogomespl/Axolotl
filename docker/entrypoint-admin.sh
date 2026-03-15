#!/bin/sh
set -e
API_URL="${API_URL:-http://backend:8000}"
sed -i "s|__API_URL__|${API_URL}|g" /usr/share/nginx/html/config.js
exec "$@"
