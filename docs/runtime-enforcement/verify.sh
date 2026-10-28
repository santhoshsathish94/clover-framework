#!/bin/sh
set -eu

# Lightweight host-side verification for the reference workspace.
# Requires Docker and Docker Compose.

printf '%s\n' 'Running Clover runtime-enforcement reference...'
docker compose -f docker-compose.yml run --rm agent
printf '%s\n' 'Runtime enforcement check completed.'

cat <<'EOF'

What this demonstrates:
  - implementation files can be written
  - existing verification files are read-only at the container boundary
  - the reference write guard resolves real paths before policy matching

For development workflows, create new tests in a writable area and promote
trusted verification artifacts into the protected boundary before using them
as evidence of the Outcome.
EOF
