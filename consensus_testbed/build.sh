#!/usr/bin/env bash
# Builds the two static binaries used by the Docker image and the benchmark client.
# Uses a local Go (1.21+) if installed, otherwise builds inside the official golang Docker image.
set -e; cd "$(dirname "$0")"; mkdir -p bin
if command -v go >/dev/null 2>&1; then
  (cd src && CGO_ENABLED=0 go build -o ../bin/node ./cmd/node && CGO_ENABLED=0 go build -o ../bin/bench ./cmd/bench)
else
  docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -e GOCACHE=/tmp/gocache -e CGO_ENABLED=0 \
    -v "$PWD":/w -w /w/src golang:1.22 sh -c "go build -o ../bin/node ./cmd/node && go build -o ../bin/bench ./cmd/bench"
fi
echo "built bin/node and bin/bench"
