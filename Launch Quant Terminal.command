#!/bin/sh
set -u

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$REPO_ROOT" || exit 1

if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN=python3.12
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
else
    echo "Python was not found."
    echo "Install Python 3.12 from https://www.python.org/downloads/ and launch again."
    echo
    printf "Press Enter to close..."
    read -r _
    exit 1
fi

"$PYTHON_BIN" scripts/launch/bootstrap.py
STATUS=$?

echo
if [ "$STATUS" -ne 0 ]; then
    echo "NEPSE Quant Terminal exited with status $STATUS."
    echo "Read the message above, fix the issue, then launch again."
else
    echo "NEPSE Quant Terminal closed."
fi
echo
printf "Press Enter to close this window..."
read -r _
exit "$STATUS"
