#!/usr/bin/env bash

set -e

if [ ! -d ".venv" ]; then
  python3 -m venv .venv || exit
fi

. .venv/bin/activate

pip install -r requirements.txt
