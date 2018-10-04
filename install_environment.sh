#!/bin/bash
cd "$(dirname "$0")"

if command -v python3 &>/dev/null; then
    echo Python 3 is installed
    python3 -m pip install --user --upgrade pip
    python3 -m pip install --user virtualenv
    python3 -m virtualenv venv
    source "venv/bin/activate"
    pip install -r requirements.txt
else
    echo Python 3 is not installed
fi

