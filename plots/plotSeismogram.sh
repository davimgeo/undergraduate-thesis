#!/bin/bash 

printf '%s\n' "$@" | while read -r A; do
    python3 plotSeismogram.py "$A"
done
