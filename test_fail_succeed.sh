#!/bin/bash
# Test script that fails first two times then succeeds
COUNTER_FILE="test_counter.txt"

if [ ! -f "$COUNTER_FILE" ]; then
    echo 0 > "$COUNTER_FILE"
fi

COUNTER=$(cat "$COUNTER_FILE")

if [ "$COUNTER" -lt 2 ]; then
    echo "Simulating failure (attempt $((COUNTER+1)))"
    echo $((COUNTER+1)) > "$COUNTER_FILE"
    exit 1
else
    echo "Simulating success on third attempt"
    rm "$COUNTER_FILE"
    exit 0
fi
