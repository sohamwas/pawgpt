#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Installing Dependencies ---"
pip install -r requirements.txt

echo "--- Creating Vector Database ---"
python scripts/1_create_vector_db.py

echo "--- Build Complete ---"
