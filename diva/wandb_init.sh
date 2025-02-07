#!/bin/bash

# Prompt user for input
read -p "Enter your entity (username or organization): " ENTITY
read -p "Enter your project name (e.g., 'diva'): " PROJECT

# Create a Python file with the variables
cat <<EOL > wandb_config.py
ENTITY = '$ENTITY'
PROJECT = '$PROJECT'
EOL

echo "wandb_config.py created with the provided entity and project."