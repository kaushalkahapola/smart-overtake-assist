#!/bin/bash

# Smart Overtake Assist - macOS Setup Script
# This script automates the installation of dependencies and environment setup.

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting setup for Smart Overtake Assist on macOS...${NC}"

# 1. Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo -e "${RED}Homebrew not found. Installing Homebrew...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add brew to path for the current session if it's a new install
    if [[ $(uname -m) == "arm64" ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    else
        eval "$(/usr/local/bin/brew shellenv)"
    fi
else
    echo -e "${GREEN}Homebrew is already installed.${NC}"
fi

# 2. Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 not found. Installing Python via Homebrew...${NC}"
    brew install python
else
    echo -e "${GREEN}Python 3 is already installed.$(python3 --version)${NC}"
fi

# 3. Create Virtual Environment
echo -e "${BLUE}Setting up virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment 'venv' already exists. Skipping creation.${NC}"
else
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment 'venv' created.${NC}"
fi

# 4. Activate Virtual Environment and Install Dependencies
echo -e "${BLUE}Activating virtual environment and installing dependencies...${NC}"
source venv/bin/activate

if [ -f "requirements.txt" ]; then
    # Upgrade pip first
    pip install --upgrade pip
    
    # Install dependencies
    pip install -r requirements.txt
    
    # Check if torch is properly installed for Mac (MPS support often comes with standard pip install of torch now)
    echo -e "${GREEN}Dependencies installed.${NC}"
else
    echo -e "${RED}Error: requirements.txt not found!${NC}"
    exit 1
fi

# 5. Verify Model Directory
echo -e "${BLUE}Verifying data/models directory...${NC}"
if [ ! -d "data/models" ]; then
    echo -e "${BLUE}Creating data/models directory...${NC}"
    mkdir -p data/models
fi

# Check for YOLO weights
if [ ! "$(ls -A data/models)" ]; then
    echo -e "${RED}Warning: No model weights found in data/models.${NC}"
    echo -e "The application will attempt to download them automatically when first run."
else
    echo -e "${GREEN}Model weights found in data/models.${NC}"
fi

echo -e "\n${GREEN}Setup Complete!${NC}"
echo -e "To start the application, run:"
echo -e "${BLUE}source venv/bin/activate${NC}"
echo -e "${BLUE}python src/main.py${NC}"
