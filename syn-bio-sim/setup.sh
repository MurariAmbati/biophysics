#!/bin/bash
#
# Setup script for Synthetic Biology Simulator
# Phase 1: Core kinetics + ODE solver + toggle switch demo
#

set -e  # Exit on error

echo "======================================================================"
echo "Synthetic Biology Simulator - Phase 1 Setup"
echo "======================================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $python_version"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Run tests
echo "Running unit tests..."
python -m pytest tests/ -v
echo "✓ All tests passed"
echo ""

# Create output directory
echo "Creating output directory..."
mkdir -p output
echo "✓ Output directory created"
echo ""

echo "======================================================================"
echo "Setup completed successfully!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the toggle switch demo:"
echo "  python demos/toggle_switch_demo.py"
echo ""
echo "To run tests:"
echo "  pytest tests/ -v"
echo "======================================================================"
