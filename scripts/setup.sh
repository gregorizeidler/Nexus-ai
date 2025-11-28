#!/bin/bash

# AML-FORENSIC AI Setup Script

set -e

echo "================================================"
echo "  AML-FORENSIC AI - Installation Script"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ pip upgraded"

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt > /dev/null 2>&1
echo "✓ Python dependencies installed"

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data/raw data/processed logs models
echo "✓ Directories created"

# Copy environment file
echo ""
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✓ .env file created (please configure it)"
else
    echo "✓ .env file already exists"
fi

# Generate synthetic data
echo ""
echo "Generating synthetic test data..."
python scripts/generate_synthetic_data.py --num-transactions 500 --output data/raw/test_transactions.json
echo "✓ Synthetic data generated"

# Setup dashboard
echo ""
echo "Setting up React dashboard..."
cd dashboard

if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install > /dev/null 2>&1
    echo "✓ Dashboard dependencies installed"
else
    echo "✓ Dashboard dependencies already installed"
fi

cd ..

# Summary
echo ""
echo "================================================"
echo "  ✨ Installation Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Configure your environment:"
echo "   Edit .env file with your settings"
echo ""
echo "2. Start the API server:"
echo "   python scripts/run_api.py"
echo ""
echo "3. Start the dashboard:"
echo "   cd dashboard && npm run dev"
echo ""
echo "4. Access the system:"
echo "   - API: http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"
echo "   - Dashboard: http://localhost:3000"
echo ""
echo "5. Process test transactions:"
echo "   python scripts/process_batch.py --file data/raw/test_transactions.json"
echo ""
echo "================================================"

