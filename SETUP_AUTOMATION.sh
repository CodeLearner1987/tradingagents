#!/bin/bash

# TradingAgents Setup Automation Script
# For beginners - automates installation
# Usage: bash SETUP_AUTOMATION.sh

echo "🚀 TradingAgents Setup Automation"
echo "==================================="
echo ""

# Check Python installation
echo "[1/6] Checking Python..."
if ! command -v python &> /dev/null; then
    echo "❌ Python 3.12 not found. Please install from https://www.python.org/downloads/"
    exit 1
fi
PYTHON_VERSION=$(python --version)
echo "✅ Found: $PYTHON_VERSION"

# Check Git installation
echo ""
echo "[2/6] Checking Git..."
if ! command -v git &> /dev/null; then
    echo "❌ Git not found. Please install from https://git-scm.com/"
    exit 1
fi
echo "✅ Git found"

# Create virtual environment
echo ""
echo "[3/6] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Skipping..."
else
    python -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "[4/6] Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"

# Install dependencies
echo ""
echo "[5/6] Installing dependencies..."
echo "This may take 2-5 minutes..."
pip install --upgrade pip > /dev/null 2>&1
pip install . > /dev/null 2>&1
echo "✅ Dependencies installed"

# Verify installation
echo ""
echo "[6/6] Verifying installation..."
if python -c "import tradingagents" 2>/dev/null; then
    echo "✅ TradingAgents installed successfully!"
else
    echo "❌ Installation verification failed"
    exit 1
fi

echo ""
echo "==================================="
echo "✅ Setup Complete!"
echo ""
echo "Next steps:"
echo "1. Configure API key: cp .env.example .env"
echo "2. Edit .env with your OpenAI API key"
echo "3. Run: tradingagents"
echo ""
echo "For detailed guide, see: BEGINNER_GUIDE.md"
