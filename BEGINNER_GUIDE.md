# 🚀 TradingAgents: Complete Beginner's Guide

**For Complete Beginners with Zero Experience**

---

## Table of Contents

1. [What is TradingAgents?](#module-1-what-is-tradingagents)
2. [Installation (No Experience Needed)](#module-2-installation)
3. [Running Your First Analysis](#module-3-first-analysis)
4. [Understanding the Bot Logic](#module-4-bot-logic)
5. [Building Your Own Token](#module-5-token-creation)
6. [Backtesting Your Strategy](#module-6-backtesting)
7. [Going Live](#module-7-production)
8. [Troubleshooting](#module-8-troubleshooting)

---

## MODULE 1: What is TradingAgents?

### 🤔 Simple Explanation

**TradingAgents** is an AI-powered system that works like a real investment firm:

```
┌─────────────────────────────────────────────────────────┐
│                   YOUR INVESTMENT FIRM                   │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  👤 Market Analyst     → Checks technical indicators    │
│  👤 Sentiment Analyst  → Reads social media             │
│  👤 News Analyst       → Monitors global news           │
│  👤 Fundamentals Expert→ Analyzes company financials    │
│                        ↓                                 │
│  👥 Research Team      → Bull vs Bear debate            │
│                        ↓                                 │
│  📊 Trader             → Makes BUY/SELL/HOLD decision   │
│                        ↓                                 │
│  ⚖️  Risk Manager       → Checks if trade is safe       │
│                        ↓                                 │
│  ✅ Portfolio Manager   → Approves or rejects trade     │
│                        ↓                                 │
│  💰 EXECUTE TRADE      → Buy or Sell stocks             │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### Key Concepts Explained Simply

**Sentiment**: How bullish (optimistic) or bearish (pessimistic) an analyst is
- 🟢 **Bullish** = Price will go UP → **BUY**
- 🔴 **Bearish** = Price will go DOWN → **SELL**
- 🟡 **Neutral** = Unclear → **HOLD**

**Confidence**: How sure the analyst is (0% to 100%)
- 50% = Not sure
- 80% = Pretty sure
- 95% = Very confident

**Position Size**: How many shares to buy
- Based on your portfolio size and confidence
- Example: $100k portfolio × 80% confidence × 10% = $8,000 position = ~67 shares at $120/share

**Risk Management**: Protecting your money
- **Stop Loss**: If price drops 2%, automatically exit (limit losses)
- **Take Profit**: If price rises 4%, automatically exit (lock in gains)

---

## MODULE 2: Installation

### Prerequisites (What You Need)

✅ **A computer** (Mac, Windows, or Linux)
✅ **Internet connection**
✅ **API keys** (we'll get these together)
✅ **30-60 minutes** of your time

### Step 1: Install Python

**Why?** TradingAgents is written in Python

**For Windows:**
1. Go to https://www.python.org/downloads/
2. Click "Download Python 3.12"
3. Run the installer
4. ✅ **IMPORTANT**: Check "Add Python to PATH"
5. Click "Install Now"

**For Mac:**
```bash
# Install Homebrew first
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Then install Python
brew install python@3.12
```

**For Linux:**
```bash
sudo apt-get update
sudo apt-get install python3.12
```

**Verify Installation:**
```bash
python --version
```

You should see: `Python 3.12.x`

### Step 2: Clone TradingAgents

```bash
# Open Terminal/Command Prompt and run:
git clone https://github.com/TauricResearch/TradingAgents.git
cd TradingAgents
```

### Step 3: Create Virtual Environment

This isolates your project so dependencies don't conflict with other projects.

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal line.

### Step 4: Install Dependencies

```bash
# Install required packages
pip install .
```

This installs all the libraries TradingAgents needs.

### Step 5: Get API Keys

TradingAgents needs API keys from LLM providers. Choose one (free tier available):

**Option A: OpenAI (Recommended for beginners)**
1. Go to https://platform.openai.com/signup
2. Sign up for free account
3. Go to https://platform.openai.com/api-keys
4. Create new API key
5. Copy the key (keep it secret!)

**Option B: Google Gemini (Free tier)**
1. Go to https://ai.google.dev/
2. Click "Get API Key"
3. Create new key

**Option C: Anthropic Claude (Limited free tier)**
1. Go to https://console.anthropic.com/
2. Sign up
3. Create API key

### Step 6: Configure API Key

Create a `.env` file in your TradingAgents folder:

```bash
# Create .env file
cp .env.example .env

# Edit .env with your API key
```

**On Windows (Notepad):**
```
OPENAI_API_KEY=sk-xxxxxxxxxxxx
```

**On Mac/Linux (Terminal):**
```bash
echo "OPENAI_API_KEY=sk-xxxxxxxxxxxx" >> .env
```

### Verify Setup

```bash
# Test that everything works
python -c "import tradingagents; print('✅ TradingAgents installed successfully!')"
```

---

## MODULE 3: Running Your First Analysis

### The Easy Way (CLI)

```bash
# Make sure you're in the TradingAgents directory
cd TradingAgents

# Activate virtual environment
source venv/bin/activate  # Mac/Linux
# or
venv\Scripts\activate  # Windows

# Run TradingAgents
tradingagents

# Or if that doesn't work:
python -m cli.main
```

### What Happens Next

1. **Select LLM Provider**: Choose OpenAI, Google, etc. (pick what you configured)
2. **Select Ticker**: Type `AAPL` (Apple) or `NVDA` (Nvidia)
3. **Select Date**: Enter today's date in format `2026-06-28`
4. **Select Analysts**: Choose which analysts to run (all 4 by default)
5. **Select Depth**: Choose "quick" or "deep" analysis

### Watch It Analyze

The system will:
```
[STEP 1] Analysts Gather Data...
  ✓ Market Analyst: Checking technical indicators
  ✓ Sentiment Analyst: Reading social media
  ✓ News Analyst: Monitoring news
  ✓ Fundamentals Analyst: Analyzing company

[STEP 2] Researchers Debate...
  Bull Researcher: "Strong upside potential!"
  Bear Researcher: "Caution on valuations"

[STEP 3] Trader Makes Decision...
  ACTION: BUY
  Quantity: 42 shares
  Confidence: 85%

[STEP 4] Risk Management Review...
  Risk Level: MEDIUM
  Stop Loss: $180.45
  Take Profit: $195.30

[STEP 5] Portfolio Manager Final Approval...
  APPROVED: YES ✓
  
Analysis complete!
```

---

## MODULE 4: Understanding the Bot Logic

### How the Decision Pipeline Works

**Simplified Flow:**

```
INPUT: Stock ticker (e.g., AAPL) + Date
   ↓
[ANALYST REPORTS] 
   Market   → Technical analysis (RSI, MACD)
   Sentiment → Social media sentiment
   News     → Global events impact
   Fundamentals → P/E ratio, growth
   ↓
[EACH REPORT CREATES SENTIMENT]
   Report 1: BULLISH (75% confidence)
   Report 2: BULLISH (80% confidence)
   Report 3: BEARISH (60% confidence)
   Report 4: BULLISH (85% confidence)
   ↓
[RESEARCHERS DEBATE]
   Bull: "3 out of 4 are bullish! BUY!"
   Bear: "One analyst is bearish, risky"
   ↓
[TRADER DECIDES]
   Bullish count: 3
   Bearish count: 1
   Sentiment strength: (3-1)/4 = 50% bullish
   → Decision: BUY
   ↓
[RISK MANAGER]
   Portfolio volatility: MEDIUM
   Max position: 7% of portfolio
   Stop loss: -2% from entry
   Take profit: +4% from entry
   ↓
[PORTFOLIO MANAGER]
   Is position within risk limits? YES
   → APPROVED
   ↓
OUTPUT: BUY 42 shares at $120.50
```

### Simple Decision Rules

**When to BUY:**
- More analysts are bullish than bearish
- Sentiment strength > 30%
- Market conditions are favorable

**When to SELL:**
- More analysts are bearish than bullish
- Sentiment strength < -30%
- Risk indicators are high

**When to HOLD:**
- Mixed signals from analysts
- Sentiment strength between -30% and +30%
- Uncertainty is high

---

## MODULE 5: Building Your Own Token

### What is a Token?

A **token** is a digital asset on the blockchain. Think of it like digital money or stock shares stored on a blockchain.

### Create an ERC-20 Token (Ethereum Standard)

**Step 1: Go to Remix IDE**

Visit: https://remix.ethereum.org/

**Step 2: Create New File**

Name it: `MyToken.sol`

**Step 3: Paste This Code**

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract MyToken is ERC20 {
    constructor(uint256 initialSupply) ERC20("My Trading Token", "MTK") {
        _mint(msg.sender, initialSupply * 10 ** decimals());
    }
}
```

**Step 4: Compile**

1. Click the **Solidity Compiler** icon (left sidebar)
2. Click **"Compile MyToken.sol"**
3. You should see a green checkmark

**Step 5: Deploy to Testnet**

1. Click **Deploy & Run Transactions** icon
2. Select **Environment**: "Injected Provider - MetaMask"
3. Click **"Connect Wallet"** (if not connected)
4. In the **Deploy** section:
   - Constructor parameter: `1000000` (creates 1 million tokens)
   - Click **"Deploy"**
5. MetaMask will ask to confirm - click **"Confirm"**

**Step 6: You Have Your Token!**

Your token contract is now on the blockchain!

---

## MODULE 6: Backtesting Your Strategy

### What is Backtesting?

Testing your strategy on **historical data** to see if it would have made money in the past.

### Simple Backtest Example

```python
# Save this as backtest.py

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
import yfinance as yf

# Configuration
TICKER = "AAPL"
START_DATE = "2025-01-01"
END_DATE = "2025-12-31"
INITIAL_CAPITAL = 100000  # $100k

# Download historical data
data = yf.download(TICKER, start=START_DATE, end=END_DATE)

print(f"Testing strategy on {TICKER} from {START_DATE} to {END_DATE}")
print(f"Initial capital: ${INITIAL_CAPITAL:,.2f}")
print(f"Trading days: {len(data)}")

# Run analysis on each month
portfolio_value = INITIAL_CAPITAL
trades = []

for date in data.index[::20]:  # Every 20 days
    date_str = date.strftime("%Y-%m-%d")
    
    # Run TradingAgents analysis
    ta = TradingAgentsGraph(debug=False, config=DEFAULT_CONFIG.copy())
    try:
        _, decision = ta.propagate(TICKER, date_str)
        
        # Extract trade info
        if decision:
            trades.append({
                "date": date_str,
                "action": decision.get("action", "HOLD"),
                "price": data.loc[date, "Close"]
            })
    except:
        pass

print(f"\nTotal trades executed: {len(trades)}")
for trade in trades[-5:]:  # Show last 5
    print(f"  {trade['date']}: {trade['action']} @ ${trade['price']:.2f}")
```

**Run it:**
```bash
python backtest.py
```

---

## MODULE 7: Going Live (Production)

### ⚠️ IMPORTANT: Start Small

Before risking real money:

1. ✅ Test on paper (simulated trading) for 2 weeks
2. ✅ Verify your strategy consistently makes sense
3. ✅ Start with $100-$500 real money only
4. ✅ Scale up gradually as you gain confidence

### Live Trading Setup

**Step 1: Connect to Real Broker**

Add to your config:

```python
config = DEFAULT_CONFIG.copy()
config["broker"] = "interactive_brokers"  # Or your broker
config["account_id"] = "YOUR_ACCOUNT_ID"
config["live_trading"] = True  # ⚠️ Real money!
```

**Step 2: Set Risk Limits**

```python
config["max_portfolio_risk"] = 0.02  # Risk max 2% per trade
config["max_position_size"] = 0.05  # Max 5% of portfolio per trade
config["daily_loss_limit"] = -0.10  # Stop trading if down 10% today
```

**Step 3: Monitor Carefully**

Set up alerts:
- Email on every trade
- Slack notifications
- Daily performance reports

---

## MODULE 8: Troubleshooting

### Common Issues & Solutions

**Issue 1: "ModuleNotFoundError: No module named 'tradingagents'"**

```bash
# Solution: Reinstall
pip install .
```

**Issue 2: "OpenAI API key invalid"**

```bash
# Check .env file exists
cat .env

# Verify key format (should start with sk-)
# If wrong, get new key from https://platform.openai.com/api-keys
```

**Issue 3: "No module named 'cli'"**

```bash
# Try alternative command
python -m tradingagents.cli.main

# Or check you're in right directory
pwd  # Should show TradingAgents folder
```

**Issue 4: "yfinance: No data found for ticker"**

```bash
# Use correct format:
# US stocks: AAPL, MSFT, TSLA
# International: 0700.HK (Hong Kong), 7203.T (Tokyo)
# Crypto: BTC-USD, ETH-USD

# Not supported: Made-up tickers, delisted stocks
```

**Issue 5: "API rate limit exceeded"**

```bash
# Wait 60 seconds, try again
# Or upgrade API tier ($5-20/month for more calls)
```

### Getting Help

1. **Official Docs**: https://github.com/TauricResearch/TradingAgents
2. **GitHub Issues**: https://github.com/TauricResearch/TradingAgents/issues
3. **Discord**: https://discord.com/invite/hk9PGKShPK
4. **ArXiv Paper**: https://arxiv.org/abs/2412.20138

---

## Quick Reference

### Installation Checklist

- [ ] Python 3.12 installed
- [ ] TradingAgents cloned
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] API key obtained
- [ ] `.env` file configured
- [ ] First analysis runs successfully

### First Steps Checklist

- [ ] Run first analysis with CLI
- [ ] Understand the output
- [ ] Read through bot logic
- [ ] Create a test token
- [ ] Run a backtest
- [ ] Start with paper trading
- [ ] Scale gradually to live trading

### Resources

| Resource | Link |
|----------|------|
| Official Repo | https://github.com/TauricResearch/TradingAgents |
| Documentation | https://github.com/TauricResearch/TradingAgents/wiki |
| Paper | https://arxiv.org/abs/2412.20138 |
| Demo Video | https://www.youtube.com/watch?v=90gr5lwjIho |
| Discord Community | https://discord.com/invite/hk9PGKShPK |

---

## Final Thoughts

🎉 **You're Ready to Start!**

Remember:
- ✅ Start small and learn gradually
- ✅ Always use stop losses
- ✅ Never invest money you can't afford to lose
- ✅ Test extensively before going live
- ✅ Ask questions in the community

**Happy trading! 🚀**

---

**Last Updated**: June 28, 2026
**For**: TradingAgents Framework
**Difficulty**: Beginner (No Experience Required)
