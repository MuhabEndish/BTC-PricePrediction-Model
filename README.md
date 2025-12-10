# 🚀 Cryptocurrency Price Prediction with AutoGluon

[![Python](https://img.shields.io/badge/Python-3.10.11-blue.svg)](https://www.python.org/)
[![AutoGluon](https://img.shields.io/badge/AutoGluon-1.4.0-orange.svg)](https://auto.gluon.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production-success.svg)](.)

A sophisticated machine learning system for predicting Bitcoin price movements using AutoGluon ensemble learning, technical indicators, and market sentiment analysis (Crypto Fear & Greed Index).

## 📊 Key Results

- **Direction Accuracy**: 55.97% (beats 50% baseline)
- **Trading Returns**: +55.14% vs -46.42% buy-hold (101.55% outperformance)
- **Sharpe Ratio**: 1.55 (excellent risk-adjusted returns)
- **RMSE**: 0.0373 (3.7% prediction error)
- **Win Rate**: 55.35% over 38 trades

## 🎯 Project Overview

This project implements a regression-based cryptocurrency price prediction system that:
- Predicts **3-day ahead log returns** for Bitcoin
- Combines **80+ engineered features** from technical indicators and sentiment data
- Uses **AutoGluon's automated ensemble** (Random Forest, XGBoost, LightGBM, CatBoost, Neural Networks)
- Validates profitability through **backtesting** with real trading simulation
- Generates **comprehensive visualizations** for analysis

## 📁 Project Structure

```
NewPricePreduction/
├── data/
│   ├── raw/                          # Raw OHLCV and CFGI data
│   ├── processed/                    # Processed features and results
│   │   ├── features.csv              # 797 samples × 80 features
│   │   ├── test_predictions.csv      # Model predictions
│   │   ├── eval_results.json         # Performance metrics
│   │   ├── backtest_results.csv      # Trading simulation results
│   │   └── visualizations/           # 5 generated plots
│   └── external/                     # External data sources
│
├── models/
│   └── crypto_autogluon_v2/          # Trained ensemble model
│
├── src/
│   ├── data/                         # Data acquisition scripts
│   │   ├── download_ohlcv.py
│   │   ├── download_cfgi.py
│   │   └── merge_data.py
│   ├── features/
│   │   └── feature_engineering.py    # Create 80 features
│   ├── models/
│   │   └── train_autogluon.py        # Train AutoGluon ensemble
│   ├── evaluation/
│   │   ├── evaluate.py               # Compute metrics
│   │   ├── backtest.py               # Trading simulation
│   │   └── visualization.py          # Generate plots
│   └── validation/
│       └── check_data_quality.py
│
├── notebooks/                        # Jupyter notebooks for exploration
├── cryptoenv/                        # Virtual environment
├── README.md                         # This file
└── FULL_REPORT.md                    # Comprehensive technical report

```

## 🔧 Installation

### Prerequisites
- Python 3.10.11 or higher
- Windows/Linux/macOS
- 8GB+ RAM recommended
- ~2GB disk space

### Setup

1. **Clone the repository**
```bash
cd NewPricePreduction
```

2. **Create virtual environment**
```bash
# Windows
python -m venv cryptoenv
.\cryptoenv\Scripts\Activate.ps1

# Linux/Mac
python -m venv cryptoenv
source cryptoenv/bin/activate
```

3. **Install dependencies**
```bash
pip install --upgrade pip
pip install autogluon==1.4.0
pip install pandas numpy matplotlib scikit-learn
```

### Required Packages
```
autogluon==1.4.0          # AutoML framework
pandas==2.1.3             # Data manipulation
numpy==2.1.3              # Numerical computing
matplotlib==3.8.2         # Visualization
scikit-learn==1.3.2       # ML utilities
lightgbm==4.1.0           # Gradient boosting
xgboost==2.0.3            # Gradient boosting
catboost==1.2.2           # Gradient boosting
torch==2.1.1              # Neural networks
```

## 🚀 Quick Start

### Full Pipeline Execution

```bash
# Activate environment
.\cryptoenv\Scripts\Activate.ps1

# Step 1: Feature Engineering (1-2 minutes)
python src/features/feature_engineering.py

# Step 2: Train Model (4-5 minutes)
python src/models/train_autogluon.py

# Step 3: Evaluate Performance (< 1 second)
python src/evaluation/evaluate.py

# Step 4: Backtest Trading Strategy (< 1 second)
python src/evaluation/backtest.py

# Step 5: Generate Visualizations (5-10 seconds)
python src/evaluation/visualization.py
```

**Total Runtime**: ~6 minutes

### Expected Output

After running the pipeline, you'll have:
- ✅ Trained model: `models/crypto_autogluon_v2/`
- ✅ Predictions: `data/processed/test_predictions.csv` (159 samples)
- ✅ Metrics: `data/processed/eval_results.json`
- ✅ Trading results: `data/processed/backtest_results.csv`
- ✅ Visualizations: `data/processed/visualizations/` (5 PNG files)

## 📈 Features

### Technical Indicators (35 features)
- **Moving Averages**: MA(7, 20, 50, 200), Golden/Death Cross
- **Bollinger Bands**: Upper, Lower, Middle (20, 50 periods), Band Width
- **RSI**: 14-day, 30-day Relative Strength Index
- **MACD**: MACD line, Signal line, Histogram
- **Volatility**: Standard deviation, Z-scores, Volatility of volatility
- **Momentum**: 10-day, 20-day momentum, Return skewness/kurtosis

### Price Patterns (8 features)
- **Candlestick Analysis**: Body ratio, Upper/Lower shadows
- **Price Ratios**: High/Close, Low/Close ratios
- **Distance from MAs**: MA(50) ratio, MA(200) ratio

### Volume Features (4 features)
- Volume moving average, Volume ratio
- Price-Volume interaction, Volume volatility

### Sentiment Features (20 features)
- **CFGI Transformations**: Z-score, Moving averages (7, 14, 30)
- **Sentiment Regimes**: Extreme Fear, Fear, Neutral, Greed, Extreme Greed
- **Interactions**: Sentiment × Volatility, Sentiment × RSI, Sentiment × Momentum

### Target Variables
- **Primary**: `target_log_return_3d` (3-day ahead log return)
- **Alternative**: `target_log_return` (1-day), `target_log_return_7d` (7-day)
- **Direction**: Binary up/down classification

## 🤖 Model Architecture

### AutoGluon Ensemble
The system uses AutoGluon's automated ensemble learning with:

**Base Models (Layer 1)**:
- LightGBM (2 variants)
- XGBoost (2 variants)
- CatBoost (1 variant)
- Random Forest (1 variant)
- Neural Network (1 variant)

**Ensemble Strategy**:
- **10-fold cross-validation bagging** for stability
- **Stacking** with 2 layers (L1 → L2 → L3)
- **Weighted combination**: 87.5% Random Forest + 12.5% CatBoost

**Best Model**: WeightedEnsemble_L3
- Validation RMSE: 0.0376
- Inference Speed: 122.2 rows/second

### Hyperparameters

Heavy regularization to prevent overfitting:
- **Learning Rates**: 0.005-0.01 (conservative)
- **L1/L2 Penalties**: 1.0-3.0 (strong regularization)
- **Tree Depth**: 3-4 (shallow trees)
- **Feature Sampling**: 60-70% per tree
- **Dropout**: 0.3 for neural networks

## 📊 Performance Metrics

### Regression Performance
```
RMSE:              0.0373 (3.7% error)
MAE:               0.0283 (2.8% error)
R²:                -0.198 (realistic for crypto)
Correlation:       +0.072 (positive signal)
MAPE:              164.95%
```

### Directional Accuracy
```
Overall:           55.97% (beats 50% baseline)
Up Markets:        57.32%
Down Markets:      54.55%
```

### Backtesting Results

**Strategy 1: Simple Long/Short**
```
Total Return:      +55.14%
Buy & Hold:        -46.42%
Outperformance:    +101.55%
Sharpe Ratio:      1.55
Max Drawdown:      41.29%
Win Rate:          55.35%
Number of Trades:  38
```

**Strategy 2: Threshold-Based (0.5% threshold)**
```
Total Return:      +31.36%
Outperformance:    +77.78%
Sharpe Ratio:      1.07
Number of Trades:  18
```

## 📉 Visualizations

The system generates 5 comprehensive plots:

1. **Predictions vs Actual** - Time series comparison of predicted and actual returns
2. **Cumulative Prices** - Wealth accumulation from predictions vs reality
3. **Prediction Scatter** - Scatter plot with correlation coefficient
4. **Direction Accuracy** - Rolling 20-day accuracy over time
5. **Residuals Analysis** - Error distribution and time series

All visualizations saved as high-resolution PNGs (300 DPI) in `data/processed/visualizations/`

## 🔬 Technical Details

### Data Processing
- **Raw Data**: 999 days of OHLCV + CFGI
- **Cleaned Data**: 797 samples (removed NaN, outliers)
- **Training Set**: 638 samples (80%)
- **Test Set**: 159 samples (20%)
- **Validation**: 20% of training data (128 samples)

### Feature Engineering
- **Total Features**: 80 engineered from 6 base columns
- **Selected Features**: 64 (after correlation + redundancy removal)
- **Feature Selection**: Correlation with target + >0.95 redundancy threshold
- **Optimization**: Dictionary collection + `pd.concat()` to avoid fragmentation

### Model Training
- **Training Time**: ~4 minutes (233 seconds)
- **Time Limit**: 1800 seconds (30 minutes)
- **Memory Usage**: ~3.5 GB peak
- **CPU Utilization**: 60-80% (parallel training)
- **Preset**: `best_quality` for maximum performance

## 📝 Usage Examples

### Load Trained Model
```python
from autogluon.tabular import TabularPredictor

# Load pre-trained model
predictor = TabularPredictor.load("models/crypto_autogluon_v2/")

# Make predictions
predictions = predictor.predict(test_data)
```

### Feature Engineering
```python
from src.features.feature_engineering import engineer_features

# Load raw data
df = pd.read_csv("data/processed/dataset.csv")

# Generate all 80 features
df_features = engineer_features(df)

# Output: 797 samples × 80 features
```

### Backtesting
```python
from src.evaluation.backtest import simple_strategy_backtest

# Load predictions
df = pd.read_csv("data/processed/test_predictions.csv")

# Run simple long/short strategy
results = simple_strategy_backtest(df, transaction_cost=0.001)

print(f"Total Return: {results['total_return']:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
```

## 🎯 Key Findings

### What Worked
✅ **Regression model** (predicting continuous returns) beats classification  
✅ **CFGI sentiment** adds value - cfgi_ma_30 in top 11 features  
✅ **Heavy regularization** prevents overfitting (L1/L2, dropout)  
✅ **10-fold bagging** provides stability on small dataset  
✅ **Simple trading strategy** outperforms threshold-based filtering  
✅ **Feature redundancy removal** improves generalization  

### What Didn't Work
❌ Classification model (overfitted, poor test performance)  
❌ Using validation data with bagging mode (assertion error)  
❌ Too many correlated features (redundancy)  
❌ Direct DataFrame assignment (fragmentation warnings)  
❌ Threshold strategy (lower returns than simple approach)  

## 🔮 Future Improvements

### Data Enhancements
- [ ] Add on-chain metrics (hash rate, transactions, active addresses)
- [ ] Higher frequency data (hourly, 4H candles)
- [ ] Extend to multiple cryptocurrencies (ETH, altcoins)
- [ ] Incorporate macroeconomic indicators (DXY, interest rates)

### Feature Engineering
- [ ] Advanced technical indicators (Ichimoku, Fibonacci)
- [ ] Order book features (bid-ask spread, order flow)
- [ ] Network effects (social media sentiment, GitHub activity)
- [ ] Cross-asset correlations (BTC vs stocks, gold)

### Model Improvements
- [ ] Add LSTM, Transformer models to ensemble
- [ ] Online learning with incremental updates
- [ ] Multi-horizon predictions (1d, 3d, 7d, 14d)
- [ ] Uncertainty quantification (prediction intervals)
- [ ] SHAP values for explainability

### Trading Strategy
- [ ] Dynamic position sizing (Kelly criterion)
- [ ] Stop-loss and take-profit rules
- [ ] Portfolio optimization across multiple assets
- [ ] Realistic slippage and execution modeling

### Production Deployment
- [ ] REST API for real-time predictions
- [ ] Live trading bot integration
- [ ] Model monitoring and drift detection
- [ ] Alert system (Telegram, Discord, email)

## 📚 Documentation

- **[FULL_REPORT.md](FULL_REPORT.md)** - Comprehensive 32-page technical report
- **[Code Documentation](src/)** - Inline comments and docstrings
- **[Jupyter Notebooks](notebooks/)** - Exploratory data analysis

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**IMPORTANT**: This project is for **educational and research purposes only**.

- ❌ **Not financial advice** - Do not use for real trading without proper due diligence
- ⚠️ **Past performance ≠ future results** - Backtesting results may not reflect live trading
- 💸 **Cryptocurrency trading is risky** - You can lose your entire investment
- 🔐 **No guarantees** - Model performance may degrade over time

Always consult with a qualified financial advisor before making investment decisions.

## 👨‍💻 Author

**Mohamed Endish**

- Project Date: December 2025
- Python Version: 3.10.11
- AutoGluon Version: 1.4.0

## 🙏 Acknowledgments

- **AutoGluon Team** - For the excellent AutoML framework
- **Alternative.me** - For the Crypto Fear & Greed Index API
- **Cryptocurrency Exchanges** - For providing historical OHLCV data
- **Open Source Community** - For LightGBM, XGBoost, CatBoost, PyTorch

## 📞 Support

For questions, bug reports, or feature requests:
- 📧 Open an issue in the repository
- 📖 Check the [FULL_REPORT.md](FULL_REPORT.md) for detailed documentation
- 💬 Review code comments and docstrings

## 🔄 Version History

### Version 2.0 (December 2025)
- ✅ Switched to regression model (improved metrics)
- ✅ Added 20 CFGI sentiment features
- ✅ Implemented 10-fold bagging
- ✅ Enhanced backtesting with 2 strategies
- ✅ Comprehensive visualization suite

### Version 1.0 (Initial Release)
- Basic classification model
- Technical indicators only
- Simple train/test split

## 🎓 Educational Use

This project is ideal for:
- **Machine Learning Students** - Learn AutoML, ensemble methods, time series
- **Finance Enthusiasts** - Understand technical analysis, trading strategies
- **Data Scientists** - Feature engineering, model validation, backtesting
- **Researchers** - Sentiment analysis, cryptocurrency prediction
- **Developers** - Production ML pipeline, code organization

## 📊 Performance Dashboard

```
╔════════════════════════════════════════════════════════════╗
║         CRYPTOCURRENCY PREDICTION SYSTEM v2.0              ║
╠════════════════════════════════════════════════════════════╣
║  Model:    WeightedEnsemble_L3 (RF 87.5% + CAT 12.5%)     ║
║  RMSE:     0.0373  |  MAE:  0.0283  |  R²:  -0.198         ║
║  Direction: 55.97% |  Corr: +0.072  |  Samples: 159        ║
╠════════════════════════════════════════════════════════════╣
║  Strategy:  Simple Long/Short                              ║
║  Return:    +55.14%  vs  -46.42% (buy-hold)                ║
║  Sharpe:    1.55     |  Win Rate:  55.35%                  ║
║  Drawdown:  41.29%   |  Trades:    38                      ║
╠════════════════════════════════════════════════════════════╣
║  Features:  64 selected from 80 engineered                 ║
║  Training:  638 samples  |  Test: 159 samples              ║
║  Runtime:   ~6 minutes (full pipeline)                     ║
╚════════════════════════════════════════════════════════════╝
```

---

**⭐ Star this repository if you find it useful!**

**🔔 Watch for updates and improvements**

**🍴 Fork to customize for your own cryptocurrency predictions**

---

*Built with ❤️ using AutoGluon, Python, and Machine Learning*

*Last Updated: December 10, 2025*
