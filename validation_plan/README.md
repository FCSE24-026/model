# 4-Week Validation Plan (Runnable)

This folder contains a corrected runnable version of the 10-script validation workflow.

## Setup

```bash
cd /home/runner/work/model/model/validation_plan
python -m venv venv
source venv/bin/activate
pip install yfinance pandas numpy scikit-learn matplotlib ta
```

## Run order

```bash
python 01_collect_data.py
python 02_features.py
python 03_train_model.py
python 04_diagnostics.py
python 05_walk_forward.py
python 06_plot_results.py
python 07_test_other_pairs.py
python 08_live_signals.py
python 09_log_trade.py
python 10_analyze_trades.py
```

## What was fixed

- Replaced invalid numeric-module imports with `from features import ...`.
- Added missing `ta` dependency in setup instructions.
- Aligned model naming with actual classifier (`GradientBoostingClassifier`).
- Fixed diagnostics indexing and confidence-bin calculations.
- Added missing output directory creation (`results`, `signals`, `trades`).
- Added DAX40 + Gold in Week 3 test set, with data-period fallback for 5m limits.
