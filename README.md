# Triple Barrier Trading Strategy

## Overview

This repository contains a minimal yet extensible Python implementation of the **Triple Barrier Method** (TBM) for labeling trades and building rule‑based or ML‑driven trading strategies. TBM simultaneously applies an **upper profit‑taking barrier**, **lower stop‑loss barrier**, and a **time‑based (vertical) barrier** to each trade in order to determine its outcome. This approach allows robust, path‑independent labeling of price moves, mitigating look‑ahead bias and class‑imbalance issues common in financial ML.

> **Why Triple Barrier?**
>
> * Combines profit‑taking and risk management in a single framework.
> * Produces balanced, information‑rich labels for supervised learning.
> * Compatible with any asset class and frequency—from tick to daily.

## Features

* **ConfigManager** – YAML based configuration loader and saver.
* **YFinanceDownloader** – Downloads OHLCV data with optional caching.
* **TechnicalIndicator** – Calculates EMA, SMA, RSI, MACD, Bollinger Bands and MFI.
* **DataLabel** – Implements Triple Barrier labeling with risk management.
* **DataSplitUtils** – Simple train/validation date splitting helpers.
* **Per‑Ticker Z‑Score** – Normalize features independently per symbol.
* **LSTMModel** – Sequence classifier with configurable architecture.
* **LogManager** – Rotating file logs written to the `Logs/` folder.

## Quick Start

1. Adjust settings in `Params.yaml` to choose tickers and parameters.
2. Run `python main.py` to download data, compute indicators, label trades and train the model.
3. Check logs under `Logs/` and the saved model specified in `Params.yaml`.

## Tests

Run unit tests with `pytest UnitTests`.

