# Triple Barrier Trading Strategy

## Overview

This repository contains a minimal yet extensible Python implementation of the **Triple Barrier Method** (TBM) for labeling trades and building rule‑based or ML‑driven trading strategies. TBM simultaneously applies an **upper profit‑taking barrier**, **lower stop‑loss barrier**, and a **time‑based (vertical) barrier** to each trade in order to determine its outcome. This approach allows robust, path‑independent labeling of price moves, mitigating look‑ahead bias and class‑imbalance issues common in financial ML.

> **Why Triple Barrier?**
>
> * Combines profit‑taking and risk management in a single framework.
> * Produces balanced, information‑rich labels for supervised learning.
> * Compatible with any asset class and frequency—from tick to daily.
