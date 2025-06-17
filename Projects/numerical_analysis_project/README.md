# P-Spline Volatility Surface Modeling: A Numerical Analysis

## Overview

This project implements and analyzes a robust numerical pipeline to construct and evaluate **Implied Volatility Surfaces (IVS)** from real market option data. [cite_start]The IVS is a critical tool in modern finance for option valuation, risk management, and market signal extraction. [cite_start]However, constructing a reliable IVS is a significant numerical challenge due to noisy, sparse, and irregular market data.

[cite_start]The core emphasis of this report is strongly within the realm of numerical analysis, focusing on the mathematical foundations, algorithmic structure, stability, convergence properties, errors, and computational issues of the selected numerical methods.

## Key Numerical Methods and Techniques Implemented

This project leverages several advanced numerical analysis techniques:

1.  **Newton-Raphson Method (for Implied Volatility Extraction):**
    * [cite_start]Used as a preprocessing step to derive implied volatility data points from observed market option prices by solving the non-linear Black-Scholes pricing equation.
    * [cite_start]The implementation incorporates robust handling of initial guesses, convergence criteria, and near-zero Vega.
    * [cite_start]Exhibits quadratic convergence under ideal conditions.

2.  **1D Natural Cubic Spline Interpolation (for Risk-Free Rates):**
    * [cite_start]Employed to obtain a smooth, continuous risk-free interest rate curve from discrete yield data points, essential for the Black-Scholes formula.
    * [cite_start]Ensures $C^2$ continuity and minimizes "bending energy".

3.  **2D Penalized B-Splines (P-Splines) (for IVS Fitting):**
    * [cite_start]The primary method for modeling the IVS, extending 1D spline concepts to two dimensions.
    * [cite_start]Utilizes a tensor-product B-spline basis and a penalty term (based on second differences of spline coefficients) to enforce smoothness and mitigate overfitting, controlled by a smoothing parameter $\lambda$.
    * [cite_start]Involves constructing sparse design matrices using 1D basis matrices and tensor product construction.

4.  **LSQR Algorithm (for Solving Penalized Least Squares System):**
    * [cite_start]An iterative method specifically designed for solving the large, potentially sparse, linear least-squares problem arising from P-spline fitting.
    * [cite_start]Crucially avoids explicit formation of normal equations (`AᵀA`), enhancing numerical stability.
    * [cite_start]Convergence is controlled by various tolerance parameters and iteration limits.

## Data Preprocessing and Filtering

[cite_start]Robust data filtering is crucial for stable and accurate numerical solutions. Filters applied include:
* [cite_start]Relative Bid-Ask Spread (`(Ask-Bid)/Mid Price > 0.30`).
* [cite_start]Minimum Time-to-Maturity (`T ≥ 0.01 years`).
* [cite_start]Minimum Volume (`≥ 3 contracts`).
* [cite_start]Post-IV calculation filtering to remove non-converged or non-positive IVs.

## Key Findings and Contributions

* [cite_start]**Smoothing Parameter Impact:** The smoothing parameter $\lambda$ significantly influences the trade-off between data fidelity and surface smoothness. [cite_start]While small $\lambda$ values (e.g., $\lambda=0.00$ or $\lambda=0.0001$) yielded the lowest quantitative errors, $\lambda=0$ resulted in unrealistic boundary oscillations. [cite_start]A practical optimal range was identified as $\lambda \in [0.01, 1.0]$ for balancing visual plausibility and low error.
* [cite_start]**Solver Performance:** Newton-Raphson showed high success rates (avg. 97.6%) and low iterations (avg. ~4) for IV determination. [cite_start]LSQR consistently converged for spline fitting, with drastically fewer iterations needed when $\lambda > 0$, highlighting the benefit of Tikhonov regularization.
* [cite_start]**Computational Efficiency:** Penalized runs ($\lambda > 0$) were noticeably faster than unpenalized fits, demonstrating a significant speed-up.

## Code Structure

The project's codebase is organized into several Python scripts:

* `main.py`: Orchestrates the overall experimental pipeline.
* `run_experiments.py`: Manages experimental design, parameter variation, and execution.
* `data_collection/yahoo_finance_data.py`: Handles data retrieval from Yahoo! Finance.
* `src/`: Contains core numerical implementations:
    * `black_scholes.py`: Black-Scholes option pricing formulas.
    * `implied_volatility.py`: Newton-Raphson implementation for IV extraction.
    * `market_data_utils.py`: Functions for risk-free rate interpolation (1D cubic splines) and data preprocessing.
    * `custom_bspline.py`: 2D Penalized B-Spline fitting, including B-spline basis evaluation (Cox-de Boor) and LSQR integration.
    * `data_analysis.py`: Scripts for analyzing and aggregating experiment results.
    * `plot_volatility_surface.py`: Utilities for visualizing the fitted volatility surfaces.
* `experiment_results/`: Stores output data from experiments (e.g., `experiment_summary_lambda_focus_v3.csv`).
* `Final Report.pdf`: The detailed project report.

## How to Run

1.  **Dependencies:** (You would list specific Python libraries here, e.g., `numpy`, `scipy`, `pandas`, `matplotlib`, `yfinance`).
2.  **Data:** (Explain how to obtain or generate the necessary market data, e.g., "Raw option chain data and Treasury yields are assumed to be in `data/raw/`..." or "Run `data_collection/yahoo_finance_data.py` to fetch data.")
3.  **Execution:** (Provide command-line instructions, e.g., "Run `python main.py` to start the experiments, or `python run_experiments.py` for specific runs.")

## Project Files

* **[View the Code Files](./src/)**
* **[View Raw Data Collection Script](./data_collection/yahoo_finance_data.py)**
* **[View Experiment Results Summary](./experiment_results/experiment_summary_lambda_focus_v3.csv)**
* **[View the Full Project Report](./Final_Report.pdf)**

---
