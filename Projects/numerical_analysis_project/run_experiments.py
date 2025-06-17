# run_experiments.py (Enhanced + IV Recalc + LSQR Debug Print + NameError Fix)

import os
import sys
import pandas as pd
import numpy as np
import time
import itertools
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from datetime import date, timedelta
import warnings # Added for handling potential plotting warnings

# --- Project Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir
src_dir = os.path.join(project_root, 'src')
data_collection_dir = os.path.join(project_root, 'data_collection')

# Add project directories to Python path
if src_dir not in sys.path: sys.path.insert(0, src_dir)
if data_collection_dir not in sys.path: sys.path.insert(0, data_collection_dir)
if project_root not in sys.path: sys.path.insert(0, project_root)

# --- Import Project Functions ---
try:
    # Assuming data_collection is available if needed, otherwise comment out
    # from data_collection.yahoo_finance_data import collect_option_data
    from src.market_data_utils import fetch_daily_yield_curve, create_rate_interpolator, get_spot_price
    # Assume analyze_interpolation_errors now returns (df, convergence_dict) and performs IV recalc
    from src.data_analysis import analyze_interpolation_errors, visualize_errors, FILTER_DEFAULTS
    from src.plot_volatility_surface import plot_volatility_surface
    print("Successfully imported project modules.")
except ImportError as e:
    print(f"Error importing project modules: {e}"); traceback.print_exc(); sys.exit(1)

# --- Configuration ---
# Tickers (Adjust as needed)
TICKERS = [
    # Your Originals:
    'AAPL', 'MSFT', 'GOOG', 'SPY', 'V','TSLA', 'NVDA', 'AMD', 'T', 'JPM',
    'MA', 'NFLX', 'COST', 'UNH', 'JNJ', 'CSCO', 'CRM',
    # Major Index ETFs:
    'QQQ', 'IWM', 'DIA',
    # S&P 100 / Large Cap Additions (various sectors):
    'AMZN', 'GOOGL', 'META', 'BRK-B', 'LLY', 'AVGO', 'XOM', 'WMT', 'HD',
    'PG', 'ABBV', 'CVX', 'MRK', 'KO', 'PEP', 'ADBE', 'BAC', 'MCD', 'CSCO', # CSCO repeated - OK
    'ACN', 'WFC', 'LIN', 'DIS', 'INTC', 'VZ', 'ABT', 'IBM', 'ORCL', 'NEE',
    'PM', 'CMCSA', 'NKE', 'HON', 'UPS', 'TXN', 'PFE', 'BMY', 'AMGN', 'RTX', # Raytheon (formerly UTX)
    'SBUX', 'CAT', 'GS', 'BLK', 'LOW', 'TMO', 'AXP', 'PYPL', # PayPal added
    'BA', 'ELV', 'LMT', 'COP', 'DE', 'GE', 'MMM', 'ADP', 'CVS', 'MDLZ',
    'SO', 'GILD', 'TJX', 'MO', 'CL', 'DUK', 'SCHW', 'USB', 'MS', 'CI',
    'ANTM', 'CME', 'ETN', 'FISV', 'NOW', # ServiceNow added
    # Other Large/Relevant Tickers:
    'PYPL', 'INTU', 'ISRG', 'BKNG', 'CAT', # Some repeated from S&P100 section - OK
    'UBER', 'ZM', 'SNOW', 'PLTR', 'SQ', 'SHOP', 'ETSY', 'PINS', 'RBLX',
    'FDX', 'GM', 'F', 'DAL', 'UAL', 'AAL', 'BA', # Boeing repeated - OK
    'PNC', 'GS', 'COF', 'USB', # Some banks repeated - OK
    'PFE', 'MRK', 'BMY', 'LLY', 'JNJ', 'ABBV', 'GILD', 'AMGN', # Some Pharma repeated - OK
    'OXY', 'SLB', 'HAL', 'EOG', 'DVN', # Energy
    'FCX', 'NEM', # Materials
    'AMT', 'PLD', 'EQIX', # REITs
    'DIS', 'CMCSA', 'NFLX', 'T', 'VZ' # Communication Services repeated - OK
]

# Experiment Parameters
# Focus: Varying the smoothness penalty (lambda) for P-splines
LAMBDA_VALUES = [0.0,0.0001,0.001, 0.01, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0] # (~10 values)

# Fixed parameters for this experiment run (Kept constant to isolate lambda effect)
FIXED_KNOT_STRATEGY = 'uniform'
FIXED_KNOT_DENSITY_KEY = 'medium' # (Moderate number of knots)
KNOT_DENSITIES = {'low': [3, 2], 'medium': [8, 5], 'high': [15, 10]}
fixed_n_knots_x, fixed_n_knots_y = KNOT_DENSITIES[FIXED_KNOT_DENSITY_KEY]

FIXED_DEGREE_KEY = 'cubic' # (Using cubic splines)
DEGREES = {'cubic': [3, 3], 'quadratic': [2, 2]}
fixed_degree_x, fixed_degree_y = DEGREES[FIXED_DEGREE_KEY]

FIXED_PENALTY_ORDER = 2 # (Standard second-order penalty)

# Fixed data filtering parameters
FIXED_FILTER_KEY = 'medium' # (Moderate filtering strictness)
BASE_FILTER = FILTER_DEFAULTS.copy(); BASE_FILTER['min_volume'] = 3
FILTERING_LEVELS = {
    'strict': {'max_bid_ask_spread_rel': 0.20, 'min_ttm_years': 0.02, 'min_volume': 3},
    'medium': {'max_bid_ask_spread_rel': 0.30, 'min_ttm_years': 0.01, 'min_volume': 3},
    'loose': {'max_bid_ask_spread_rel': 0.50, 'min_ttm_years': 0.005, 'min_volume': 1}
}
FIXED_FILTER_PARAMS = BASE_FILTER.copy()
FIXED_FILTER_PARAMS.update(FILTERING_LEVELS[FIXED_FILTER_KEY])
# Explicitly set fixed filter values (ensures update worked or overrides)
FIXED_FILTER_PARAMS['max_bid_ask_spread_rel'] = FILTERING_LEVELS[FIXED_FILTER_KEY]['max_bid_ask_spread_rel']
FIXED_FILTER_PARAMS['min_ttm_years'] = FILTERING_LEVELS[FIXED_FILTER_KEY]['min_ttm_years']
FIXED_FILTER_PARAMS['min_volume'] = FILTERING_LEVELS[FIXED_FILTER_KEY]['min_volume']


print(f"--- Experiment Configuration ---")
print(f"Tickers: {TICKERS}")
print(f"Lambda Values (Penalty Strength): {LAMBDA_VALUES}")
print(f"Fixed Knot Strategy: {FIXED_KNOT_STRATEGY}")
print(f"Fixed Knot Density: {FIXED_KNOT_DENSITY_KEY} ({fixed_n_knots_x}, {fixed_n_knots_y}) knots")
print(f"Fixed Spline Degree: {FIXED_DEGREE_KEY} ({fixed_degree_x}, {fixed_degree_y})")
print(f"Fixed Penalty Order: {FIXED_PENALTY_ORDER}")
print(f"Fixed Data Filtering ('{FIXED_FILTER_KEY}'): {FIXED_FILTER_PARAMS}")

# Other Fixed Config
MAX_MATURITY = 1.5 # Max option maturity in years to consider
MIN_SEPARATION_X = 1.0; MIN_SEPARATION_Y = 0.01 # Min knot separation
# LSQR solver parameters (using defaults from data_analysis.py if not overridden)
SOLVER_PARAMS = {'lsqr_iter_lim': 10000, 'lsqr_tol': 1e-8}

# Output Directories
RESULTS_DIR = os.path.join(project_root, "experiment_results")
RESULTS_DATA_FILE = os.path.join(RESULTS_DIR, "experiment_summary_lambda_focus_v3.csv") # Ensure this matches your desired output
AGGREGATE_PLOTS_DIR = os.path.join(RESULTS_DIR, "aggregate_plots_lambda_focus_v3") # Ensure this matches your desired output
RAW_DATA_DIR = os.path.join(project_root, 'data', 'raw')

# --- Helper Functions ---
def get_analysis_date():
    """Gets the most recent weekday, excluding today."""
    analysis_date = date.today() - timedelta(days=1)
    # Ensure it's a weekday
    if analysis_date.weekday() >= 5: # 5=Saturday, 6=Sunday
        analysis_date -= timedelta(days=analysis_date.weekday() - 4) # Go back to Friday
    return analysis_date

def create_output_dirs():
    """Creates directories for saving results and plots."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(AGGREGATE_PLOTS_DIR, exist_ok=True)
    os.makedirs(RAW_DATA_DIR, exist_ok=True) # Ensure raw data dir exists
    print(f"\nResults CSV: {RESULTS_DATA_FILE}")
    print(f"Aggregate plots: {AGGREGATE_PLOTS_DIR}")
    print(f"Raw data directory: {RAW_DATA_DIR}")

# --- Interactive Plotting Helpers (Optional - Keep commented if not used) ---
# def generate_surface_plot_interactive(run_info): ...
# def generate_error_plot_interactive(run_info): ...


# --- Main Execution ---
if __name__ == "__main__":
    main_start_time = time.time()
    create_output_dirs()
    analysis_date = get_analysis_date()
    all_results = [] # List to store result dictionaries for each run
    failed_data_collections = []

    # 1. Check for Raw Data
    print(f"\n--- Step 1: Checking/Using Existing Option Data for date {analysis_date} ---")
    # (Data collection block intentionally omitted, assumes data exists)

    active_tickers = []
    for ticker in TICKERS:
        pattern = f"{ticker}_call_options_*"
        ticker_files = glob.glob(os.path.join(RAW_DATA_DIR, pattern))
        if ticker_files:
            active_tickers.append(ticker)
        else:
            print(f"Warning: No raw data files found for {ticker} in {RAW_DATA_DIR}. Skipping.")
            failed_data_collections.append(ticker)

    if not active_tickers:
        print("ERROR: No data available for any tickers. Exiting."); sys.exit(1)
    print(f"Tickers to be processed: {active_tickers}")

    # 2. Fetch Yield Curve ONCE
    print(f"\n--- Step 2: Fetching Yield Curve for {analysis_date} ---")
    rate_interpolator = None
    try:
         yield_curve = fetch_daily_yield_curve(analysis_date)
         if yield_curve is None: raise ValueError("Yield curve fetch returned None.")
         rate_interpolator = create_rate_interpolator(yield_curve)
         if rate_interpolator is None: raise ValueError("Rate interpolator creation failed.")
         print("Successfully obtained rate interpolator.")
    except Exception as e:
        print(f"ERROR: Failed to get yield curve/interpolator: {e}. Exiting."); sys.exit(1)

    # 3. Run Experiments per Ticker
    print("\n--- Step 3: Running Experiments ---")
    # Create combinations of ticker and lambda value
    param_combinations = list(itertools.product(
        active_tickers,
        LAMBDA_VALUES
    ))
    total_runs = len(param_combinations)
    current_run = 0

    # Loop through each combination
    for ticker, lambda_val in param_combinations:
        current_run += 1
        ticker_lambda_start_time = time.time()
        print(f"\n=== Processing {current_run}/{total_runs}: Ticker={ticker}, Lambda={lambda_val:.2e} ===")
        spot_price = None; ticker_files = []

        # Setup ticker context (spot price, file paths)
        try:
             spot_price = get_spot_price(ticker, analysis_date)
             if spot_price is None: raise ValueError("Spot price fetch returned None.")
             pattern = f"{ticker}_call_options_*"
             ticker_files = glob.glob(os.path.join(RAW_DATA_DIR, pattern))
             if not ticker_files: raise FileNotFoundError(f"No data files found for {ticker}")
             print(f"  Spot: {spot_price:.2f}, Files: {len(ticker_files)}")
        except Exception as e:
            print(f"  ERROR setting up ticker {ticker}: {e}. Skipping run.")
            run_id = f"{ticker}_L{lambda_val:.2e}_SETUP_ERROR"
            results={'run_id':run_id,'ticker':ticker,'status':f'Failed-Setup', 'lambda_val': lambda_val}
            all_results.append(results)
            continue # Skip to the next combination

        # --- Prepare parameters for this specific run ---
        strategy = FIXED_KNOT_STRATEGY
        penalty_order = FIXED_PENALTY_ORDER
        run_id = f"{ticker}_{strategy}_L{lambda_val:.2e}_Pen{penalty_order}_{FIXED_KNOT_DENSITY_KEY}_{FIXED_DEGREE_KEY}"
        print(f"  Running Analysis for: {run_id}")

        # Define knot parameters for spline
        knot_params_x = {'num_internal_knots': fixed_n_knots_x, 'strategy': strategy, 'min_separation': MIN_SEPARATION_X}
        knot_params_y = {'num_internal_knots': fixed_n_knots_y, 'strategy': strategy, 'min_separation': MIN_SEPARATION_Y}

        # Assemble arguments for the main analysis function
        analysis_args = {
            'file_paths': ticker_files,
            'spot_price': spot_price,
            'rate_interpolator': rate_interpolator,
            'max_maturity_years': MAX_MATURITY,
            'degree_x': fixed_degree_x,
            'degree_y': fixed_degree_y,
            'knot_params_x': knot_params_x,
            'knot_params_y': knot_params_y,
            'filter_params': FIXED_FILTER_PARAMS.copy() # Pass a copy
            # IV calculation params could be passed here if needed, e.g.:
            # 'iv_tolerance': 1e-7, 'iv_max_iterations': 100
        }
        # Assemble solver/lambda parameters (passed as keyword arguments)
        solver_lambda_kwargs = {
            "lambda_x": lambda_val, # Varying lambda
            "lambda_y": lambda_val, # Using same lambda for both axes
            "penalty_order": penalty_order, # Fixed
            **SOLVER_PARAMS # Include LSQR params
        }

        # Initialize dictionary to store results for this run
        results = {
            'run_id': run_id, 'ticker': ticker, 'analysis_date': analysis_date.strftime("%Y-%m-%d"), # Basic info
            # Fixed Parameters for this run
            'knot_strategy': strategy, 'lambda_val': lambda_val, 'penalty_order': penalty_order,
            'knot_density_key': FIXED_KNOT_DENSITY_KEY, 'degree_key': FIXED_DEGREE_KEY, 'filter_key': FIXED_FILTER_KEY,
            'n_knots_x': fixed_n_knots_x, 'n_knots_y': fixed_n_knots_y, 'degree_x': fixed_degree_x, 'degree_y': fixed_degree_y,
            'filter_min_vol': FIXED_FILTER_PARAMS['min_volume'],'filter_spread_rel': FIXED_FILTER_PARAMS['max_bid_ask_spread_rel'],'filter_min_ttm': FIXED_FILTER_PARAMS['min_ttm_years'],
            # Performance Metrics (initialize as NaN)
            'iv_mae': np.nan, 'iv_rmse': np.nan, 'iv_median_ae': np.nan, 'iv_std_err': np.nan,'iv_min_abs_err': np.nan, 'iv_max_abs_err': np.nan, # IV errors
            'price_mae': np.nan, 'price_rmse': np.nan, 'price_median_ae': np.nan, 'price_std_err': np.nan,'price_min_abs_err': np.nan, 'price_max_abs_err': np.nan, # Price errors
            # Convergence Metrics (initialize as NaN)
            'spline_solver_iterations': np.nan,'spline_solver_status': np.nan,'avg_iv_iterations': np.nan,'iv_success_rate': np.nan,
            # Other Metrics
            'computation_time': np.nan,'final_data_points': 0,'status': 'Failed - Unknown',
            # Store detailed results (DataFrame) - remove before final save
            'detailed_error_df': None
        }

        analysis_start_time = time.time()
        error_dataframe = None
        convergence_metrics = {} # Initialize convergence dict

        # --- Execute Analysis ---
        try:
            # Call the main analysis function (now includes IV recalc)
            analysis_result = analyze_interpolation_errors(**analysis_args, **solver_lambda_kwargs)

            # Unpack the results (DataFrame and convergence dictionary)
            if isinstance(analysis_result, tuple) and len(analysis_result) == 2:
                error_dataframe, convergence_metrics = analysis_result
                if not isinstance(convergence_metrics, dict): convergence_metrics = {}
            elif isinstance(analysis_result, pd.DataFrame):
                 error_dataframe = analysis_result; convergence_metrics = {}
                 print(f"  Warning: analyze_interpolation_errors only returned DataFrame for {run_id}.")
            else: # Handle None return
                error_dataframe = None; convergence_metrics = {}

            # <<< --- ADDED DEBUG PRINT --- >>>
            print(f"  DEBUG: Received convergence_metrics: {convergence_metrics}")

            analysis_time = time.time() - analysis_start_time
            results['computation_time'] = analysis_time

            # Process results based on success/failure
            if error_dataframe is None or error_dataframe.empty:
                results['status'] = 'Failed - Analysis Empty'
                print(f"  Analysis failed or returned empty dataframe. Time: {analysis_time:.2f}s")
                 # Store any convergence metrics even if analysis failed later
                results['spline_solver_iterations'] = convergence_metrics.get('spline_solver_iterations', np.nan)
                results['spline_solver_status'] = convergence_metrics.get('spline_solver_status', np.nan)
                results['avg_iv_iterations'] = convergence_metrics.get('avg_iv_iterations', np.nan)
                results['iv_success_rate'] = convergence_metrics.get('iv_success_rate', np.nan)
            else:
                # --- Analysis Succeeded ---
                results['status'] = 'Success'
                results['final_data_points'] = len(error_dataframe)
                results['detailed_error_df'] = error_dataframe # Store the detailed dataframe for later analysis

                # Calculate error summary statistics
                if 'iv_abs_error' in error_dataframe.columns and not error_dataframe['iv_abs_error'].isnull().all():
                    results['iv_mae'] = error_dataframe['iv_abs_error'].mean()
                    results['iv_median_ae'] = error_dataframe['iv_abs_error'].median()
                    results['iv_min_abs_err'] = error_dataframe['iv_abs_error'].min()
                    results['iv_max_abs_err'] = error_dataframe['iv_abs_error'].max()
                if 'iv_error' in error_dataframe.columns and not error_dataframe['iv_error'].isnull().all():
                    results['iv_rmse'] = np.sqrt(np.mean(error_dataframe['iv_error']**2))
                    results['iv_std_err'] = error_dataframe['iv_error'].std()
                if 'price_abs_error' in error_dataframe.columns and not error_dataframe['price_abs_error'].isnull().all():
                     results['price_mae'] = error_dataframe['price_abs_error'].mean()
                     results['price_median_ae'] = error_dataframe['price_abs_error'].median()
                     results['price_min_abs_err'] = error_dataframe['price_abs_error'].min()
                     results['price_max_abs_err'] = error_dataframe['price_abs_error'].max()
                if 'price_error' in error_dataframe.columns and not error_dataframe['price_error'].isnull().all():
                     results['price_rmse'] = np.sqrt(np.mean(error_dataframe['price_error']**2))
                     results['price_std_err'] = error_dataframe['price_error'].std()

                # Populate convergence metrics from the received dictionary
                results['spline_solver_iterations'] = convergence_metrics.get('spline_solver_iterations', np.nan)
                results['spline_solver_status'] = convergence_metrics.get('spline_solver_status', np.nan)
                results['avg_iv_iterations'] = convergence_metrics.get('avg_iv_iterations', np.nan)
                results['iv_success_rate'] = convergence_metrics.get('iv_success_rate', np.nan)

                print(f"  Success: IV MAE={results.get('iv_mae', 'N/A'):.6f}, Time: {analysis_time:.2f}s")

        except Exception as e: # Catch exceptions during the analysis call
            results['computation_time'] = time.time() - analysis_start_time
            results['status'] = f'Failed - {type(e).__name__}'
            print(f"  ERROR during analysis execution for {run_id}: {e}"); traceback.print_exc(limit=1)
            # Store any partial convergence metrics even on exception
            results['spline_solver_iterations'] = convergence_metrics.get('spline_solver_iterations', np.nan)
            results['spline_solver_status'] = convergence_metrics.get('spline_solver_status', np.nan)
            results['avg_iv_iterations'] = convergence_metrics.get('avg_iv_iterations', np.nan)
            results['iv_success_rate'] = convergence_metrics.get('iv_success_rate', np.nan)

        # Append results for this run to the main list
        all_results.append(results)
        # --- End Individual Run ---

    # --- End All Runs Loop ---

    # 4. Save Final Results Summary
    print("\n--- Step 4: Saving Final Detailed Results Summary ---")
    if not all_results:
        print("No results generated. Exiting."); sys.exit(1)
    results_df = pd.DataFrame(all_results)
    # Exclude columns not suitable for CSV (like objects, detailed dataframes)
    cols_to_exclude_from_csv = ['detailed_error_df'] # Removed others as they weren't in the dict definition anyway
    cols_to_save = [col for col in results_df.columns if col not in cols_to_exclude_from_csv]
    results_to_save_df = results_df[cols_to_save].copy()
    # Optional rounding for numeric columns
    numeric_cols_for_rounding = results_to_save_df.select_dtypes(include=np.number).columns
    results_to_save_df[numeric_cols_for_rounding] = results_to_save_df[numeric_cols_for_rounding].round(6)
    # Save to CSV
    results_to_save_df.to_csv(RESULTS_DATA_FILE, index=False)
    print(f"Saved experiment summary to: {RESULTS_DATA_FILE}")


    # --- Step 5 & 6: Aggregated Analysis and Visualization ---
    print("\n--- Step 5 & 6: Aggregated Analysis and Visualization ---")
    # Filter results for successful runs that have detailed data
    success_results = [r for r in all_results if r['status'] == 'Success' and r.get('detailed_error_df') is not None]

    # Initialize list for interactive plotting BEFORE the check
    top_5_run_infos_interactive = []

    if not success_results:
        print("No successful runs with detailed data found for aggregate analysis.")
    else:
        print(f"Analyzing {len(success_results)} successful runs with detailed data...")
        plt.style.use('seaborn-v0_8-darkgrid')
        agg_plot_dir = AGGREGATE_PLOTS_DIR
        all_error_dfs = [] # List to store all detailed error dataframes

        # --- Aggregation Phase: Combine detailed data from all runs ---
        print("  Aggregating detailed error data...")
        for run_result in success_results:
            df = run_result['detailed_error_df'].copy()
            # Add key parameters from the run to the detailed df for easier plotting/grouping
            df['run_id'] = run_result['run_id']
            df['ticker'] = run_result['ticker']
            df['lambda_val'] = run_result['lambda_val']
            all_error_dfs.append(df)

        if not all_error_dfs:
             print("  No detailed error dataframes could be extracted. Skipping aggregate plots.")
        else:
            # Combine into one large dataframe
            combined_errors_df = pd.concat(all_error_dfs, ignore_index=True)
            print(f"  Combined error dataframe created with {len(combined_errors_df)} total rows.")

            # --- Calculation & Plotting Phase ---

            # Plot 1: Spline Solver (LSQR) Iterations
            print("  Generating Convergence Plots...")
            solver_iters = results_df.loc[results_df['status'] == 'Success', 'spline_solver_iterations'].dropna() # Use summary df
            if not solver_iters.empty:
                fig_lsqr, ax_lsqr = plt.subplots(figsize=(8, 5))
                sns.histplot(solver_iters, kde=True, ax=ax_lsqr)
                ax_lsqr.set_title('Distribution of Spline Solver (LSQR) Iterations per Run')
                ax_lsqr.set_xlabel('LSQR Iterations'); ax_lsqr.set_ylabel('Frequency')
                plt.tight_layout(); fig_lsqr.savefig(os.path.join(agg_plot_dir, "dist_spline_solver_iterations.png")); plt.close(fig_lsqr)
                print("    Saved plot: dist_spline_solver_iterations.png")
            else:
                print("    Skipping LSQR iterations plot (no data).")

            # Plot 2: Implied Volatility (Newton-Raphson) Iterations
            # Needs 'iterations' column from IV recalc, now in combined_errors_df
            if 'iterations' in combined_errors_df.columns:
                 iv_iters = combined_errors_df['iterations'].dropna()
                 iv_iters = iv_iters[iv_iters > 0] # Exclude non-positive
                 if not iv_iters.empty:
                    fig_iv, ax_iv = plt.subplots(figsize=(8, 5))
                    plot_iv_iters = iv_iters[iv_iters < iv_iters.quantile(0.99)] if not iv_iters.empty else iv_iters # Cap outliers
                    sns.histplot(plot_iv_iters, kde=True, ax=ax_iv, bins=min(len(plot_iv_iters.unique()), 50))
                    ax_iv.set_title('Distribution of Implied Volatility (Newton-Raphson) Iterations')
                    ax_iv.set_xlabel('Newton-Raphson Iterations (capped at 99th percentile)'); ax_iv.set_ylabel('Frequency')
                    plt.tight_layout(); fig_iv.savefig(os.path.join(agg_plot_dir, "dist_implied_vol_iterations.png")); plt.close(fig_iv)
                    print("    Saved plot: dist_implied_vol_iterations.png")
                 else:
                    print("    Skipping IV iterations plot (no valid data > 0 in detailed df).")
            else:
                print("    Skipping IV iterations plot ('iterations' column missing from detailed df).")

            # Plot 3 & 4: Error Scatter Plots
            print("  Generating Error Pattern Plots...")
            sample_df = combined_errors_df.sample(n=min(50000, len(combined_errors_df)), random_state=1) if len(combined_errors_df) > 0 else combined_errors_df
            # IV Error vs TTM
            if 'iv_error' in sample_df.columns and 'time_to_maturity' in sample_df.columns:
                 fig_iv_ttm, ax_iv_ttm = plt.subplots(figsize=(10, 6))
                 y_col_plot = 'iv_abs_error' if 'iv_abs_error' in sample_df.columns else 'iv_error'
                 sns.scatterplot(data=sample_df, x='time_to_maturity', y=y_col_plot, alpha=0.3, s=10, ax=ax_iv_ttm, hue='lambda_val', palette='viridis')
                 ax_iv_ttm.set_title(f'{y_col_plot.replace("_"," ").title()} vs Time to Maturity (Sampled)'); ax_iv_ttm.set_xlabel('Time to Maturity (Years)'); ax_iv_ttm.set_ylabel(y_col_plot.replace("_"," ").title())
                 try: # Handle legend
                     handles, labels = ax_iv_ttm.get_legend_handles_labels()
                     if len(handles) > 1: ax_iv_ttm.legend(title='Lambda', bbox_to_anchor=(1.05, 1), loc='upper left')
                     else: ax_iv_ttm.get_legend().remove()
                 except Exception: pass
                 plt.tight_layout(rect=[0, 0, 0.85, 1]); fig_iv_ttm.savefig(os.path.join(agg_plot_dir, "scatter_iv_error_vs_ttm.png")); plt.close(fig_iv_ttm)
                 print("    Saved plot: scatter_iv_error_vs_ttm.png")
            else:
                 print("    Skipping IV Error vs TTM plot (missing columns).")
            # Price Error vs TTM
            if 'price_error' in sample_df.columns and 'time_to_maturity' in sample_df.columns:
                 fig_pr_ttm, ax_pr_ttm = plt.subplots(figsize=(10, 6))
                 y_col_plot = 'price_abs_error' if 'price_abs_error' in sample_df.columns else 'price_error'
                 sns.scatterplot(data=sample_df, x='time_to_maturity', y=y_col_plot, alpha=0.3, s=10, ax=ax_pr_ttm, hue='lambda_val', palette='viridis')
                 ax_pr_ttm.set_title(f'{y_col_plot.replace("_"," ").title()} vs Time to Maturity (Sampled)'); ax_pr_ttm.set_xlabel('Time to Maturity (Years)'); ax_pr_ttm.set_ylabel(y_col_plot.replace("_"," ").title())
                 try: # Handle legend
                      handles, labels = ax_pr_ttm.get_legend_handles_labels()
                      if len(handles) > 1: ax_pr_ttm.legend(title='Lambda', bbox_to_anchor=(1.05, 1), loc='upper left')
                      else: ax_pr_ttm.get_legend().remove()
                 except Exception: pass
                 plt.tight_layout(rect=[0, 0, 0.85, 1]); fig_pr_ttm.savefig(os.path.join(agg_plot_dir, "scatter_price_error_vs_ttm.png")); plt.close(fig_pr_ttm)
                 print("    Saved plot: scatter_price_error_vs_ttm.png")
            else:
                 print("    Skipping Price Error vs TTM plot (missing columns).")

            # Plot 5: Error vs Parameters (Box Plot)
            print("  Generating Error vs Parameter Plots...")
            success_df_agg = results_df[results_df['status'] == 'Success'].copy() # Use summary df
            if 'iv_mae' in success_df_agg.columns and 'lambda_val' in success_df_agg.columns:
                success_df_agg['lambda_val_cat'] = success_df_agg['lambda_val'].astype(str)
                lambda_order = sorted(success_df_agg['lambda_val'].unique())
                lambda_order_str = [str(l) for l in lambda_order] # Ensure correct order
                fig_box, ax_box = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=success_df_agg, x='lambda_val_cat', y='iv_mae', ax=ax_box, order=lambda_order_str)
                ax_box.set_title('Distribution of IV MAE vs Smoothness Lambda'); ax_box.set_xlabel('Lambda Value'); ax_box.set_ylabel('IV MAE')
                plt.xticks(rotation=45, ha='right'); plt.tight_layout(); fig_box.savefig(os.path.join(agg_plot_dir, "boxplot_iv_mae_vs_lambda.png")); plt.close(fig_box)
                print("    Saved plot: boxplot_iv_mae_vs_lambda.png")
            else:
                print("    Skipping IV MAE vs Lambda boxplot (missing columns).")

            # --- Identify Top Runs (Based on summary results_df) ---
            print("\n  --- Top 5 Runs Analysis (based on summary results) ---")
            try:
                success_df_summary = results_df[results_df['status'] == 'Success'].copy()
                if success_df_summary.empty:
                    print("    No successful runs in summary to find top 5.")
                else:
                    numeric_cols = ['iv_mae', 'price_rmse', 'computation_time']
                    for col in numeric_cols:
                        success_df_summary[col] = pd.to_numeric(success_df_summary[col], errors='coerce')
                    success_df_summary.dropna(subset=['iv_mae'], inplace=True)

                    if not success_df_summary.empty:
                        top_5_runs = success_df_summary.nsmallest(5, 'iv_mae')
                        print("    Top 5 Runs (Lowest IV MAE Overall from summary):")
                        if not top_5_runs.empty:
                            top_run_ids = top_5_runs['run_id'].tolist()
                            rank = 0
                            for run_id_to_plot in top_run_ids:
                                # Find the original full run info from all_results
                                full_run_info = next((r for r in all_results if r.get('run_id') == run_id_to_plot), None)
                                if full_run_info:
                                    rank += 1
                                    # Add necessary components for potential interactive plotting
                                    full_run_info['rate_interpolator'] = rate_interpolator
                                    top_5_run_infos_interactive.append(full_run_info)
                                    print(f"      Rank {rank}: {run_id_to_plot} (IV MAE: {full_run_info.get('iv_mae', np.nan):.6f})")
                                else:
                                    print(f"      Warning: Could not find full run info for {run_id_to_plot}")
                        else:
                            print("    No runs eligible for top 5.")
                    else:
                        print("    No successful runs with valid IV MAE in summary.")
            except Exception as e:
                print(f"    ERROR during top 5 runs identification: {e}"); traceback.print_exc(limit=1)
            print("  ----------------------------------------------------")
        # --- End of the 'else' block for if all_error_dfs ---
    # --- End of the 'else' block for if not success_results ---


    # 7. Interactive Plotting Menu (Optional)
    print("\n--- Step 7: Interactive Plotting for Top 5 Runs (Optional) ---")
    if not top_5_run_infos_interactive:
        print("No successful runs were identified for interactive plotting.")
    else:
        print(f"{len(top_5_run_infos_interactive)} top runs identified. Interactive plotting menu available (uncomment code to enable).")
    # (Interactive menu code remains commented out)

    # --- Step 8: Print Key Experiment Statistics ---
    print("\n--- Step 8: Key Experiment Statistics ---")
    successful_runs = results_df[results_df['status'] == 'Success']

    if successful_runs.empty:
        print("No successful runs to generate statistics.")
    else:
        print(f"\nNumber of Successful Runs: {len(successful_runs)}")
        print("\n=== Overall Error Statistics ===")
        print(f"Average IV MAE: {successful_runs['iv_mae'].mean():.4f}")
        print(f"Median IV MAE: {successful_runs['iv_mae'].median():.4f}")
        print(f"Minimum IV MAE: {successful_runs['iv_mae'].min():.4f}")
        print(f"Maximum IV MAE: {successful_runs['iv_mae'].max():.4f}")
        print(f"\nAverage Price MAE: ${successful_runs['price_mae'].mean():.4f}")
        print(f"Median Price MAE: ${successful_runs['price_mae'].median():.4f}")

        # Lambda Value Analysis (sorted by lambda value)
        lambda_stats = successful_runs.groupby('lambda_val').agg(
            avg_iv_mae=('iv_mae', 'mean'),
            median_iv_mae=('iv_mae', 'median'),
            avg_price_mae=('price_mae', 'mean'),
            num_runs=('run_id', 'count')
        ).sort_index().reset_index()  # Sort by lambda value

        print("\n=== Performance by Lambda Value (sorted ascending) ===")
        print(lambda_stats.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

        # Generate Lambda vs Accuracy plot
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        # Plot IV MAE
        lambda_stats.plot(x='lambda_val', y='avg_iv_mae',
                         marker='o', ax=ax, color='tab:blue',
                         label='Average IV MAE')

        # Plot Price MAE on secondary axis
        ax2 = ax.twinx()
        lambda_stats.plot(x='lambda_val', y='avg_price_mae',
                         marker='s', ax=ax2, color='tab:red',
                         label='Average Price MAE')

        ax.set_title('Smoothing Parameter (λ) vs Model Accuracy')
        ax.set_xlabel('Lambda Value (log scale)')
        ax.set_ylabel('IV MAE', color='tab:blue')
        ax2.set_ylabel('Price MAE', color='tab:red')
        ax.set_xscale('log')  # Use log scale for lambda values

        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper center')

        plt.tight_layout()
        plot_path = os.path.join(AGGREGATE_PLOTS_DIR, "lambda_vs_accuracy.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"\nSaved lambda vs accuracy plot to: {plot_path}")

        # Top/Bottom 5 Runs
        top_5_iv = successful_runs.nsmallest(5, 'iv_mae')[['run_id', 'ticker', 'lambda_val', 'iv_mae', 'price_mae']]
        bottom_5_iv = successful_runs.nlargest(5, 'iv_mae')[['run_id', 'ticker', 'lambda_val', 'iv_mae', 'price_mae']]

        print("\n=== Top 5 Runs by IV Accuracy ===")
        print(top_5_iv.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
        print("\n=== Bottom 5 Runs by IV Accuracy ===")
        print(bottom_5_iv.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

        # Ticker Analysis
        ticker_stats = successful_runs.groupby('ticker').agg(
            avg_iv_mae=('iv_mae', 'mean'),
            best_iv_mae=('iv_mae', 'min'),
            avg_lambda=('lambda_val', 'mean'), # Avg lambda used for best results
            num_runs=('run_id', 'count')
        ).sort_values('avg_iv_mae')

        print("\n=== Ticker Performance Summary ===")
        print(ticker_stats.to_string(float_format=lambda x: f"{x:,.4f}"))

        # Best Lambda per Ticker
        # Find the row index corresponding to the minimum 'iv_mae' for each 'ticker'
        best_idx = successful_runs.loc[successful_runs.groupby('ticker')['iv_mae'].idxmin()]
        print("\n=== Best Lambda Configuration per Ticker ===")
        print(best_idx[['ticker', 'lambda_val', 'iv_mae']].to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    # --- End ---
    total_time = time.time() - main_start_time
    print(f"\n--- Experiment Run Complete ---")
    print(f"Total execution time: {total_time / 60:.2f} minutes")
    print(f"Aggregate plots saved to: {AGGREGATE_PLOTS_DIR}")
    print(f"Summary results saved to: {RESULTS_DATA_FILE}")