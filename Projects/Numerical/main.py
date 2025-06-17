# main.py (Updated for Simplified Plotting/Analysis Input)

import os
import sys
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import time # Import time module for market context timing

# --- Add src and data_collection directories to path ---
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
data_collection_path = os.path.join(project_root, 'data_collection')
if src_path not in sys.path: sys.path.insert(0, src_path)
if data_collection_path not in sys.path: sys.path.insert(0, data_collection_path)
# --- End Path Setup ---


# --- Import Project Modules ---
try:
    import yfinance as yf
    from market_data_utils import fetch_daily_yield_curve, create_rate_interpolator, get_spot_price
    # Assuming yahoo_finance_data.py is now in data_collection path
    from data_collection.yahoo_finance_data import collect_option_data
    from black_scholes import black_scholes_call
    # Assumes implied_volatility returns dict {volatility, iterations, status, ...}
    from implied_volatility import implied_volatility
    # Assumes create_spline_representation returns (spline_obj, convergence_dict)
    # Assumes analyze_interpolation_errors returns (df, convergence_dict)
    from volatility_surface_interpolation import prepare_volatility_data, create_spline_representation, get_interpolated_volatility
    from plot_volatility_surface import plot_volatility_surface
    from data_analysis import analyze_interpolation_errors, visualize_errors, FILTER_DEFAULTS, SPLINE_DEFAULTS
    print("Successfully imported all project modules.")
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure all required files are in the 'src/' or 'data_collection/' directory and dependencies are installed.")
    traceback.print_exc()
    sys.exit(1)
# --- End Imports ---

# --- Default Configuration ---
DEFAULT_TICKER = "AAPL"
DEFAULT_DATA_DIR = os.path.join(project_root, 'data', 'raw')

# --- Use defaults from data_analysis directly for consistency ---
# These now reflect the FIXED parameters from run_experiments.py if data_analysis.py is aligned
DEFAULT_FILTER_PARAMS = FILTER_DEFAULTS.copy()
DEFAULT_SPLINE_PARAMS = SPLINE_DEFAULTS.copy() # Contains degree, knots, lambdas, penalty, solver params

# Override defaults here if main.py should use different base settings than run_experiments
# Example: Use the 'medium' knot density from run_experiments as default here
DEFAULT_SPLINE_PARAMS['knot_params_x'] = {'num_internal_knots': 8, 'strategy': 'uniform', 'min_separation': 1.0}
DEFAULT_SPLINE_PARAMS['knot_params_y'] = {'num_internal_knots': 5, 'strategy': 'uniform', 'min_separation': 0.01}
# Set default lambdas if desired for interactive use
# DEFAULT_SPLINE_PARAMS['lambda_x'] = 0.1
# DEFAULT_SPLINE_PARAMS['lambda_y'] = 0.1


DEFAULT_MAX_MATURITY = 1.5 # Match run_experiments
# --- End Configuration ---

# --- Global cache ---
MARKET_CONTEXT = { "ticker": None, "date": None, "spot_price": None, "rate_interpolator": None, "yield_curve_data": None }

# --- Helper Functions for User Input ---
# (get_string_input, get_float_input, get_int_input, get_yes_no remain unchanged)
def get_string_input(prompt, default):
    value = input(f"{prompt} [{default}]: ").strip()
    return value if value else default

def get_float_input(prompt, default):
    default_display = f"{default:.4f}" if isinstance(default, float) and default is not None else str(default)
    while True:
        value_str = input(f"{prompt} [{default_display}]: ").strip()
        if not value_str:
            if default is None: print("Error: A value is required."); continue
            return default
        try: return float(value_str)
        except ValueError: print("Invalid input. Please enter a number.")

def get_int_input(prompt, default):
    while True:
        value_str = input(f"{prompt} [{default}]: ").strip()
        if not value_str: return default
        try: return int(value_str)
        except ValueError: print("Invalid input. Please enter an integer.")

def get_yes_no(prompt, default='n'):
    while True:
        value = input(f"{prompt} (y/n) [{default}]: ").lower().strip()
        if not value: value = default
        if value in ['y', 'yes']: return True
        elif value in ['n', 'no']: return False
        else: print("Invalid input. Please enter 'y' or 'n'.")

# --- Market Context and Rate Functions ---
def update_market_context(ticker, analysis_date):
    """Fetches and caches spot price and rate interpolator."""
    global MARKET_CONTEXT
    if isinstance(analysis_date, str):
        try: analysis_date = datetime.strptime(analysis_date, "%Y-%m-%d").date()
        except ValueError: print("Error: Invalid date string format."); return False

    if MARKET_CONTEXT.get("ticker") == ticker and MARKET_CONTEXT.get("date") == analysis_date and \
       MARKET_CONTEXT.get("spot_price") is not None and MARKET_CONTEXT.get("rate_interpolator") is not None:
        return True # Already cached

    print(f"\nUpdating market context for {ticker} on {analysis_date}...");
    # Clear previous context before updating
    MARKET_CONTEXT = {"ticker": ticker, "date": analysis_date}
    start_ctx = time.time()
    spot = get_spot_price(ticker, analysis_date);
    if spot is None: print("Failed to get spot price."); return False;
    MARKET_CONTEXT["spot_price"] = spot
    yield_curve = fetch_daily_yield_curve(analysis_date);
    if yield_curve is None: print("Failed to fetch yield curve."); return False;
    MARKET_CONTEXT["yield_curve_data"] = yield_curve
    rate_interpolator = create_rate_interpolator(yield_curve);
    if rate_interpolator is None: print("Failed to create rate interpolator."); return False;
    MARKET_CONTEXT["rate_interpolator"] = rate_interpolator
    print(f"Market context updated successfully (Spot={spot:.2f}). Took {time.time() - start_ctx:.2f}s");
    return True

def get_rate_for_T(T):
    """Safely gets interpolated rate for T using cached interpolator."""
    interpolator = MARKET_CONTEXT.get("rate_interpolator");
    if interpolator is None: print("Error: Rate interpolator N/A."); return None
    try:
        r_t = interpolator(T);
        if pd.isna(r_t): print(f"Warning: Interpolated rate for T={T:.4f} is NaN."); return None
        # if not (-0.05 < r_t < 0.25): print(f"Warning: Rate {r_t:.4f} for T={T:.4f} unrealistic.") # Optional check
        return float(r_t)
    except Exception as e: print(f"Error interpolating rate for T={T:.4f}: {e}"); return None

# --- Utility to get ticker and file paths ---
def get_ticker_and_paths(default_ticker, default_data_dir):
    # (Unchanged)
     ticker = get_string_input("Enter Ticker Symbol:", default_ticker); data_dir = get_string_input("Enter Data Directory:", default_data_dir)
     pattern = f"{ticker}_call_options_"
     try:
         if not os.path.isdir(data_dir): print(f"Error: Data directory not found: {data_dir}"); return ticker, None
         all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and f.startswith(pattern)]
         if not all_files: print(f"Warning: No data files found for '{ticker}' in {data_dir}"); return ticker, []
         else: print(f"Found {len(all_files)} data files for {ticker}."); file_paths = [os.path.join(data_dir, f) for f in all_files]; return ticker, file_paths
     except Exception as e: print(f"Error accessing data directory: {e}"); return ticker, None

# --- Menu Display Function ---
def display_main_menu():
    # (Unchanged)
    print("\n--- Main Menu ---")
    print("1. Collect Option Data")
    print("2. Calculate Black-Scholes Price")
    print("3. Calculate Implied Volatility")
    print("4. Interpolate Volatility (P-Spline)") # Renamed slightly
    print("5. Plot Surface (P-Spline)")
    print("6. Analyze Errors (P-Spline)")
    print("0. Exit")
    print("-----------------")

# --- Handler Functions (Simplified Input) ---

def run_data_collection():
    # (Unchanged)
    print("\n--- Data Collection ---"); ticker = get_string_input("Enter Ticker Symbol:", DEFAULT_TICKER); data_dir = get_string_input("Enter Data Dir:", DEFAULT_DATA_DIR)
    try: print(f"Starting collection for {ticker}..."); collect_option_data(ticker, data_dir); print("Collection finished.")
    except Exception as e: print(f"Error: {e}"); traceback.print_exc()

def run_black_scholes():
    # (Unchanged)
    print("\n--- Black-Scholes Calculator ---")
    current_ticker = MARKET_CONTEXT.get("ticker", DEFAULT_TICKER); ticker_to_use = get_string_input("Ticker:", current_ticker)
    analysis_date = date.today(); # Use today's date for interactive BS
    if not update_market_context(ticker_to_use, analysis_date): print("Cannot proceed."); return
    default_spot = MARKET_CONTEXT["spot_price"] if MARKET_CONTEXT["ticker"] == ticker_to_use else None
    S = get_float_input("Underlying (S)", default_spot); K = get_float_input("Strike (K)", 100.0); T = get_float_input("TTM (T years)", 1.0)
    sigma = get_float_input("Volatility (sigma)", 0.2); r_t = get_rate_for_T(T)
    if r_t is None: print("Cannot calc without rate."); return
    if S is None or K is None or T is None or sigma is None: print("Error: Missing params."); return
    try: price = black_scholes_call(S, K, T, r_t, sigma); print(f"\nCalculated BS Call Price: {price:.4f} (r={r_t:.4f})")
    except Exception as e: print(f"BS Calc Error: {e}"); traceback.print_exc()

def run_implied_vol():
    # (Unchanged - still uses the modified function that returns a dict)
    print("\n--- Implied Volatility Calculator ---"); current_ticker = MARKET_CONTEXT.get("ticker", DEFAULT_TICKER); ticker_to_use = get_string_input("Ticker:", current_ticker)
    analysis_date = date.today(); # Use today's date for interactive IV
    if not update_market_context(ticker_to_use, analysis_date): print("Cannot proceed."); return
    default_spot = MARKET_CONTEXT["spot_price"] if MARKET_CONTEXT["ticker"] == ticker_to_use else None
    S = get_float_input("Underlying (S)", default_spot); K = get_float_input("Strike (K)", 100.0); T = get_float_input("TTM (T years)", 1.0)
    market_price = get_float_input("Market Price", 10.5); initial_sigma = get_float_input("Initial Sigma", 0.2)
    r_t = get_rate_for_T(T);
    if r_t is None: print("Cannot calc without rate."); return
    if market_price is None or S is None or K is None or T is None: print("Error: Missing params."); return
    try:
        iv_result = implied_volatility(market_price, S, K, T, r_t, initial_sigma=initial_sigma)
        print("\n--- IV Calculation Result ---")
        for key, value in iv_result.items():
            if isinstance(value, float): print(f"  {key:<20}: {value:.6f}")
            else: print(f"  {key:<20}: {value}")
        if iv_result['status'] != 'Converged': print("  Warning: Calculation may not be reliable.")
    except Exception as e: print(f"IV Calc Error: {e}"); traceback.print_exc()

def run_interpolation(): # <<< MODIFIED INPUTS
    print("\n--- Interpolate Volatility (P-Spline) ---")
    current_ticker = MARKET_CONTEXT.get("ticker", DEFAULT_TICKER); ticker, file_paths = get_ticker_and_paths(current_ticker, DEFAULT_DATA_DIR)
    if not file_paths: print(f"No data files found for {ticker}."); return

    strike = get_float_input("Strike to interpolate:", 200.0);
    maturity = get_float_input("Maturity (years) to interpolate:", 0.5)
    max_maturity_filter = get_float_input("Max Maturity for spline data (years):", DEFAULT_MAX_MATURITY)

    # --- Use Fixed Defaults for Filter/Structure ---
    current_filter_params = DEFAULT_FILTER_PARAMS.copy()
    print(f"Using Default Filter parameters: {current_filter_params}")
    spline_structure_params = {
        "degree_x": DEFAULT_SPLINE_PARAMS['degree_x'],
        "degree_y": DEFAULT_SPLINE_PARAMS['degree_y'],
        "knot_params_x": DEFAULT_SPLINE_PARAMS['knot_params_x'],
        "knot_params_y": DEFAULT_SPLINE_PARAMS['knot_params_y'],
    }
    print(f"Using Default Spline Structure (Degree/Knots): {spline_structure_params}")
    # ---

    # --- Only Prompt for Lambdas ---
    lambda_x = get_float_input("Smoothness Lambda X (strike):", DEFAULT_SPLINE_PARAMS['lambda_x'])
    lambda_y = get_float_input("Smoothness Lambda Y (TTM):", DEFAULT_SPLINE_PARAMS['lambda_y'])
    # ---

    # Combine all parameters needed, using defaults for penalty order and solver params
    all_params = {
        "max_maturity_years": max_maturity_filter,
        **current_filter_params,
        **spline_structure_params,
        "lambda_x": lambda_x,
        "lambda_y": lambda_y,
        "penalty_order": DEFAULT_SPLINE_PARAMS['penalty_order'], # Use default
        "lsqr_iter_lim": DEFAULT_SPLINE_PARAMS['lsqr_iter_lim'], # Use default
        "lsqr_tol": DEFAULT_SPLINE_PARAMS['lsqr_tol']           # Use default
    }

    try:
        print("\nRunning interpolation...")
        # get_interpolated_volatility internally calls prepare_data and create_spline_representation
        vol = get_interpolated_volatility(file_paths, strike, maturity, verbose=True, **all_params)
        if vol is not None and not np.isnan(vol): print(f"\nInterpolated Volatility at (Strike={strike}, Maturity={maturity:.4f}): {vol:.4f}")
        else: print("\nInterpolation failed or returned NaN.")
    except Exception as e: print(f"Interpolation Error: {e}"); traceback.print_exc()

def run_plot_surface(): # <<< MODIFIED INPUTS
    print("\n--- Plot Surface (P-Spline) ---")
    current_ticker = MARKET_CONTEXT.get("ticker", DEFAULT_TICKER); ticker, file_paths = get_ticker_and_paths(current_ticker, DEFAULT_DATA_DIR)
    if not file_paths: print(f"No data files found for {ticker}."); return

    max_maturity_filter = get_float_input("Max Maturity for plot data (years):", DEFAULT_MAX_MATURITY)

    # --- Use Fixed Defaults for Filter/Structure ---
    current_filter_params = DEFAULT_FILTER_PARAMS.copy()
    print(f"Using Default Filter parameters: {current_filter_params}")
    spline_structure_params = {
        "degree_x": DEFAULT_SPLINE_PARAMS['degree_x'],
        "degree_y": DEFAULT_SPLINE_PARAMS['degree_y'],
        "knot_params_x": DEFAULT_SPLINE_PARAMS['knot_params_x'],
        "knot_params_y": DEFAULT_SPLINE_PARAMS['knot_params_y'],
    }
    print(f"Using Default Spline Structure (Degree/Knots): {spline_structure_params}")
    # ---

    # --- Only Prompt for Lambdas ---
    lambda_x = get_float_input("Smoothness Lambda X (strike):", DEFAULT_SPLINE_PARAMS['lambda_x'])
    lambda_y = get_float_input("Smoothness Lambda Y (TTM):", DEFAULT_SPLINE_PARAMS['lambda_y'])
    # ---

    # Combine all spline/solver parameters needed by plot_volatility_surface
    plot_spline_params = {
        **spline_structure_params,
        "lambda_x": lambda_x,
        "lambda_y": lambda_y,
        "penalty_order": DEFAULT_SPLINE_PARAMS['penalty_order'], # Use default
        "lsqr_iter_lim": DEFAULT_SPLINE_PARAMS['lsqr_iter_lim'], # Use default
        "lsqr_tol": DEFAULT_SPLINE_PARAMS['lsqr_tol']           # Use default
    }

    z_min = get_float_input("Plot Z-axis Minimum IV", 0.0)
    z_max = get_float_input("Plot Z-axis Maximum IV", 1.5)

    try:
        print("\nGenerating plot...")
        fig = plot_volatility_surface(
            file_paths,
            max_maturity_years=max_maturity_filter,
            filter_params=current_filter_params,
            spline_params=plot_spline_params, # Pass the combined dict
            z_lim_min=z_min, z_lim_max=z_max
        )
        if fig:
            plt.show() # Show plot interactively
            plt.close(fig)
            print("Plot closed.")
        else:
            print("Plot generation failed.")

    except Exception as e: print(f"Plotting Error: {e}"); traceback.print_exc(); plt.close('all')

def run_analyze_errors(): # <<< MODIFIED INPUTS
    print("\n--- Analyze Interpolation Errors (P-Spline) ---")
    current_ticker = MARKET_CONTEXT.get("ticker", DEFAULT_TICKER); ticker, file_paths = get_ticker_and_paths(current_ticker, DEFAULT_DATA_DIR)
    if not file_paths: print(f"No data files found for {ticker}."); return

    # Use analysis date consistent with how context is updated
    analysis_date = date.today(); # Use today's date to align with context fetching
    if not update_market_context(ticker, analysis_date): print("Cannot proceed without market context."); return
    spot_price = MARKET_CONTEXT["spot_price"]; rate_interpolator = MARKET_CONTEXT["rate_interpolator"]
    if spot_price is None or rate_interpolator is None: print("Error: Market context missing."); return

    max_maturity_filter = get_float_input("Max Maturity for analysis data (years):", DEFAULT_MAX_MATURITY)

    # --- Use Fixed Defaults for Filter/Structure ---
    current_filter_params = DEFAULT_FILTER_PARAMS.copy()
    print(f"Using Default Filter parameters: {current_filter_params}")
    spline_structure_params = {
        "degree_x": DEFAULT_SPLINE_PARAMS['degree_x'],
        "degree_y": DEFAULT_SPLINE_PARAMS['degree_y'],
        "knot_params_x": DEFAULT_SPLINE_PARAMS['knot_params_x'],
        "knot_params_y": DEFAULT_SPLINE_PARAMS['knot_params_y'],
    }
    print(f"Using Default Spline Structure (Degree/Knots): {spline_structure_params}")
    # ---

    # --- Only Prompt for Lambdas ---
    lambda_x = get_float_input("Smoothness Lambda X (strike):", DEFAULT_SPLINE_PARAMS['lambda_x'])
    lambda_y = get_float_input("Smoothness Lambda Y (TTM):", DEFAULT_SPLINE_PARAMS['lambda_y'])
    # ---

    # Prepare kwargs for analyze_interpolation_errors (solver/lambda params)
    solver_lambda_kwargs = {
        "lambda_x": lambda_x,
        "lambda_y": lambda_y,
        "penalty_order": DEFAULT_SPLINE_PARAMS['penalty_order'], # Use default
        "lsqr_iter_lim": DEFAULT_SPLINE_PARAMS['lsqr_iter_lim'], # Use default
        "lsqr_tol": DEFAULT_SPLINE_PARAMS['lsqr_tol']           # Use default
    }

    try:
        print("\nRunning analysis (including IV recalculation)...")
        # Call analyze_interpolation_errors which now returns (df, convergence_dict)
        analysis_result = analyze_interpolation_errors(
            file_paths,
            spot_price=spot_price,
            rate_interpolator=rate_interpolator,
            max_maturity_years=max_maturity_filter,
            filter_params=current_filter_params,
            # Pass spline structure args explicitly
            degree_x=spline_structure_params['degree_x'],
            degree_y=spline_structure_params['degree_y'],
            knot_params_x=spline_structure_params['knot_params_x'],
            knot_params_y=spline_structure_params['knot_params_y'],
            # Pass solver/lambda params via kwargs
            **solver_lambda_kwargs
        )

        # Unpack results
        error_dataframe = None
        convergence_info = {}
        if isinstance(analysis_result, tuple) and len(analysis_result) == 2:
            error_dataframe, convergence_info = analysis_result
        elif analysis_result is not None: # Handle case if only df is returned (less likely now)
            error_dataframe = analysis_result

        if error_dataframe is not None and not error_dataframe.empty:
            print("\nAnalysis complete. Visualizing errors...")
            # --- Extract and print summary stats ---
            print("\nError Summary Statistics:")
            if 'iv_abs_error' in error_dataframe.columns and error_dataframe['iv_abs_error'].notna().any():
                 print(f"  IV MAE       : {error_dataframe['iv_abs_error'].mean():.6f}")
                 print(f"  IV RMSE      : {np.sqrt((error_dataframe['iv_error']**2).mean()):.6f}")
                 print(f"  IV Median AE : {error_dataframe['iv_abs_error'].median():.6f}")
            else: print("  IV Errors: Not available or all NaN.")

            if 'price_abs_error' in error_dataframe.columns and error_dataframe['price_abs_error'].notna().any():
                 print(f"  Price MAE    : ${error_dataframe['price_abs_error'].mean():.4f}")
                 print(f"  Price RMSE   : ${np.sqrt((error_dataframe['price_error']**2).mean()):.4f}")
                 print(f"  Price Median AE: ${error_dataframe['price_abs_error'].median():.4f}")
            else: print("  Price Errors: Not available or all NaN.")

            # --- Print convergence info ---
            print("\nConvergence Summary:")
            print(f"  IV Recalc Success Rate : {convergence_info.get('iv_success_rate', np.nan):.3f}")
            print(f"  Avg IV Recalc Iterations: {convergence_info.get('avg_iv_iterations', np.nan):.2f}")
            print(f"  Spline LSQR Status Code: {convergence_info.get('spline_solver_status', 'N/A')}")
            print(f"  Spline LSQR Iterations : {convergence_info.get('spline_solver_iterations', 'N/A')}")
            # --- End Stats ---

            fig = visualize_errors(error_dataframe, ticker=ticker)
            if fig:
                plt.show() # Show interactively
                plt.close(fig)
                print("Error plot closed.")
            else:
                print("Error visualization failed.")
        else:
            print("\nError analysis failed or produced no data.")
            # Print convergence info even if analysis failed later
            print("\nConvergence Summary (Partial):")
            print(f"  IV Recalc Success Rate : {convergence_info.get('iv_success_rate', np.nan):.3f}")
            print(f"  Avg IV Recalc Iterations: {convergence_info.get('avg_iv_iterations', np.nan):.2f}")
            print(f"  Spline LSQR Status Code: {convergence_info.get('spline_solver_status', 'N/A')}")
            print(f"  Spline LSQR Iterations : {convergence_info.get('spline_solver_iterations', 'N/A')}")


    except Exception as e:
        print(f"Error During Analysis/Visualization: {e}")
        traceback.print_exc()
        plt.close('all')


# --- Main Loop ---
def main_loop():
    # (Unchanged)
    print("Welcome to the Options Analysis Toolkit!")
    while True:
        display_main_menu()
        choice = get_int_input("Enter your choice:", -1)
        try:
            if choice == 1: run_data_collection()
            elif choice == 2: run_black_scholes()
            elif choice == 3: run_implied_vol()
            elif choice == 4: run_interpolation()      # Uses simplified input
            elif choice == 5: run_plot_surface()         # Uses simplified input
            elif choice == 6: run_analyze_errors()       # Uses simplified input
            elif choice == 0: print("Exiting."); break
            else: print("Invalid choice, please try again.")
        except Exception as handler_e: print(f"\n--- UNEXPECTED ERROR: {handler_e} ---"); traceback.print_exc()
        if choice != 0: input("\nPress Enter to return to the menu...")

# --- Script Execution ---
if __name__ == '__main__':
    main_loop()