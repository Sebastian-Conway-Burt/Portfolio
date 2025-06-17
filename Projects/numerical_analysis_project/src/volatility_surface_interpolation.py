# src/volatility_surface_interpolation.py (Updated for Penalized Splines & Convergence Return)

import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime, date, timedelta
import warnings
import traceback

# --- Import Custom B-Spline Implementation ---
try:
    # Import the NEW penalized fitting function and the evaluator
    # Assumes fit_bspline_surface_penalized now includes convergence info in spline_object
    from src.custom_bspline import fit_bspline_surface_penalized, evaluate_bspline_surface
except ImportError:
     # Fallback for running directly from src/?
    try:
      from custom_bspline import fit_bspline_surface_penalized, evaluate_bspline_surface
    except ImportError as e:
      print(f"Error importing custom_bspline: {e}")
      print("Ensure custom_bspline.py includes 'fit_bspline_surface_penalized'")
      sys.exit(1)

# --- Import Implied Volatility (for potential future use, though not called in prepare_volatility_data now) ---
try:
    # Assumes implied_volatility returns a dict now
    from src.implied_volatility import implied_volatility
except ImportError:
    try:
        from implied_volatility import implied_volatility
    except ImportError:
        print("Warning: implied_volatility module not found.")
        implied_volatility = None # Define as None if not found
# --- End Import ---


# -------------------------
# Utility Functions (Unchanged)
# -------------------------
def calculate_time_to_maturity(expiry_date_str, last_trade_date_str):
    """Calculates time to maturity in years."""
    # (Function unchanged)
    try:
        expiry_date = datetime.strptime(expiry_date_str, '%Y-%m-%d')
        if isinstance(last_trade_date_str, str):
            last_trade_date_part = last_trade_date_str.split()[0]
            last_trade_date = datetime.strptime(last_trade_date_part, '%Y-%m-%d')
        elif isinstance(last_trade_date_str, (datetime, date, pd.Timestamp)):
             last_trade_date = pd.to_datetime(last_trade_date_str).normalize().to_pydatetime()
        else: return np.nan # Cannot handle other types
        if expiry_date < last_trade_date: return 0.0 # Option expired
        time_difference = expiry_date - last_trade_date
        # Use a small positive floor for TTM > 0
        return max(1e-6, time_difference.days / 365.0)
    except (ValueError, TypeError): return np.nan
    except Exception as e: return np.nan


# -------------------------
# Data Preparation
# -------------------------
def prepare_volatility_data(data_file_paths,
                            max_maturity_years=1.5,
                            min_iv=0.01,
                            max_iv=1.5,
                            min_volume=3,
                            min_open_interest=0,
                            max_bid_ask_spread_abs=0.50,
                            max_bid_ask_spread_rel=0.30,
                            min_ttm_years=0.005,
                            verbose=False
                           ):
    """
    Loads, cleans, filters, and groups raw options volatility data.

    NOTE: This function DOES NOT recalculate implied volatility with convergence tracking.
          It requires spot price and risk-free rate, which are not available here.
          IV calculation with convergence should happen later in the analysis pipeline
          (e.g., within analyze_interpolation_errors).
          This function primarily focuses on filtering based on market data quality
          and grouping by strike/maturity. It uses the 'impliedVolatility' column
          from the input files mainly for filtering purposes if min_iv/max_iv are set.
    """
    all_data = []
    if verbose: print(f"\n--- Preparing Raw Option Data ...")
    try: # Check files and columns
        if not data_file_paths: print("Error: No data files provided."); return pd.DataFrame()
        # Determine available columns from the first file
        first_df_cols = pd.read_csv(data_file_paths[0], nrows=0).columns
        # Core columns needed for processing here
        core_cols = ['strike', 'impliedVolatility', 'lastTradeDate']
        # Optional columns used for filtering or pricing
        opt_cols = ['volume', 'openInterest', 'bid', 'ask', 'lastPrice']
        needed_cols = core_cols + opt_cols
        available_cols = list(set(needed_cols).intersection(first_df_cols))

        # Check critical columns for calculating market price
        can_calc_market_price = False
        if 'bid' in available_cols and 'ask' in available_cols:
             can_calc_market_price = True
        elif 'lastPrice' in available_cols:
             can_calc_market_price = True
             if verbose: print("  Note: Using 'lastPrice' as market price (bid/ask missing).")
        else:
            print("Error: Cannot determine market price ('bid'/'ask' or 'lastPrice' missing)."); return pd.DataFrame()

    except Exception as e: print(f"Error reading columns: {e}"); return pd.DataFrame()

    required_cols_core = ['strike', 'impliedVolatility', 'lastTradeDate']
    processed_files_count = 0; skipped_files_count = 0; valid_data_found = False; total_files = len(data_file_paths)

    # --- Loop through each raw data file ---
    for i, file_path in enumerate(data_file_paths):
        try:
            df = pd.read_csv(file_path, usecols=lambda col: col in available_cols)
            if df.empty or not all(col in df.columns for col in required_cols_core):
                skipped_files_count += 1; continue

            df.dropna(subset=required_cols_core, inplace=True);
            if df.empty: skipped_files_count += 1; continue

            # Extract expiry date from filename
            file_name = os.path.basename(file_path); expiry_date_str = file_name.split('_')[-1].split('.')[0]
            datetime.strptime(expiry_date_str, '%Y-%m-%d') # Validate format

            # Calculate Time to Maturity (TTM)
            df['time_to_maturity'] = df.apply(lambda row: calculate_time_to_maturity(expiry_date_str, row['lastTradeDate']), axis=1)
            df.dropna(subset=['time_to_maturity'], inplace=True);
            if df.empty: skipped_files_count += 1; continue

            # Convert relevant columns to numeric
            cols_to_numeric = [col for col in available_cols if col != 'lastTradeDate'];
            for col in cols_to_numeric: df[col] = pd.to_numeric(df[col], errors='coerce')
            essential_numeric = ['strike', 'impliedVolatility', 'time_to_maturity'];
            df.dropna(subset=[col for col in essential_numeric if col in df.columns], inplace=True);
            if df.empty: skipped_files_count += 1; continue

            # Calculate Market Mid Price
            if 'bid' in df.columns and 'ask' in df.columns:
                 df.dropna(subset=['bid', 'ask'], inplace=True)
                 valid_spread = (df['bid'] > 1e-6) & (df['ask'] > 1e-6) & (df['ask'] >= df['bid'])
                 df['market_mid_price'] = np.nan
                 df.loc[valid_spread, 'market_mid_price'] = (df.loc[valid_spread, 'bid'] + df.loc[valid_spread, 'ask']) / 2
                 # Fill missing mid-prices with lastPrice if available
                 if 'lastPrice' in df.columns: df['market_mid_price'] = df['market_mid_price'].fillna(df['lastPrice'])
            elif 'lastPrice' in df.columns:
                 df['market_mid_price'] = df['lastPrice']
            # No else needed due to initial check

            df.dropna(subset=['market_mid_price'], inplace=True);
            if df.empty: skipped_files_count += 1; continue

            # --- Apply Filters ---
            initial_rows = len(df)
            # TTM Filter
            if min_ttm_years is not None and min_ttm_years > 0: df = df[df['time_to_maturity'] >= min_ttm_years]
            # IV Filter (using original IV from data)
            df = df[(df['impliedVolatility'] >= min_iv) & (df['impliedVolatility'] <= max_iv)]
            # Volume Filter
            if 'volume' in available_cols and min_volume is not None: df = df[df['volume'].fillna(0) >= min_volume]
            # Open Interest Filter
            if 'openInterest' in available_cols and min_open_interest is not None: df = df[df['openInterest'].fillna(0) >= min_open_interest]
            # Spread Filter
            if 'bid' in available_cols and 'ask' in available_cols:
                 # Create temporary df for spread calculations only on valid bid/ask rows
                 df_valid = df[(df['bid']>1e-6)&(df['ask']>1e-6)&(df['ask']>=df['bid'])].copy();
                 if not df_valid.empty:
                     df_valid['abs_spread']=df_valid['ask']-df_valid['bid']
                     df_valid['mid_p']=(df_valid['ask']+df_valid['bid'])/2 # Use calculated mid here
                     # Avoid division by zero for relative spread
                     df_valid['rel_spread']=np.where(df_valid['mid_p']>1e-6, df_valid['abs_spread']/df_valid['mid_p'], np.inf)
                     # Find indices that fail the spread filters
                     invalid_abs_idx = df_valid[df_valid['abs_spread']>max_bid_ask_spread_abs].index if max_bid_ask_spread_abs is not None else pd.Index([])
                     invalid_rel_idx = df_valid[df_valid['rel_spread']>max_bid_ask_spread_rel].index if max_bid_ask_spread_rel is not None else pd.Index([])
                     invalid_spread_idx = invalid_abs_idx.union(invalid_rel_idx)
                     # Keep only rows that are NOT in the invalid spread index set
                     df = df.drop(invalid_spread_idx)

            if df.empty: skipped_files_count += 1; continue # Skip if filters removed all data

            # --- Select final columns for this file ---
            # Keep original IV for now, it will be handled/replaced later
            final_cols=['strike','time_to_maturity','impliedVolatility','market_mid_price']
            cols_to_keep=[c for c in final_cols if c in df.columns]
            data_subset=df[cols_to_keep].dropna()

            if not data_subset.empty:
                 all_data.append(data_subset)
                 valid_data_found = True
            else:
                 skipped_files_count += 1; continue # Skip if final selection is empty

        except Exception as e:
            print(f"  ERROR processing {file_path}: {e}. Skip."); traceback.print_exc(limit=1); skipped_files_count += 1; continue

        processed_files_count += 1
    # --- End File Loop ---

    if verbose: print(f"\nFinished processing loop.");
    print(f"Successfully processed {processed_files_count}/{total_files} files.")
    if skipped_files_count > 0: print(f"Skipped {skipped_files_count} files due to errors or empty data after filtering.")

    if not valid_data_found: print("No valid data accumulated across all files."); return pd.DataFrame()

    # Combine data from all processed files
    combined_data = pd.concat(all_data, ignore_index=True)
    if verbose: print(f"Combined data rows before final checks: {len(combined_data)}")

    # Final check for NaNs in essential columns
    final_check_cols = ['strike', 'time_to_maturity', 'impliedVolatility', 'market_mid_price'];
    cols_present = [c for c in final_check_cols if c in combined_data.columns]
    for col in cols_present: combined_data[col] = pd.to_numeric(combined_data[col], errors='coerce')
    rows_before = len(combined_data); combined_data.dropna(subset=cols_present, inplace=True); rows_after = len(combined_data)
    if verbose and (rows_before - rows_after > 0): print(f"Combined data rows after final NaN drop: {rows_after} (removed {rows_before - rows_after}).")
    if combined_data.empty: print("Combined data empty after final checks."); return pd.DataFrame()

    # Apply max maturity filter globally after combining
    if max_maturity_years is not None:
         rows_before = len(combined_data);
         filtered_data = combined_data[combined_data['time_to_maturity'] <= max_maturity_years].copy()
         rows_after = len(filtered_data)
         if verbose: print(f"Data points after Max Maturity ({max_maturity_years}yr): {rows_after} (removed {rows_before - rows_after})")
         if filtered_data.empty: print("No data remaining after max maturity filter."); return pd.DataFrame()
    else:
         filtered_data = combined_data.copy()

    # --- Grouping by Strike and Maturity ---
    # Average the market price and the original IV for duplicate points.
    # More sophisticated aggregation (e.g., using IV status/iterations) should happen later.
    group_cols = ['strike', 'time_to_maturity'];
    cols_to_avg = ['impliedVolatility', 'market_mid_price'] # Average original IV and market price
    final_cols_group = group_cols + [c for c in cols_to_avg if c in filtered_data.columns]

    if not all(c in final_cols_group for c in group_cols): print("Error: Required grouping columns missing."); return pd.DataFrame()

    data_to_group = filtered_data[final_cols_group].copy();
    rows_before = len(data_to_group);
    agg_dict = {c:'mean' for c in cols_to_avg if c in data_to_group.columns} # Simple mean aggregation

    if not agg_dict: # Should not happen if checks above passed
        grouped_data = data_to_group.drop_duplicates(subset=group_cols)
    else:
        # Perform grouping
        grouped_data = data_to_group.groupby(group_cols, as_index=False).agg(agg_dict)

    if verbose: print(f"Grouped data unique points: {len(grouped_data)} (from {rows_before} initial valid rows after filtering).")
    print(f"--- Data Preparation Complete ---")
    return grouped_data


# -------------------------
# Spline Creation (Using Penalized Splines, returning convergence)
# -------------------------
def create_spline_representation(grouped_data,
                                 degree_x=3, degree_y=3,
                                 knot_params_x=None,
                                 knot_params_y=None,
                                 verbose=False,
                                 # Accept smoothness params & other solver params via kwargs
                                 **solver_params_and_lambdas
                                ):
    """
    Creates a 2D penalized spline representation using custom B-spline fitter.
    Extracts lambda_x, lambda_y and passes them to fit_bspline_surface_penalized.

    Error Sources:
    - Approximation Error: How well the spline function class (defined by degree
      and knots) can represent the true underlying volatility surface. Depends on
      surface smoothness, knot density/placement, and spline degree.
    - Smoothing Error (P-Splines): Introduced by the penalty terms (lambda_x, lambda_y).
      A trade-off exists: higher lambdas lead to smoother surfaces (less variance,
      more bias) but may deviate more from individual data points. Lower lambdas
      fit data closer but can be wiggly (more variance, less bias).
    - Numerical Stability: The least-squares problem (especially when penalized)
      can be ill-conditioned if knots are too close, data is sparse, or the design
      matrix has dependent columns. LSQR is generally robust but can fail or take
      many iterations (indicated by lsqr_status/lsqr_iterations).

    Returns:
        tuple: (spline_object, spline_convergence_info)
            - spline_object (dict or None): Contains knots, coeffs, degrees, and lsqr info.
            - spline_convergence_info (dict): {'lsqr_iterations': int, 'lsqr_status': int}
    """
    # Initialize convergence info dictionary
    spline_convergence_info = {
        'lsqr_iterations': np.nan,
        'lsqr_status': np.nan
    }

    # Handle None defaults for knot params
    if knot_params_x is None: knot_params_x = {'num_internal_knots': 8, 'strategy': 'uniform', 'min_separation': 1.0}
    if knot_params_y is None: knot_params_y = {'num_internal_knots': 5, 'strategy': 'uniform', 'min_separation': 0.01}

    if grouped_data is None or grouped_data.empty:
        print("Error: Cannot create spline from empty data."); return None, spline_convergence_info
    required_cols = ['strike', 'time_to_maturity', 'impliedVolatility'] # Uses original IV for fitting
    if not all(col in grouped_data.columns for col in required_cols):
        print(f"Error: Missing required columns: {required_cols}."); return None, spline_convergence_info

    x_data = grouped_data['strike'].values
    y_data = grouped_data['time_to_maturity'].values
    z_data = grouped_data['impliedVolatility'].values # Fit to the (grouped) original IV

    # --- Extract parameters for the penalized fitter ---
    lambda_x = solver_params_and_lambdas.pop('lambda_x', 0.0) # Pop with default
    lambda_y = solver_params_and_lambdas.pop('lambda_y', lambda_x) # Default Y to X if not given
    penalty_order = solver_params_and_lambdas.pop('penalty_order', 2)
    # Remaining items in solver_params_and_lambdas are passed to lsqr
    lsqr_kwargs = solver_params_and_lambdas

    if verbose:
        print(f"\nAttempting to fit Penalized B-Spline surface...")
        print(f"Degrees: dx={degree_x}, dy={degree_y}")
        print(f"Knot Params X: {knot_params_x}"); print(f"Knot Params Y: {knot_params_y}")
        print(f"Smoothness Params: lambda_x={lambda_x:.2e}, lambda_y={lambda_y:.2e}, penalty_order={penalty_order}")
        if lsqr_kwargs: print(f"LSQR Params: {lsqr_kwargs}")

    # --- Call the penalized fitting function ---
    # Assumes fit_bspline_surface_penalized now returns a spline_object dict
    # which *includes* 'lsqr_status' and 'lsqr_iterations'
    spline_object = fit_bspline_surface_penalized(
        x_data, y_data, z_data,
        degree_x, degree_y,
        knot_params_x, knot_params_y,
        lambda_x=lambda_x,
        lambda_y=lambda_y,
        penalty_order=penalty_order,
        verbose=verbose,
        **lsqr_kwargs
    )

    if spline_object is None:
        print("Penalized B-Spline fitting failed."); return None, spline_convergence_info

    # --- Extract convergence info from the returned spline object ---
    spline_convergence_info['lsqr_iterations'] = spline_object.get('lsqr_iterations', np.nan)
    spline_convergence_info['lsqr_status'] = spline_object.get('lsqr_status', np.nan)

    # Return the spline object and the extracted convergence info
    return spline_object, spline_convergence_info


# -------------------------
# Single Point Interpolation (Handles new return from create_spline_representation)
# -------------------------
def get_interpolated_volatility(data_file_paths, strike_interp, maturity_interp, verbose=False, **kwargs):
    """
    Gets interpolated volatility for a single point using PENALIZED splines.
    Less efficient than batch analysis/evaluation.
    Handles the tuple return from create_spline_representation internally.
    """
    # Separate kwargs (unchanged)
    prepare_args = {}; spline_args = {}; solver_lambdas_args = {}
    prepare_keys = ['max_maturity_years', 'min_iv', 'max_iv', 'min_volume', 'min_open_interest', 'max_bid_ask_spread_abs', 'max_bid_ask_spread_rel', 'min_ttm_years']
    spline_keys = ['degree_x', 'degree_y', 'knot_params_x', 'knot_params_y']
    for key, value in kwargs.items():
        if key in prepare_keys: prepare_args[key] = value
        elif key in spline_keys: spline_args[key] = value
        else: solver_lambdas_args[key] = value

    # Set defaults (unchanged)
    if 'degree_x' not in spline_args: spline_args['degree_x'] = 3
    if 'degree_y' not in spline_args: spline_args['degree_y'] = 3

    # Prepare Data (unchanged)
    grouped_filtered_data = prepare_volatility_data(data_file_paths, verbose=verbose, **prepare_args)
    if grouped_filtered_data is None or grouped_filtered_data.empty:
        print("Interp failed: No data."); return np.nan

    # --- Create Penalized Spline Representation (Handle Tuple Return) ---
    spline_result = create_spline_representation(
        grouped_filtered_data,
        degree_x=spline_args['degree_x'],
        degree_y=spline_args['degree_y'],
        knot_params_x=spline_args.get('knot_params_x'),
        knot_params_y=spline_args.get('knot_params_y'),
        verbose=verbose,
        **solver_lambdas_args
    )

    # Unpack the tuple
    spline_object = None
    if isinstance(spline_result, tuple) and len(spline_result) == 2:
        spline_object, _ = spline_result # Ignore convergence info here
    elif spline_result is not None: # Handle backward compatibility if only object is returned
         spline_object = spline_result

    if spline_object is None:
        print("Interp failed: Spline creation failed."); return np.nan
    # --- End Handle Tuple ---

    # Evaluate Spline (Unchanged)
    try:
        volatility = evaluate_bspline_surface(strike_interp, maturity_interp, spline_object)
        if volatility is None or not np.isfinite(volatility):
            # print("Eval invalid or returned NaN.") # Optional print
            return np.nan
        return float(volatility)
    except Exception as e:
        print(f"Error during single point eval: {e}"); traceback.print_exc(limit=1); return np.nan

# --- Main Execution Block (Needs Update for Penalized) ---
if __name__ == '__main__':
    # ... (Example usage remains largely the same, as get_interpolated_volatility hides the internal tuple handling) ...
    script_dir = os.path.dirname(os.path.abspath(__file__)); project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data', 'raw')
    print(f"--- Running Volatility Surface Interpolation Example (Penalized) ---")
    print(f"Looking for CSV files in: {data_dir}")
    ticker_to_use = "AAPL"

    # Define example parameters for penalized fit
    default_filter_params = {"max_maturity_years": 1.0,"min_iv": 0.01, "max_iv": 1.5,"min_volume": 3,"min_open_interest": 0,"max_bid_ask_spread_abs": 0.50, "max_bid_ask_spread_rel": 0.30,"min_ttm_years": 0.005}
    default_spline_params = {"degree_x": 3,"degree_y": 3, "knot_params_x": {'num_internal_knots': 15, 'strategy': 'uniform', 'min_separation': 1e-6}, "knot_params_y": {'num_internal_knots': 10, 'strategy': 'uniform', 'min_separation': 1e-6}}
    # Include example smoothness parameters
    default_solver_lambdas = {"lambda_x": 1.0, "lambda_y": 10.0, "lsqr_iter_lim": 10000}

    all_params = {**default_filter_params, **default_spline_params, **default_solver_lambdas}

    try: pattern = f"{ticker_to_use}_call_options_*"; all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv') and f.startswith(pattern)]
    except Exception as e: print(f"File listing error: {e}"); all_files = []

    if not all_files: print(f"No {ticker_to_use} files found.")
    else:
        print(f"Found {len(all_files)} {ticker_to_use} files.")
        file_paths = all_files
        current_date_str = date.today().strftime("%Y-%m-%d")
        strike_1 = 190; expiry_1 = (date.today() + timedelta(days=180)).strftime("%Y-%m-%d")
        ttm_1 = calculate_time_to_maturity(expiry_1, current_date_str); vol_1_str = "NaN"

        if ttm_1 is not None and ttm_1 > 0:
            print(f"\nAttempting single point interpolation (Penalized) for {ticker_to_use} K={strike_1}, Expiry={expiry_1} (TTM={ttm_1:.4f})")
            # Pass all params; get_interpolated_volatility will sort them out
            vol_1 = get_interpolated_volatility(file_paths, strike_1, ttm_1, verbose=True, **all_params)
            if vol_1 is not None and not np.isnan(vol_1): vol_1_str = f"{vol_1:.4f}"
            print(f">>> Result (Penalized Single Point): Vol(K={strike_1}, TTM={ttm_1:.4f}) = {vol_1_str}")
        else: print(f"\nCould not calculate valid TTM for Expiry={expiry_1}")