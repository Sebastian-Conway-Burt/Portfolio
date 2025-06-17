# src/data_analysis.py (Fixing time conflict, Implementing IV Recalculation)

import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
# --- Import time module separately ---
import time # <<< FIX: Import the time module
# --- Import specific components from datetime ---
from datetime import datetime, date, timedelta # Removed time from here to avoid confusion, can be added back if needed elsewhere
import warnings
import traceback
import yfinance as yf

# --- Project Path Setup ---
# (Unchanged)
script_dir = os.path.dirname(os.path.abspath(__file__)) #
project_root = os.path.dirname(script_dir) #
if project_root not in sys.path: sys.path.insert(0, project_root) #
# --- End Path Setup ---

# --- Import Project Modules ---
try:
    from src.volatility_surface_interpolation import prepare_volatility_data, create_spline_representation #
    from src.custom_bspline import evaluate_bspline_surface #
    from src.market_data_utils import fetch_daily_yield_curve, create_rate_interpolator, get_spot_price #
    from src.black_scholes import black_scholes_call #
    from src.implied_volatility import implied_volatility #
except ImportError as e: #
    print(f"Error importing project modules in data_analysis: {e}") #
    try: # Fallback
        from volatility_surface_interpolation import prepare_volatility_data, create_spline_representation
        from custom_bspline import evaluate_bspline_surface
        from market_data_utils import fetch_daily_yield_curve, create_rate_interpolator, get_spot_price
        from black_scholes import black_scholes_call
        from implied_volatility import implied_volatility # Fallback import
    except ImportError: #
        print("Ensure all required src/ files are accessible.") #
        sys.exit(1) #
# --- End Imports ---


# --- Default Configuration ---
# (Unchanged)
DATA_DIR_DEFAULT = os.path.join(project_root, 'data', 'raw') #
TICKER_DEFAULT = "AAPL" #
MAX_MATURITY_YEARS_DEFAULT = 1.0 #
MIN_IV_THRESHOLD_DEFAULT = 1e-5 #
FILTER_DEFAULTS = { #
    "min_iv": 0.01, "max_iv": 1.5, "min_volume": 3, "min_open_interest": 0, #
    "max_bid_ask_spread_abs": 0.50, "max_bid_ask_spread_rel": 0.30, #
    "min_ttm_years": 0.005 #
} #
SPLINE_DEFAULTS = { #
    "degree_x": 3,"degree_y": 3, #
    "knot_params_x": {'num_internal_knots': 8, 'strategy': 'uniform', 'min_separation': 1.0}, #
    "knot_params_y": {'num_internal_knots': 5, 'strategy': 'uniform', 'min_separation': 0.01}, #
    "lambda_x": 0.00,"lambda_y": 0.00,"penalty_order": 2, #
    "lsqr_iter_lim": None,"lsqr_tol": 1e-8, #
} #
# --- End Default Configuration ---


# --- Analysis Function (Implementing IV Recalculation) ---
def analyze_interpolation_errors(file_paths,
                                 spot_price,
                                 rate_interpolator,
                                 max_maturity_years=MAX_MATURITY_YEARS_DEFAULT,
                                 degree_x=SPLINE_DEFAULTS['degree_x'],
                                 degree_y=SPLINE_DEFAULTS['degree_y'],
                                 knot_params_x=None,
                                 knot_params_y=None,
                                 filter_params=None,
                                 iv_tolerance=1e-7,
                                 iv_max_iterations=100,
                                 **solver_kwargs
                                ):
    """
    Analyzes interpolation errors using penalized B-Splines.
    Includes recalculation of Implied Volatility with convergence tracking.
    Uses the successfully recalculated IVs for spline fitting.
    Compares final interpolated IV against the original market IV.

    Returns:
        tuple: (pd.DataFrame, dict)
            - pd.DataFrame: Contains analysis data with errors.
            - dict: Contains convergence metrics (spline solver & IV).
    """
    convergence_metrics = {'spline_solver_iterations': np.nan,'spline_solver_status': np.nan,'avg_iv_iterations': np.nan,'iv_success_rate': np.nan} #

    # Handle defaults
    # (Unchanged)
    if filter_params is None: filter_params = FILTER_DEFAULTS.copy() #
    if knot_params_x is None: knot_params_x = SPLINE_DEFAULTS['knot_params_x'].copy() #
    if knot_params_y is None: knot_params_y = SPLINE_DEFAULTS['knot_params_y'].copy() #
    spline_structure_params = {"degree_x": degree_x, "degree_y": degree_y, "knot_params_x": knot_params_x, "knot_params_y": knot_params_y} #
    solver_lambda_params = { #
        "lambda_x": SPLINE_DEFAULTS['lambda_x'], "lambda_y": SPLINE_DEFAULTS['lambda_y'], #
        "penalty_order": SPLINE_DEFAULTS['penalty_order'],"lsqr_iter_lim": SPLINE_DEFAULTS['lsqr_iter_lim'], #
        "lsqr_tol": SPLINE_DEFAULTS['lsqr_tol'], #
    } #
    solver_lambda_params.update(solver_kwargs) #

    # Step 1: Prepare Data
    # (Unchanged)
    grouped_filtered_data = prepare_volatility_data( #
        file_paths, max_maturity_years=max_maturity_years, **filter_params #
    ) #
    if grouped_filtered_data is None or grouped_filtered_data.empty: #
        print("Analysis failed: No data after preparation.") #
        return None, convergence_metrics #
    analysis_data = grouped_filtered_data.copy() #

    # --- Step 1.5: Recalculate Implied Volatility ---
    # (Unchanged logic, but uses the imported 'time' module now)
    if implied_volatility is None: #
        print("Error: implied_volatility function not available for recalculation.") #
        return None, convergence_metrics #

    iv_results_list = [] #
    required_iv_cols = ['market_mid_price', 'strike', 'time_to_maturity'] #
    if not all(col in analysis_data.columns for col in required_iv_cols): #
        print(f"Error: Missing columns required for IV recalculation: {required_iv_cols}") #
        return None, convergence_metrics #

    if 'impliedVolatility' in analysis_data.columns: #
        analysis_data.rename(columns={'impliedVolatility': 'market_iv_original'}, inplace=True) #
    else: #
        print("Warning: Original 'impliedVolatility' column missing from prepared data.") #
        analysis_data['market_iv_original'] = np.nan #

    # --- Use imported time module ---
    calculation_start_time = time.time() # <<< FIX: Uses time module now
    for index, row in analysis_data.iterrows(): #
        market_price = row['market_mid_price']; K = row['strike']; T = row['time_to_maturity'] #
        try: #
            r_t = rate_interpolator(T) #
            if not np.isfinite(r_t): raise ValueError("Invalid interpolated rate") #
            initial_guess = row.get('market_iv_original', 0.2) #
            if pd.isna(initial_guess) or initial_guess < 0.01 or initial_guess > 1.5: initial_guess = 0.2 #
            iv_result_dict = implied_volatility( #
                market_price=market_price, S=spot_price, K=K, T=T, r=r_t, #
                initial_sigma=initial_guess, tolerance=iv_tolerance, max_iterations=iv_max_iterations #
            ) #
            iv_results_list.append(iv_result_dict) #
        except Exception as iv_calc_e: #
            print(f"Warning: IV calculation failed for row {index}: {iv_calc_e}") #
            iv_results_list.append({'implied_volatility': None, 'iterations': 0, 'final_diff': np.nan, 'status': 'Calculation Error (Outer)'}) #

    calculation_time = time.time() - calculation_start_time # <<< FIX: Uses time module now

    iv_details_df = pd.DataFrame(iv_results_list, index=analysis_data.index) #
    analysis_data = pd.concat([analysis_data, iv_details_df], axis=1) #

    # Filter based on IV calculation success
    # (Unchanged)
    initial_rows_iv = len(analysis_data) #
    success_mask = (analysis_data['status'] == 'Converged') & (analysis_data['implied_volatility'].notna()) & (analysis_data['implied_volatility'] > 1e-9) #
    analysis_data = analysis_data[success_mask].copy() #
    rows_after_iv_filter = len(analysis_data) #
    if analysis_data.empty: #
        print("Analysis failed: No options with successfully calculated positive IV.") #
        if initial_rows_iv > 0: convergence_metrics['iv_success_rate'] = rows_after_iv_filter / initial_rows_iv #
        return None, convergence_metrics #

    # Aggregate IV Convergence Metrics
    # (Unchanged)
    if 'iterations' in analysis_data.columns: #
        valid_iv_iters = analysis_data.loc[analysis_data['iterations'] > 0, 'iterations'].dropna() #
        if not valid_iv_iters.empty: convergence_metrics['avg_iv_iterations'] = valid_iv_iters.mean() #
    convergence_metrics['iv_success_rate'] = rows_after_iv_filter / initial_rows_iv if initial_rows_iv > 0 else 0.0 #

    # Rename calculated IV to 'impliedVolatility' for spline fitting
    # (Unchanged)
    analysis_data.rename(columns={'implied_volatility': 'impliedVolatility'}, inplace=True) #

    # Step 2: Create Spline Representation
    # (Unchanged)
    spline_result = create_spline_representation( #
        analysis_data, **spline_structure_params, **solver_lambda_params #
    ) #
    spline_object = None; spline_convergence_info = {} #
    if isinstance(spline_result, tuple) and len(spline_result) == 2: #
        spline_object, spline_convergence_info = spline_result #
        if not isinstance(spline_convergence_info, dict): spline_convergence_info = {} #
    elif spline_result is not None: spline_object = spline_result; print("Warning: create_spline_representation did not return convergence info.") #
    if spline_object is None: print("Analysis failed: Spline representation creation failed."); return analysis_data, convergence_metrics #
    convergence_metrics['spline_solver_iterations'] = spline_convergence_info.get('lsqr_iterations', np.nan) #
    convergence_metrics['spline_solver_status'] = spline_convergence_info.get('lsqr_status', np.nan) #


    # Step 3: Calculate Interpolated IVs
    # (Unchanged logic and debug prints)
    strikes_to_eval = analysis_data['strike'].values #
    maturities_to_eval = analysis_data['time_to_maturity'].values #
    interpolated_ivs = None #
    try: #
        interpolated_ivs = evaluate_bspline_surface(strikes_to_eval, maturities_to_eval, spline_object) #
        if interpolated_ivs is None: raise ValueError("Evaluation returned None") #
        if hasattr(interpolated_ivs, 'shape') and interpolated_ivs.shape[0] != len(analysis_data.index): #
             print(f"  DEBUG: Potential length mismatch! Index len {len(analysis_data.index)}, IVs len {interpolated_ivs.shape[0]}") #
        analysis_data['interpolated_iv'] = interpolated_ivs #
    except Exception as e: #
        print(f"Error during spline evaluation: {e}"); traceback.print_exc(limit=1) #
        analysis_data['interpolated_iv'] = np.nan #
    rows_before_iv_drop = len(analysis_data) #
    analysis_data.dropna(subset=['interpolated_iv'], inplace=True) #
    rows_after_iv_drop = len(analysis_data) #
    if rows_before_iv_drop > rows_after_iv_drop: print(f"  DEBUG: Dropped {rows_before_iv_drop - rows_after_iv_drop} rows due to NaN interpolated_iv.") #
    if analysis_data.empty: print("Analysis failed: No valid interpolated IVs after dropna."); return None, convergence_metrics #

    # Step 4: Calculate Model Prices
    # (Unchanged logic and debug prints)
    model_prices = [] #
    required_bs_cols = ['time_to_maturity', 'strike', 'interpolated_iv'] #
    if not all(col in analysis_data.columns for col in required_bs_cols): print(f"Error: Missing columns required for Black-Scholes calculation: {required_bs_cols}"); return analysis_data, convergence_metrics #
    for index, row in analysis_data.iterrows(): #
        T = row['time_to_maturity']; K = row['strike']; model_iv = row['interpolated_iv'] #
        try: #
            r_t = rate_interpolator(T) #
            if not np.isfinite(r_t) or r_t < -0.05: raise ValueError(f"Invalid rate {r_t:.4f}") #
            model_iv = max(model_iv, 1e-6) #
            model_price = black_scholes_call(spot_price, K, T, r_t, model_iv) #
            scalar_price = float(model_price) if np.isfinite(model_price) and model_price >= 0 else np.nan #
            model_prices.append(scalar_price) #
        except ValueError as ve: model_prices.append(np.nan) #
        except Exception as e: print(f"Warning: Error calculating model price for row {index}: {e}"); model_prices.append(np.nan) #
    if len(model_prices) != len(analysis_data.index): print(f"  DEBUG: Potential length mismatch! Index len {len(analysis_data.index)}, model_prices len {len(model_prices)}"); raise ValueError(f"Model prices length ({len(model_prices)}) != index length ({len(analysis_data.index)})") #
    analysis_data['model_price'] = model_prices #
    rows_before_price_drop = len(analysis_data) #
    analysis_data.dropna(subset=['model_price'], inplace=True) #
    rows_after_price_drop = len(analysis_data) #
    if rows_before_price_drop > rows_after_price_drop: print(f"  DEBUG: Dropped {rows_before_price_drop - rows_after_price_drop} rows due to NaN model_price.") #
    if analysis_data.empty: print("Analysis failed: No valid model prices calculated after dropna."); return analysis_data, convergence_metrics #

    # Step 5: Calculate Errors
    # (Unchanged logic and debug prints, compares to 'market_iv_original')
    if 'market_iv_original' not in analysis_data.columns: #
         print("Error: 'market_iv_original' column missing for final error calculation.") #
         analysis_data['iv_error'] = np.nan; analysis_data['iv_abs_error'] = np.nan #
    else: analysis_data['market_iv_original'] = pd.to_numeric(analysis_data['market_iv_original'], errors='coerce') #
    numeric_cols = ['market_mid_price', 'model_price', 'market_iv_original', 'interpolated_iv'] #
    cols_present_final = [c for c in numeric_cols if c in analysis_data.columns] #
    analysis_data.dropna(subset=cols_present_final, inplace=True) #
    if analysis_data.empty: #
        print("Analysis failed: No valid numeric data remaining for final error calculation.") #
        if 'market_iv_original' not in analysis_data.columns: return analysis_data, convergence_metrics #
        else: return None, convergence_metrics #
    try: #
        if 'market_iv_original' in analysis_data.columns: #
            iv_error_vals = analysis_data['interpolated_iv'] - analysis_data['market_iv_original'] #
            if len(iv_error_vals) != len(analysis_data.index): raise ValueError("iv_error length mismatch") #
            analysis_data['iv_error'] = iv_error_vals #
            analysis_data['iv_abs_error'] = np.abs(analysis_data['iv_error']) #
        price_error_vals = analysis_data['market_mid_price'] - analysis_data['model_price'] #
        if len(price_error_vals) != len(analysis_data.index): raise ValueError("price_error length mismatch") #
        analysis_data['price_error'] = price_error_vals #
        analysis_data['price_abs_error'] = np.abs(analysis_data['price_error']) #
    except Exception as e: #
         print(f"Error during final error calculation/assignment: {e}"); traceback.print_exc(limit=1) #
         return analysis_data, convergence_metrics #

    return analysis_data, convergence_metrics #


# --- Visualization Function ---
# (visualize_errors remains unchanged)
# ...

# --- Placeholder Helper Plotting Functions ---
# (Placeholders remain unchanged)
# ...

# --- Main Execution Block ---
# (Main block remains unchanged)
# ...
# --- Visualization Function ---
# (visualize_errors remains unchanged)
def visualize_errors(error_df, ticker="Unknown", plot_title=None):
    # ... (function body unchanged) ...
    if error_df is None or error_df.empty: return None #
    fig=None #
    try: #
        fig=plt.figure(figsize=(15,10)); numeric_cols=['iv_error','actual_iv','interpolated_iv','time_to_maturity','iv_abs_error','strike','price_error','price_abs_error','market_mid_price','model_price']; error_df_vis=error_df.copy() #
        for col in numeric_cols: #
            if col in error_df_vis.columns: error_df_vis[col]=pd.to_numeric(error_df_vis[col],errors='coerce') #
        error_df_vis.dropna(subset=[c for c in ['iv_error','actual_iv','interpolated_iv','time_to_maturity','iv_abs_error','strike'] if c in error_df_vis.columns], inplace=True) #
        if error_df_vis.empty: print("Skipping visualization: No valid numeric data for core IV plots."); plt.close(fig); return None #

        ax1=fig.add_subplot(2,2,1); sns.histplot(error_df_vis['iv_error'],kde=True,bins=30,ax=ax1); ax1.set_title(f'{ticker} - IV Error Distribution'); ax1.set_xlabel('IV Error (Interpolated - Actual)'); ax1.set_ylabel('Frequency') #

        ax2=fig.add_subplot(2,2,2); actual_valid=error_df_vis['actual_iv']; interp_valid=error_df_vis['interpolated_iv']; #
        max_val=max(actual_valid.max(),interp_valid.max())*1.1; min_val=max(0,min(actual_valid.min(),interp_valid.min())*0.9); min_val=min(min_val,max_val-1e-6) #
        ax2.scatter(error_df_vis['actual_iv'],error_df_vis['interpolated_iv'],alpha=0.5,s=10); ax2.plot([min_val,max_val],[min_val,max_val],'r--',label='y=x'); ax2.set_xlabel('Actual Market IV'); ax2.set_ylabel('Interpolated IV'); ax2.set_title(f'{ticker} - Actual vs. Interpolated IV'); #
        try: ax2.set_xlim(min_val,max_val); ax2.set_ylim(min_val,max_val) #
        except ValueError: print("Warn: Could not set plot limits for IV scatter.") #
        ax2.legend(); ax2.grid(True) #

        ax3=fig.add_subplot(2,2,3); ax3.scatter(error_df_vis['time_to_maturity'],error_df_vis['iv_abs_error'],alpha=0.5,s=10); ax3.set_xlabel('Time to Maturity (Years)'); ax3.set_ylabel('Absolute IV Error'); ax3.set_title(f'{ticker} - Error vs. Maturity'); #
        try: #
            if(error_df_vis['iv_abs_error'].dropna() > 1e-9).any(): ax3.set_yscale('log') #
        except ValueError: print("Warn: Could not set log scale for IV Error vs TTM.") #
        ax3.grid(True,which="both",ls="-",alpha=0.5) #

        ax4=fig.add_subplot(2,2,4) #
        if 'price_abs_error' in error_df_vis.columns and not error_df_vis['price_abs_error'].isnull().all(): #
            ax4.scatter(error_df_vis['time_to_maturity'],error_df_vis['price_abs_error'],alpha=0.5,s=10); ax4.set_xlabel('Time to Maturity (Years)'); ax4.set_ylabel('Absolute Price Error ($)'); ax4.set_title(f'{ticker} - Price Error vs. Maturity'); #
            try: #
                if(error_df_vis['price_abs_error'].dropna() > 1e-9).any(): ax4.set_yscale('log') #
            except ValueError: print("Warn: Could not set log scale for Price Error vs TTM.") #
            ax4.grid(True,which="both",ls="-",alpha=0.5) #
        else: #
             ax4.text(0.5, 0.5, 'Price Error Data N/A', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes) #
             ax4.set_title(f'{ticker} - Price Error vs. Maturity') #

        if plot_title: fig.suptitle(plot_title,fontsize=16); plt.tight_layout(rect=[0,0.03,1,0.95]) #
        else: plt.tight_layout() #
        # plt.show() # REMOVED - Figure is returned
        return fig #
    except Exception as vis_e: #
        print(f"ERROR during visualization: {vis_e}"); traceback.print_exc(); #
        if fig: plt.close(fig); # Ensure figure is closed on error #
        return None #


# --- Placeholder Helper Plotting Functions ---
# (Placeholders remain unchanged)
# ...

# --- Main Execution Block ---
# (Main block remains unchanged)
# ...