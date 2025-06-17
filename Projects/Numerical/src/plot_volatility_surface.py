# src/plot_volatility_surface.py (Corrected variable names in __main__)

import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import traceback

# --- Project Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path: sys.path.insert(0, project_root)
# --- End Path Setup ---

# --- Import project functions ---
try:
    # create_spline_representation now returns a tuple (spline_dict, convergence_dict)
    from src.volatility_surface_interpolation import (
        prepare_volatility_data,
        create_spline_representation
    )
    from src.custom_bspline import evaluate_bspline_surface
except ImportError as e:
    print(f"Import Error in plot_volatility_surface: {e}")
    try: # Fallback
        from volatility_surface_interpolation import prepare_volatility_data, create_spline_representation
        from custom_bspline import evaluate_bspline_surface
    except ImportError:
        print("Ensure volatility_surface_interpolation.py and custom_bspline.py are accessible.")
        sys.exit(1)
# --- End Imports ---

# --- Default Configuration ---
# Defaults specifically for standalone plotting in this file
MAX_MATURITY_YEARS_DEFAULT = 1.0
FILTER_DEFAULTS_PLOT = { # Renamed to avoid conflict
    "min_iv": 0.01, "max_iv": 1.5, "min_volume": 3, "min_open_interest": 0,
    "max_bid_ask_spread_abs": 0.50, "max_bid_ask_spread_rel": 0.30,
    "min_ttm_years": 0.005
}
SPLINE_DEFAULTS_PLOT = { # Renamed to avoid conflict
    "degree_x": 3,
    "degree_y": 3,
    "knot_params_x": {'num_internal_knots': 15, 'strategy': 'uniform', 'min_separation': 1e-6},
    "knot_params_y": {'num_internal_knots': 10, 'strategy': 'uniform', 'min_separation': 1e-6},
    "lambda_x": 0.5,
    "lambda_y": 0.5,
    "penalty_order": 2,
    "lsqr_iter_lim": None,
    "lsqr_tol": 1e-8,
}
# --- End Default Configuration ---

# --- Plotting Function ---
def plot_volatility_surface(data_file_paths,
                            max_maturity_years=MAX_MATURITY_YEARS_DEFAULT,
                            spline_params=None, # Dict containing degrees, knots, lambdas, solver params
                            filter_params=None,
                            z_lim_min=0.0,
                            z_lim_max=1.5,
                            ax=None,
                            plot_title=None
                           ):
    """
    Plots the volatility surface using the Penalized B-Spline implementation.
    Handles the tuple return value from create_spline_representation.
    Uses SPLINE_DEFAULTS_PLOT and FILTER_DEFAULTS_PLOT internally.
    """
    # --- Parameter Handling ---
    # Use the defaults defined *within this file*
    current_spline_params = SPLINE_DEFAULTS_PLOT.copy()
    if spline_params is not None:
        # Nested dict update logic
        if 'knot_params_x' in spline_params and isinstance(spline_params['knot_params_x'], dict):
             current_spline_params['knot_params_x'].update(spline_params.pop('knot_params_x'))
        if 'knot_params_y' in spline_params and isinstance(spline_params['knot_params_y'], dict):
             current_spline_params['knot_params_y'].update(spline_params.pop('knot_params_y'))
        # Update remaining top-level keys
        current_spline_params.update(spline_params)

    current_filter_params = FILTER_DEFAULTS_PLOT.copy()
    if filter_params is not None: current_filter_params.update(filter_params)

    # Separate args for create_spline_representation
    explicit_args = {
        "degree_x": current_spline_params.pop('degree_x'),
        "degree_y": current_spline_params.pop('degree_y'),
        "knot_params_x": current_spline_params.pop('knot_params_x'),
        "knot_params_y": current_spline_params.pop('knot_params_y'),
        "verbose": False # Keep plotting less verbose
    }
    solver_lambda_kwargs = current_spline_params # Remaining are solver/lambda params
    # --- End Parameter Handling ---

    # Step 1: Prepare Data
    grouped_filtered_data = prepare_volatility_data(data_file_paths,
                                                    max_maturity_years=max_maturity_years,
                                                    **current_filter_params)
    if grouped_filtered_data is None or grouped_filtered_data.empty:
        print("Plotting failed: No data after preparation."); return None

    # --- Step 2: Create Spline Representation ---
    spline_result = create_spline_representation(
        grouped_filtered_data,
        **explicit_args,
        **solver_lambda_kwargs
    )

    # Unpack the tuple result
    spline_object_dict = None
    spline_convergence_info = {}
    if isinstance(spline_result, tuple) and len(spline_result) == 2:
        spline_object_dict, spline_convergence_info = spline_result
    elif spline_result is not None:
        spline_object_dict = spline_result
        print("  Warning (Plot): create_spline_representation did not return expected tuple.")

    if spline_object_dict is None:
        print("Plotting failed: Could not create spline representation."); return None
    # --- End Step 2 ---

    # --- Steps 3-6: Extract data, Grid, Evaluate Spline ---
    # (Unchanged logic, passes spline_object_dict)
    maturity_scatter = grouped_filtered_data['time_to_maturity'].values; strike_scatter = grouped_filtered_data['strike'].values;
    iv_scatter = grouped_filtered_data['impliedVolatility'].values if 'impliedVolatility' in grouped_filtered_data else np.full_like(strike_scatter, np.nan)
    unique_strikes = np.sort(grouped_filtered_data['strike'].unique()); unique_ttms = np.sort(grouped_filtered_data['time_to_maturity'].unique())
    if len(unique_strikes)<2 or len(unique_ttms)<2: print("Plotting failed: Insufficient unique points for grid."); return None
    strike_min=unique_strikes.min(); strike_max=unique_strikes.max(); maturity_min=unique_ttms.min(); maturity_max=unique_ttms.max()
    grid_points=50; strike_axis=np.linspace(strike_min,strike_max,grid_points); maturity_axis=np.linspace(maturity_min,maturity_max,grid_points); strike_grid,maturity_grid=np.meshgrid(strike_axis,maturity_axis)
    try:
        x_flat=strike_grid.ravel(); y_flat=maturity_grid.ravel();
        Z_flat=evaluate_bspline_surface(x_flat, y_flat, spline_object_dict) # Use dict
        if Z_flat is None or np.all(np.isnan(Z_flat)): print("Error: Spline evaluation failed."); return None
        interpolated_volatility=Z_flat.reshape(strike_grid.shape); interpolated_volatility=np.maximum(interpolated_volatility,0)
    except Exception as e: print(f"Error during spline evaluation for plot: {e}"); traceback.print_exc(limit=1); return None

    # --- Steps 7-10: Plotting ---
    # (Unchanged logic)
    if ax is None: fig = plt.figure(figsize=(14, 9)); ax_to_plot_on = fig.add_subplot(111, projection='3d')
    else: fig = ax.figure; ax_to_plot_on = ax

    scatter_label = f'Cleaned Data (TTM <= {max_maturity_years:.2f}yr)'
    valid_scatter = ~np.isnan(iv_scatter)
    if np.any(valid_scatter): ax_to_plot_on.scatter(strike_scatter[valid_scatter], maturity_scatter[valid_scatter], iv_scatter[valid_scatter], c='red', marker='o', label=scatter_label, depthshade=True, s=15, alpha=0.8)
    else: print("Warning: No valid original IV data points to scatter.")
    surf = ax_to_plot_on.plot_surface(strike_grid, maturity_grid, interpolated_volatility, cmap='viridis', alpha=0.6, edgecolor='none', rstride=1, cstride=1)
    ax_to_plot_on.set_xlabel('Strike Price'); ax_to_plot_on.set_ylabel('Time to Maturity (Years)'); ax_to_plot_on.set_zlabel('Implied Volatility')
    if plot_title is None:
        dx = explicit_args['degree_x']; dy = explicit_args['degree_y']
        knot_strategy = explicit_args['knot_params_x'].get('strategy', 'unknown')
        nkx = explicit_args['knot_params_x'].get('num_internal_knots', '?')
        nky = explicit_args['knot_params_y'].get('num_internal_knots', '?')
        lambda_x = solver_lambda_kwargs.get('lambda_x', 0.0)
        lambda_y = solver_lambda_kwargs.get('lambda_y', 0.0)
        title_suffix = f"(Deg=({dx},{dy}), Knots=({nkx},{nky}), Strat={knot_strategy}"
        if lambda_x > 1e-9 or lambda_y > 1e-9: title_suffix += f", Lx={lambda_x:.1e}, Ly={lambda_y:.1e}"
        title_suffix += ")"
        plot_title = f'Penalized Volatility Surface {title_suffix}'
    ax_to_plot_on.set_title(plot_title)
    try: fig.colorbar(surf, shrink=0.5, aspect=10, label='Interpolated Implied Volatility', ax=ax_to_plot_on)
    except Exception as cbar_e: print(f"Warning: Could not draw colorbar: {cbar_e}")
    ax_to_plot_on.view_init(elev=20., azim=-125);
    if np.any(valid_scatter): ax_to_plot_on.legend()
    ax_to_plot_on.set_zlim(z_lim_min, z_lim_max)
    if ax is None: plt.tight_layout()
    return fig
# --- End Plotting Function ---

# --- Main Execution Block (for standalone testing) --- ### <<< CORRECTED HERE <<< ###
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__)); project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data', 'raw'); ticker_to_plot = "AAPL"
    print(f"--- Running Plot Volatility Surface Example (Penalized) for {ticker_to_plot} ---")
    try:
        if not os.path.isdir(data_dir): print(f"Error: Data directory not found: {data_dir}")
        else:
             pattern = f"{ticker_to_plot}_call_options_*"; all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv') and f.startswith(pattern)]
             if not all_files: print(f"No '{pattern}*.csv' files found in {data_dir}.")
             else:
                 print(f"Found {len(all_files)} {ticker_to_plot} files.")
                 file_paths = all_files
                 # --- Use the defaults defined within this file ---
                 plot_spline_params = SPLINE_DEFAULTS_PLOT.copy()
                 plot_filter_params = FILTER_DEFAULTS_PLOT.copy()
                 # --- End Use Defaults ---
                 plot_z_min = 0.0; plot_z_max = 1.5

                 fig_object = plot_volatility_surface(
                     file_paths,
                     max_maturity_years=MAX_MATURITY_YEARS_DEFAULT,
                     filter_params=plot_filter_params, # Pass correct filter dict
                     spline_params=plot_spline_params, # Pass correct spline dict
                     z_lim_min=plot_z_min, z_lim_max=plot_z_max
                 )

                 if fig_object:
                     # Use .get() on plot_spline_params which holds the lambda values used
                     lx = plot_spline_params.get('lambda_x',0); ly = plot_spline_params.get('lambda_y',0)
                     save_path = os.path.join(project_root, f"{ticker_to_plot}_penalized_surface_Lx{lx:.1e}_Ly{ly:.1e}.png")
                     print(f"Attempting to save example plot to: {save_path}")
                     fig_object.savefig(save_path);
                     # plt.show() # Uncomment ONLY if running this file directly
                     plt.close(fig_object)
                     print("Example plot saved and closed.")
                 else: print("Plot generation failed.")

    except Exception as e: print(f"\nError in main execution block: {e}"); traceback.print_exc(); plt.close('all')
# --- End Main Block ---