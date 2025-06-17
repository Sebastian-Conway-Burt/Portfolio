# src/market_data_utils.py (Updated for short-term flat extrapolation)

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import traceback
# import io # Seems unused
import requests
import xml.etree.ElementTree as ET
import yfinance as yf
import matplotlib.pyplot as plt # Keep for if __name__ == '__main__' plotting

# --- Maturity Mapping (Unchanged) ---
TREASURY_MATURITY_MAP = {
    'BC_1MONTH': 1/12, 'BC_2MONTH': 2/12, 'BC_3MONTH': 3/12,
    'BC_4MONTH': 4/12, 'BC_6MONTH': 6/12, 'BC_1YEAR': 1.0,
    'BC_2YEAR': 2.0, 'BC_3YEAR': 3.0, 'BC_5YEAR': 5.0, 'BC_7YEAR': 7.0,
    'BC_10YEAR': 10.0, 'BC_20YEAR': 20.0, 'BC_30YEAR': 30.0
}

# --- Market Context Functions (Unchanged) ---
def get_spot_price(ticker, analysis_date):
    """Fetches closing spot price for a given date, handling timezones."""
    # print(f"Fetching spot price for {ticker} on or before {analysis_date}...") # Reduced print
    try:
        ticker_obj = yf.Ticker(ticker); start_date = analysis_date - timedelta(days=7); end_date = analysis_date + timedelta(days=1)
        hist = ticker_obj.history(start=start_date, end=end_date, auto_adjust=True) # Use auto_adjust=True maybe?
        if hist.empty: print(f"Warning: No yfinance data for {ticker} {start_date} to {end_date}."); return None
        # Ensure index is timezone-naive or consistent for comparison
        hist.index = hist.index.tz_localize(None)
        analysis_timestamp_naive = pd.Timestamp(analysis_date)

        # Find the latest data point on or before the analysis date
        hist_filtered = hist[hist.index <= analysis_timestamp_naive]
        if hist_filtered.empty:
             # If no data on or before, maybe use the first available point after? Or just fail?
             # Let's try using the last available point from the original history if filtered is empty
             if not hist.empty:
                  spot = hist['Close'].iloc[-1]
                  price_date = hist.index[-1].strftime('%Y-%m-%d')
                  print(f"Warn: No spot price found on or before {analysis_date} for {ticker}. Using last available price ({spot:.2f}) from {price_date}")
             else: return None # No data at all
        else:
            spot = hist_filtered['Close'].iloc[-1]
            price_date = hist_filtered.index[-1].strftime('%Y-%m-%d')
            # print(f"Using Spot Price (S): {spot:.2f} from {price_date}") # Reduced print

        if pd.isna(spot): print(f"Error: Spot price is NaN for {ticker} around {analysis_date}."); return None
        return float(spot)
    except Exception as e: print(f"Error fetching spot price for {ticker}: {e}"); traceback.print_exc(limit=1); return None

def fetch_daily_yield_curve(analysis_date):
    """Fetches the US Treasury par yield curve rates for a specific date."""
    if isinstance(analysis_date, str):
        try: analysis_date = datetime.strptime(analysis_date, "%Y-%m-%d").date()
        except ValueError: print(f"Error: Invalid date string format: {analysis_date}."); return None
    elif not isinstance(analysis_date, date): print("Error: analysis_date must be string or date."); return None

    analysis_date_str_long = analysis_date.strftime("%Y-%m-%d")
    analysis_date_str_req = analysis_date.strftime("%Y%m")
    base_url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml"
    params = {'data': 'daily_treasury_yield_curve', 'field_tdr_date_value_month': analysis_date_str_req}
    # print(f"Attempting fetch yield curve for month {analysis_date_str_req} from Treasury XML...") # Reduced print

    try:
        response = requests.get(base_url, params=params, timeout=20) # Increased timeout slightly
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        xml_content = response.content
        # Quick check if content seems valid XML before parsing
        if not xml_content.strip().startswith(b'<'):
            print(f"Warning: Invalid content received from Treasury URL for {analysis_date_str_req}. Content: {xml_content[:100]}...")
            return None
        root = ET.fromstring(xml_content)
        # Define namespaces - Check these if the Treasury XML structure changes
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'd': 'http://schemas.microsoft.com/ado/2007/08/dataservices',
            'm': 'http://schemas.microsoft.com/ado/2007/08/dataservices/metadata'
        }

        entry_data = None
        # Use namespaces in findall
        for entry in root.findall('.//atom:entry', ns):
            properties = entry.find('.//m:properties', ns)
            if properties is not None:
                date_element = properties.find('d:NEW_DATE', ns)
                if date_element is not None and date_element.text is not None and date_element.text.startswith(analysis_date_str_long):
                    entry_data = properties
                    break # Found the entry for the specific date

        if entry_data is None:
            print(f"Warning: No yield curve data found specifically for date {analysis_date_str_long}.")
            # Add fallback? Maybe try previous day? For now, return None.
            return None

        rates_dict = {}
        for element in entry_data:
            # Extract tag name without namespace
            tag_name = element.tag.split('}')[-1]
            if tag_name in TREASURY_MATURITY_MAP and element.text is not None:
                try:
                    # Convert rate to decimal (e.g., 5.12 -> 0.0512)
                    rate = float(element.text) / 100.0
                    rates_dict[TREASURY_MATURITY_MAP[tag_name]] = rate
                except (ValueError, TypeError):
                    # print(f"Warn: Could not parse rate for {tag_name}: {element.text}") # Optional warning
                    pass # Skip if rate is not a valid number

        if len(rates_dict) < 3: # Need at least 3 points for cubic spline
            print(f"Warning: Found fewer than 3 valid rate points for {analysis_date_str_long}. Cannot create spline.")
            return None

        yield_curve = pd.Series(rates_dict).sort_index()
        # print(f"Successfully fetched yield curve for {analysis_date_str_long}.") # Reduced print
        return yield_curve
    except requests.exceptions.RequestException as e: print(f"Error fetching Treasury data: {e}"); return None
    except ET.ParseError as e: print(f"Error parsing Treasury XML: {e}"); return None
    except Exception as e: print(f"Unexpected error during yield curve fetch: {e}"); traceback.print_exc(limit=1); return None


# --- Custom Cubic Spline Interpolation ---

def _solve_natural_cubic_spline_derivs(x, y):
    """Solves for the second derivatives of a natural cubic spline."""
    n = len(x) - 1
    if n < 2: print("Error: Need at least 3 points for spline derivatives."); return None
    h = np.diff(x)
    # Check for duplicate x values which cause h=0
    if np.any(h <= 1e-12): print("Error: Duplicate x values found in spline input."); return None
    try:
        # Using simplified matrix construction might be clearer
        A = np.zeros((n-1, n-1))
        B = np.zeros(n-1)
        # Diagonal
        np.fill_diagonal(A, 2 * (h[:-1] + h[1:]))
        # Off-diagonals
        np.fill_diagonal(A[1:], h[1:-1])
        np.fill_diagonal(A[:, 1:], h[1:-1])
        # Right-hand side
        B = 6 * (np.diff(y[1:]) / h[1:] - np.diff(y[:-1]) / h[:-1])

        M_internal = np.linalg.solve(A, B)
        M = np.concatenate(([0.0], M_internal, [0.0])) # Natural spline boundary conditions
        return M
    except np.linalg.LinAlgError:
        print("Error: Solving for spline derivatives failed (matrix may be singular)."); return None
    except Exception as e:
        print(f"Error in _solve_natural_cubic_spline_derivs: {e}"); return None

# --- RENAMED and MODIFIED Interpolation Function ---
def custom_cubic_spline_interp1d_with_short_extrap(x_eval, x_known, y_known, M_derivs):
    """
    Evaluates the cubic spline at x_eval points.
    - Uses flat extrapolation for 0 <= x_eval < x_known[0] using y_known[0].
    - Interpolates for x_known[0] <= x_eval <= x_known[-1].
    - Handles cases where consecutive x_known points might be identical.
    - Raises ValueError if x_eval > x_known[-1] or x_eval < 0.
    """
    is_scalar_input = np.isscalar(x_eval)
    x_eval = np.atleast_1d(x_eval).astype(float)
    y_interp = np.full_like(x_eval, np.nan, dtype=float)

    n = len(x_known) - 1
    min_x, max_x = x_known[0], x_known[-1]
    y_min = y_known[0]

    # --- Check Boundaries ---
    if np.any(x_eval < 0):
        raise ValueError(f"Input x_eval contains negative values (minimum allowed is 0).")
    if np.any(x_eval > max_x):
        # Find the first out-of-bounds value for a clearer error message
        first_bad_val = x_eval[x_eval > max_x][0]
        raise ValueError(f"Input x_eval contains value ({first_bad_val:.4f}) greater than the maximum interpolation point ({max_x:.4f}).")

    # --- Identify points for different treatments ---
    extrap_mask = (x_eval >= 0) & (x_eval < min_x)
    interp_mask = (x_eval >= min_x) & (x_eval <= max_x)

    # --- Apply short-end flat extrapolation ---
    if np.any(extrap_mask):
        y_interp[extrap_mask] = y_min

    # --- Apply cubic spline interpolation ---
    if np.any(interp_mask):
        x_interp = x_eval[interp_mask]
        indices = np.searchsorted(x_known, x_interp, side='right') - 1
        indices[x_interp == max_x] = n - 1
        indices = np.clip(indices, 0, n - 1)

        h_i = x_known[indices+1] - x_known[indices]

        # --- FIX: Handle h_i == 0 during coefficient calculation ---
        # Avoid division by zero, default to linear interpolation (or endpoint) if h_i is zero
        # Create masks for safe calculation
        h_is_zero_mask = np.isclose(h_i, 0)
        h_is_not_zero_mask = ~h_is_zero_mask

        # Initialize coefficients safely
        A = np.zeros_like(h_i)
        B = np.zeros_like(h_i)
        C = np.zeros_like(h_i)
        D = np.zeros_like(h_i)

        # Calculate A and B only where h_i is not zero
        if np.any(h_is_not_zero_mask):
            h_i_safe = h_i[h_is_not_zero_mask]
            x_interp_safe = x_interp[h_is_not_zero_mask]
            indices_safe = indices[h_is_not_zero_mask]

            A[h_is_not_zero_mask] = (x_known[indices_safe + 1] - x_interp_safe) / h_i_safe
            B[h_is_not_zero_mask] = (x_interp_safe - x_known[indices_safe]) / h_i_safe
            C[h_is_not_zero_mask] = (1/6) * (A[h_is_not_zero_mask]**3 - A[h_is_not_zero_mask]) * h_i_safe**2
            D[h_is_not_zero_mask] = (1/6) * (B[h_is_not_zero_mask]**3 - B[h_is_not_zero_mask]) * h_i_safe**2

        # Where h_i was zero, effectively B=1, A=0, C=0, D=0 (take value at left known point)
        # This assignment happens implicitly because B,C,D remain 0, and A remains 0
        # But we need to handle the main y_interp calculation

        # Calculate interpolated values using np.where
        y_interp[interp_mask] = np.where(
            h_is_zero_mask,
            y_known[indices], # If h_i is zero, just take the left point's value
            A * y_known[indices] + B * y_known[indices+1] + C * M_derivs[indices] + D * M_derivs[indices+1] # Otherwise use spline formula
        )
        # --- End FIX ---


    if np.any(np.isnan(y_interp)):
        print("Warning: NaN values generated during spline evaluation.")

    return y_interp[0] if is_scalar_input and len(y_interp)==1 else y_interp

# --- MODIFIED Interpolator Creation Function ---
def create_rate_interpolator(yield_curve_data):
    """
    Creates a 1D NATURAL CUBIC SPLINE interpolation function for yield curve data.
    Allows flat extrapolation for TTM between 0 and the shortest yield curve maturity.
    Raises ValueError for TTM > longest maturity or TTM < 0.

    Args:
        yield_curve_data (pd.Series): Index=maturity (float years), values=yield (float decimal).

    Returns:
        callable: A function that takes TTM (float years) and returns the interpolated/extrapolated rate,
                  or None if input data is invalid or spline setup fails.
    """
    if yield_curve_data is None or not isinstance(yield_curve_data, pd.Series) or yield_curve_data.empty or len(yield_curve_data) < 3:
        print("Error: Cannot create cubic spline interpolator. Need at least 3 yield curve points.")
        return None

    try:
        # Ensure index is numeric and values are numeric, handle NaNs
        maturities = pd.to_numeric(yield_curve_data.index, errors='coerce')
        rates = pd.to_numeric(yield_curve_data.values, errors='coerce')
        valid_mask = ~np.isnan(maturities) & ~np.isnan(rates)
        if valid_mask.sum() < 3:
             print("Error: Fewer than 3 valid numeric points after cleaning yield curve data.")
             return None

        maturities_clean = maturities[valid_mask]
        rates_clean = rates[valid_mask]

        # Sort by maturity and remove duplicates (keeping first)
        sort_idx = np.argsort(maturities_clean)
        maturities_sorted = maturities_clean[sort_idx]
        rates_sorted = rates_clean[sort_idx]
        unique_maturities, unique_indices = np.unique(maturities_sorted, return_index=True)

        if len(unique_maturities) < 3:
             print(f"Error: Fewer than 3 unique maturity points after cleaning ({len(unique_maturities)} found). Cannot create spline.")
             return None

        maturities_final = maturities_sorted[unique_indices]
        rates_final = rates_sorted[unique_indices]

        # --- Setup Phase: Solve for second derivatives ---
        # print("Setting up custom cubic spline: solving for second derivatives...") # Reduced print
        M_derivs = _solve_natural_cubic_spline_derivs(maturities_final, rates_final)
        if M_derivs is None:
            print("Error: Cubic spline setup failed (solving derivatives)."); return None
        # print("Cubic spline setup complete.") # Reduced print

        min_maturity = maturities_final[0]
        max_maturity = maturities_final[-1]

        # Return a function that calls the MODIFIED evaluation function
        def interpolator_func(ttm):
            """Evaluates spline with flat extrapolation for 0 <= ttm < min_maturity."""
            try:
                return custom_cubic_spline_interp1d_with_short_extrap(
                    x_eval=ttm,
                    x_known=maturities_final,
                    y_known=rates_final,
                    M_derivs=M_derivs
                )
            except ValueError as ve:
                # Handle expected errors (out of bounds > max_maturity or < 0) gracefully
                print(f"Warning: Rate interpolation failed. {ve}")
                # Return NaN for invalid inputs instead of crashing analysis
                return np.nan if np.isscalar(ttm) else np.full_like(np.atleast_1d(ttm), np.nan)
            except Exception as e:
                # Catch unexpected errors during evaluation
                print(f"Error during rate interpolation evaluation: {e}")
                traceback.print_exc(limit=1)
                return np.nan if np.isscalar(ttm) else np.full_like(np.atleast_1d(ttm), np.nan)

        print(f"Custom rate interpolator created (flat extrap for TTM < {min_maturity:.4f}, range [0, {max_maturity:.4f}]).") # Updated print
        return interpolator_func

    except Exception as e:
        print(f"Error creating custom cubic spline rate interpolator: {e}")
        traceback.print_exc(limit=1)
        return None

# --- Example Usage ---
if __name__ == '__main__':
    test_date = date.today() - timedelta(days=1)
    if test_date.weekday() >= 5: test_date -= timedelta(days=test_date.weekday() - 4)
    test_date_str = test_date.strftime("%Y-%m-%d")
    print(f"--- Market Data Utils Example ({test_date_str}) ---")

    # Fetch real data or use dummy
    yield_curve = fetch_daily_yield_curve(test_date)
    if yield_curve is None:
        print("\nUsing dummy yield curve data as fetch failed.")
        dummy_maturities = np.array([0.083, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
        dummy_rates = np.array([0.050, 0.051, 0.052, 0.053, 0.050, 0.048, 0.047, 0.046, 0.045, 0.042, 0.040])
        yield_curve = pd.Series(dummy_rates, index=dummy_maturities)

    if yield_curve is not None:
        print(f"\nYield Curve Data Points:")
        print(yield_curve.map('{:.5f}'.format))
        min_m, max_m = yield_curve.index.min(), yield_curve.index.max()

        rate_func = create_rate_interpolator(yield_curve)

        if rate_func is not None:
            print("\nInterpolated/Extrapolated Rates:")
            # Test points including short extrapolation, interpolation, and boundaries
            test_ttms = np.array([0.01, 0.05, min_m, 0.25, 0.75, 1.5, 4.0, max_m])

            try:
                interpolated_rates = rate_func(test_ttms)
                for ttm, rate in zip(test_ttms, interpolated_rates):
                     rate_str = f"{rate:.5f} ({rate*100:.3f}%)" if not pd.isna(rate) else "NaN"
                     print(f"  T = {ttm:.3f} years: r = {rate_str}")
            except Exception as eval_e:
                 print(f"Error during example evaluation: {eval_e}")

            # --- Test out-of-bounds behaviour ---
            print("\nTesting out-of-bounds TTM:")
            try:
                # This should now raise ValueError and be caught by the wrapper in create_rate_interpolator
                rate_neg = rate_func(-0.1)
                print(f"  T = -0.1 years: r = {rate_neg if not pd.isna(rate_neg) else 'NaN'} (Error if NaN not printed)") # Should print NaN now
            except Exception as ve: # Catching generic Exception as wrapper might mask ValueError
                print(f"  T = -0.1 years: Correctly handled (returned NaN or raised error): {ve}")
            try:
                # This should also raise ValueError and be caught
                rate_high = rate_func(max_m + 5.0)
                print(f"  T = {max_m + 5.0:.1f} years: r = {rate_high if not pd.isna(rate_high) else 'NaN'} (Error if NaN not printed)") # Should print NaN now
            except Exception as ve:
                print(f"  T = {max_m + 5.0:.1f} years: Correctly handled (returned NaN or raised error): {ve}")

            # --- Optional Plot ---
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(yield_curve.index, yield_curve.values, 'o', label='Data Points', zorder=5)
                # Plot spline including short extrapolation range
                plot_ttms = np.linspace(0, yield_curve.index.max(), 300) # Start from 0
                plot_rates = rate_func(plot_ttms)
                plt.plot(plot_ttms, plot_rates, '-', label='Custom Cubic Spline (Short Extrap.)')
                plt.title('Yield Curve Interpolation (Short TTM Flat Extrapolation)')
                plt.xlabel('Time to Maturity (Years)'); plt.ylabel('Yield (Rate)')
                plt.legend(); plt.grid(True); plt.ylim(bottom=min(0, yield_curve.values.min()*0.9))
                save_path = "yield_curve_interpolation_example.png"
                plt.savefig(save_path)
                plt.close()
                print(f"\nSaved example interpolation plot to {save_path}")
            except ImportError: print("\nMatplotlib not found. Skipping plot.")
            except Exception as plot_e: print(f"\nError plotting: {plot_e}"); plt.close('all')