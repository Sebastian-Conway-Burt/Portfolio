# src/implied_volatility.py (Updated return type and documentation)

import sys
import os
import numpy as np
from scipy.stats import norm
import traceback # Added for detailed error logging if needed

# Get the current working directory
current_dir = os.getcwd()
project_root = current_dir # Assuming run from Project directory

# Add the project root to the Python path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Assuming black_scholes.py is in the same directory or accessible via path
try:
    from src.black_scholes import black_scholes_call # Updated import path
except ImportError:
     # Fallback if running directly from src directory
    try:
        from black_scholes import black_scholes_call
    except ImportError:
        print("Error: black_scholes module not found. Ensure black_scholes.py is accessible.")
        sys.exit(1)


def implied_volatility(market_price, S, K, T, r, initial_sigma=0.2, tolerance=1e-8, max_iterations=5000, vega_threshold=1e-8):
    """
    Calculates the implied volatility of a European call option using the Newton-Raphson method.

    The Newton-Raphson method iteratively finds the root (sigma) of the function:
        f(sigma) = BlackScholes(S, K, T, r, sigma) - market_price = 0
    The update rule is:
        sigma_next = sigma - f(sigma) / f'(sigma)
    where f'(sigma) is the Vega of the option.

    Convergence Properties:
    - Under ideal conditions (smooth function, good initial guess, non-zero derivative near root),
      Newton-Raphson exhibits quadratic convergence, meaning the number of correct digits
      roughly doubles with each iteration.

    Potential Issues & Failure Modes:
    - Poor Initial Guess: If the initial sigma is too far from the true value, the method might
      diverge or converge to an incorrect root (though less common for BS IV).
    - Vega Near Zero: If Vega (f'(sigma)) is very small, the update step (division by Vega)
      becomes numerically unstable or leads to large jumps in sigma. This often happens for
      deep in-the-money or deep out-of-the-money options, especially with short TTM.
      The `vega_threshold` parameter prevents division by near-zero Vega.
    - Non-Existence/Multiple Roots: While typically a unique positive IV exists for valid market
      prices, arbitrage violations or extreme inputs could theoretically lead to issues.
    - Max Iterations: If convergence within the specified `tolerance` is not achieved within
      `max_iterations`, the process stops.

    Args:
        market_price (float): The market price of the option.
        S (float): Current price of the underlying asset.
        K (float): Strike price of the option.
        T (float): Time to maturity (in years). Must be positive.
        r (float): Risk-free interest rate (as a decimal).
        initial_sigma (float, optional): Initial guess for volatility. Defaults to 0.2.
        tolerance (float, optional): The absolute difference between the model price and market
                                     price required to stop (convergence criterion). Defaults to 1e-7.
        max_iterations (int, optional): Maximum number of iterations allowed. Defaults to 5000.
        vega_threshold (float, optional): Minimum absolute Vega allowed for the Newton-Raphson step.
                                         Prevents division by near-zero values. Defaults to 1e-8.

    Returns:
        dict: A dictionary containing:
            - 'implied_volatility' (float or None): The calculated IV, or None if calculation failed.
            - 'iterations' (int): The number of iterations performed.
            - 'final_diff' (float): The final absolute difference between market price and model price. NaN if calculation failed early.
            - 'status' (str): The reason for termination ('Converged', 'Max Iterations Reached',
                              'Vega Too Small', 'Calculation Error', 'Non-Positive TTM', 'Non-Positive Sigma').
    """
    sigma = initial_sigma
    diff = np.nan # Initialize diff

    # --- Input Validation ---
    if T <= 1e-9: # Check for non-positive time to maturity
         return {
            'implied_volatility': None, 'iterations': 0, 'final_diff': np.nan,
            'status': 'Non-Positive TTM'
        }
    if market_price <= 0: # Market price must be positive
        return {
            'implied_volatility': None, 'iterations': 0, 'final_diff': np.nan,
            'status': 'Non-Positive Market Price' # Added status
        }


    for i in range(max_iterations):
        try:
            price = black_scholes_call(S, K, T, r, sigma)
            # Recalculate d1 here as sigma changes
            # Handle potential sigma=0 or T=0 during d1 calculation
            if abs(sigma * np.sqrt(T)) < 1e-15:
                 # If sigma*sqrt(T) is zero, Vega calculation will fail
                 raise ValueError("sigma * sqrt(T) too close to zero for Vega calculation")
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            vega = S * np.sqrt(T) * norm.pdf(d1)

        except (ValueError, OverflowError, ZeroDivisionError) as e:
             # Catch potential math errors during BS calculation or Vega calc
             # print(f"Warning: Math error during BS/Vega calc for S={S}, K={K}, T={T:.4f}, sigma={sigma:.4f}. Error: {e}") # Optional Debug
             return {
                'implied_volatility': None, 'iterations': i + 1, 'final_diff': np.nan,
                'status': 'Calculation Error'
            }

        diff = market_price - price # Difference f(sigma)

        # Check for convergence
        if abs(diff) < tolerance:
            # Check if the resulting sigma is reasonably positive
            if sigma > 1e-9: # Use a small threshold > 0
                 return {
                    'implied_volatility': sigma, 'iterations': i + 1, 'final_diff': abs(diff),
                    'status': 'Converged'
                 }
            else:
                 # Converged to a non-positive or near-zero sigma
                 return {
                    'implied_volatility': None, 'iterations': i + 1, 'final_diff': abs(diff),
                    'status': 'Non-Positive Sigma'
                 }

        # --- VEGA CHECK ---
        if abs(vega) < vega_threshold:
             # print(f"Warning: Vega ({vega:.2e}) too small for S={S}, K={K}, T={T:.4f}. Sigma={sigma:.4f}. Stopping iteration.") # Optional warning
             return {
                'implied_volatility': None, 'iterations': i + 1, 'final_diff': abs(diff),
                'status': 'Vega Too Small'
             }
        # --- END VEGA CHECK ---

        # Perform the Newton-Raphson update step
        sigma_change = diff / vega
        sigma = sigma + sigma_change

        # --- Optional: Add bounds or checks on sigma ---
        if sigma <= 1e-9: # If sigma goes non-positive or too small during iteration
             # print(f"Warning: Sigma became non-positive ({sigma:.2e}) during iteration {i+1}. Resetting or stopping.") # Optional
             # Option 1: Stop immediately
             return {
                 'implied_volatility': None, 'iterations': i + 1, 'final_diff': abs(diff),
                 'status': 'Non-Positive Sigma'
             }
             # Option 2: Try resetting (might oscillate or fail later)
             # sigma = initial_sigma / (2**(i%3+1)) # Example reset strategy
             # if sigma <= 1e-9: # If even reset fails, give up
             #    return {'implied_volatility': None, 'iterations': i + 1, 'final_diff': abs(diff), 'status': 'Non-Positive Sigma'}

        # Example upper bound check (optional)
        # if sigma > 10.0: # If volatility goes unreasonably high
        #     return {'implied_volatility': None, 'iterations': i + 1, 'final_diff': abs(diff), 'status': 'Sigma Too High'}

    # If the loop finishes without converging
    # print(f"Warning: Implied volatility did not converge after {max_iterations} iterations for S={S}, K={K}, T={T:.4f}.") # Optional
    return {
        'implied_volatility': None, # Return None for IV as it didn't converge
        'iterations': max_iterations,
        'final_diff': abs(diff), # Return the last calculated difference
        'status': 'Max Iterations Reached'
    }


if __name__ == '__main__':
    # Example usage:
    market_price = 10.5
    S = 100
    K = 100
    T = 1.0 # Ensure T > 0
    r = 0.05

    print("--- Testing Standard Case ---")
    iv_result = implied_volatility(market_price, S, K, T, r)

    print("Result Dictionary:")
    for key, value in iv_result.items():
        if isinstance(value, float):
             print(f"  {key}: {value:.6f}")
        else:
             print(f"  {key}: {value}")

    if iv_result['status'] == 'Converged':
        print(f"\nImplied volatility found: {iv_result['implied_volatility']:.6f}")
    else:
        print(f"\nImplied volatility calculation failed or did not converge. Status: {iv_result['status']}")

    # Example of low Vega scenario (Deep OTM, Short TTM)
    market_price_otm = 0.01
    S_otm = 100
    K_otm = 150
    T_otm = 0.02 # Very short TTM
    r_otm = 0.05
    print(f"\n--- Testing Potentially Low Vega Case (S={S_otm}, K={K_otm}, T={T_otm}) ---")
    iv_result_otm = implied_volatility(market_price_otm, S_otm, K_otm, T_otm, r_otm)

    print("Result Dictionary (Low Vega Case):")
    for key, value in iv_result_otm.items():
         if isinstance(value, float):
             print(f"  {key}: {value:.6f}")
         else:
             print(f"  {key}: {value}")

    if iv_result_otm['status'] == 'Converged':
        print(f"\nImplied volatility found: {iv_result_otm['implied_volatility']:.6f}")
    else:
        print(f"\nImplied volatility calculation failed or did not converge. Status: {iv_result_otm['status']}")

    # Example of Max Iterations
    print(f"\n--- Testing Max Iterations Case (using low max_iterations) ---")
    # Use parameters likely to converge, but set max_iterations low
    iv_result_maxiter = implied_volatility(market_price, S, K, T, r, max_iterations=3)

    print("Result Dictionary (Max Iterations Case):")
    for key, value in iv_result_maxiter.items():
         if isinstance(value, float):
             print(f"  {key}: {value:.6f}")
         else:
             print(f"  {key}: {value}")
    print(f"\nStatus: {iv_result_maxiter['status']}") # Should be 'Max Iterations Reached'