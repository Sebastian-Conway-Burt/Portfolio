# src/custom_bspline.py (Ensuring correct solver return)

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
from functools import lru_cache
import warnings
import traceback
import sys

# --- Knot Vector Calculation ---
def calculate_knot_vector(unique_data_values, degree, num_internal_knots,
                          strategy='quantile', min_separation=1e-5):
    # (Code unchanged)
    if not isinstance(unique_data_values, np.ndarray) or unique_data_values.ndim != 1: print("Error: unique_data_values must be a 1D numpy array."); return None
    if len(unique_data_values) < 2: print("Error: Need at least 2 unique data points."); return None
    if degree < 1: print("Error: Degree must be at least 1."); return None
    if num_internal_knots < 0: print("Error: Number of internal knots cannot be negative."); return None
    min_val=unique_data_values[0]; max_val=unique_data_values[-1]; internal_knots=[]
    if num_internal_knots > 0:
        if strategy == 'quantile': quantiles = np.linspace(0, 1, num_internal_knots + 2)[1:-1]; internal_knots = np.quantile(unique_data_values, quantiles, method='linear')
        elif strategy == 'uniform': internal_knots = np.linspace(min_val, max_val, num_internal_knots + 2)[1:-1]
        else: print(f"Warning: Unknown knot strategy '{strategy}'."); num_internal_knots = 0
        if len(internal_knots) > 1:
            internal_knots = np.sort(internal_knots); refined = [internal_knots[0]]
            for i in range(1, len(internal_knots)):
                if internal_knots[i] - refined[-1] >= min_separation: refined.append(internal_knots[i])
            if len(refined) < len(internal_knots): print(f"Warning: Reduced internal knots from {len(internal_knots)} to {len(refined)} due to min_separation={min_separation}")
            internal_knots = np.array(refined); num_internal_knots = len(internal_knots)
    b_start = np.repeat(min_val, degree + 1); b_end = np.repeat(max_val, degree + 1)
    full_knots = np.concatenate([b_start, np.sort(internal_knots), b_end])
    if not np.all(np.diff(full_knots) >= -1e-12): print(f"Error: Knot vector not non-decreasing."); return None
    exp_len = 2*(degree+1)+num_internal_knots;
    if len(full_knots)!=exp_len: print(f"Warning: Knot vector length mismatch ({len(full_knots)} vs {exp_len}).")
    return full_knots

# --- B-Spline Basis Function Evaluation (Cox-de Boor) ---
@lru_cache(maxsize=None)
def cox_de_boor(x, k, i, t):
    # (Code unchanged)
    t=np.asarray(t);
    if k==0: is_last=(i==len(t)-2); in_interval=(t[i]<=x<t[i+1])or(is_last and x==t[i+1]); return 1.0 if in_interval else 0.0
    t1=0.0; d1=t[i+k]-t[i];
    if d1>1e-12: t1=(x-t[i])/d1*cox_de_boor(x,k-1,i,tuple(t))
    t2=0.0; d2=t[i+k+1]-t[i+1];
    if d2>1e-12: t2=(t[i+k+1]-x)/d2*cox_de_boor(x,k-1,i+1,tuple(t))
    return t1+t2

# --- Evaluate Basis Matrix ---
def evaluate_basis_matrix(eval_points, knots, degree):
    # (Code unchanged)
    if not isinstance(eval_points, np.ndarray)or eval_points.ndim!=1: return None
    if not isinstance(knots, np.ndarray)or knots.ndim!=1: return None
    if degree<0: return None
    num_knots=len(knots); num_basis=num_knots-degree-1;
    if num_basis<=0: return None
    mat=sps.lil_matrix((len(eval_points),num_basis),dtype=float); kt=tuple(knots);
    for i, x in enumerate(eval_points):
        for j in range(num_basis):
            val=cox_de_boor(x,degree,j,kt);
            if abs(val)>1e-15: mat[i,j]=val
    return mat.tocsc()

# --- Design Matrix Construction ---
def build_design_matrix(x_data, y_data, knots_x, knots_y, degree_x, degree_y):
    # (Code unchanged)
    if len(x_data)!=len(y_data): print("Error: x/y lengths differ."); return None
    Bx=evaluate_basis_matrix(x_data,knots_x,degree_x);
    if Bx is None: print("Error: Failed X basis matrix."); return None
    By=evaluate_basis_matrix(y_data,knots_y,degree_y);
    if By is None: print("Error: Failed Y basis matrix."); return None
    n_pts=len(x_data); nx=Bx.shape[1]; ny=By.shape[1]; n_coeffs=nx*ny;
    rows,cols,vals=[],[],[]; rx,cx=Bx.nonzero(); ry,cy=By.nonzero();
    vx_dict={(r,c): Bx[r,c] for r,c in zip(rx,cx)}; vy_dict={(r,c): By[r,c] for r,c in zip(ry,cy)};
    proc_rows=set(); common=set(rx)&set(ry);
    for i in common:
        proc_rows.add(i); idx_j=[c for r,c in zip(rx,cx) if r==i]; idx_l=[c for r,c in zip(ry,cy) if r==i];
        for j in idx_j:
            vbx=vx_dict.get((i,j),0.0);
            for l in idx_l:
                vby=vy_dict.get((i,l),0.0);
                if abs(vbx)>1e-15 and abs(vby)>1e-15: c_idx=j*ny+l; rows.append(i); cols.append(c_idx); vals.append(vbx*vby)
    if len(proc_rows)<n_pts: print(f"Warning: {n_pts-len(proc_rows)} data points had no overlapping basis support.") # Moved print here
    try: A=sps.csc_matrix((vals,(rows,cols)),shape=(n_pts,n_coeffs)); return A
    except Exception as e: print(f"Error creating sparse matrix A: {e}"); return None

# --- Helper function to create difference matrix ---
def create_difference_matrix(size, order=2):
    """Creates a sparse difference matrix D of specified order."""
    # (Code unchanged)
    if size < order + 1: print(f"Warning: Cannot create difference matrix order {order} for size {size}."); return None
    if order == 1: D = sps.diags([-1, 1], [0, 1], shape=(size - 1, size), format='csc')
    elif order == 2: D = sps.diags([1, -2, 1], [0, 1, 2], shape=(size - 2, size), format='csc')
    else: raise ValueError("Only order 1 or 2 differences are supported.")
    return D

# --- Penalized Least Squares Solver --- ### <<< ENSURE THIS VERSION IS USED ###
def solve_penalized_bspline_coeffs(A, z_data, num_basis_x, num_basis_y,
                                   lambda_x=0.0, lambda_y=0.0,
                                   penalty_order=2,
                                   lsqr_iter_lim=None, lsqr_tol=1e-8):
    """
    Solves the Anisotropic P-Spline system for B-spline coefficients c.
    Returns: tuple: (coeffs, istop, itn)
    """
    if A is None or z_data is None: return None, None, None
    if A.shape[0] != len(z_data): return None, None, None
    num_coeffs = A.shape[1]
    if num_coeffs != num_basis_x * num_basis_y: return None, None, None

    # Construct Penalty Matrices (Anisotropic)
    penalties = []
    if lambda_x > 1e-12:
        Dx = create_difference_matrix(num_basis_x, order=penalty_order)
        if Dx is not None:
            try: Iy = sps.identity(num_basis_y, format='csc'); Px = sps.kron(Iy, Dx, format='csc'); penalties.append(np.sqrt(lambda_x) * Px)
            except Exception as e: print(f"Warning: Could not create X penalty matrix: {e}")
        else: print(f"Warning: Skipping X penalty (num_basis_x={num_basis_x} too small for order {penalty_order}).")
    if lambda_y > 1e-12:
        Dy = create_difference_matrix(num_basis_y, order=penalty_order)
        if Dy is not None:
            try: Ix = sps.identity(num_basis_x, format='csc'); Py = sps.kron(Dy, Ix, format='csc'); penalties.append(np.sqrt(lambda_y) * Py)
            except Exception as e: print(f"Warning: Could not create Y penalty matrix: {e}")
        else: print(f"Warning: Skipping Y penalty (num_basis_y={num_basis_y} too small for order {penalty_order}).")

    # Build Augmented System
    if penalties:
        try:
            Aug_A = sps.vstack([A] + penalties, format='csc')
            aug_zeros = sum(p.shape[0] for p in penalties)
            Aug_z = np.concatenate([z_data, np.zeros(aug_zeros)])
            A_solve, z_solve = Aug_A, Aug_z
            if lambda_x > 1e-12 or lambda_y > 1e-12: print(f"Solving PENALIZED least squares (lambda_x={lambda_x:.2e}, lambda_y={lambda_y:.2e})")
        except Exception as e: print(f"Error building augmented system: {e}. Falling back to unpenalized."); A_solve, z_solve = A, z_data
    else: A_solve, z_solve = A, z_data

    # Solve using LSQR
    num_vars = A_solve.shape[1]
    default_limit = max(20 * num_vars, 10000)
    effective_iter_lim = lsqr_iter_lim if lsqr_iter_lim is not None else default_limit

    try:
        # Directly unpack the first 3 results from lsqr
        coeffs, istop, itn, _, _, _, _, _, _, _ = spsla.lsqr( # Unpack first 3 needed items explicitly
            A_solve, z_solve, atol=lsqr_tol, btol=lsqr_tol, iter_lim=effective_iter_lim
        )
        # Optional: Print warnings based on istop
        if istop not in [0, 1, 2]:
             status_msgs = {3:"Cond num too large.", 4:"Sol tol not met (iter limit?).", 5:"Cond num too large.", 6:"Sol tol not met.", 7:f"Iter limit ({itn}/{effective_iter_lim}) reached."}
             print(f"Warning: LSQR solver status: {status_msgs.get(istop, f'Code:{istop}')} (iterations: {itn})")
        # --- Ensure 3 values are returned ---
        return coeffs, int(istop), int(itn) # Cast status/iters to int for safety
    except Exception as e:
        print(f"Error solving penalized least squares: {e}"); traceback.print_exc(limit=1, file=sys.stdout)
        # --- Ensure 3 values are returned on error ---
        return None, None, None

# --- Main Fitting Function --- ### <<< ENSURE THIS VERSION IS USED ###
def fit_bspline_surface_penalized(x_data, y_data, z_data, degree_x, degree_y,
                                  knot_params_x, knot_params_y,
                                  lambda_x=0.0, lambda_y=0.0,
                                  penalty_order=2,
                                  lsqr_iter_lim=None, lsqr_tol=1e-8, verbose=False):
    """ Fits the B-Spline surface using Anisotropic P-splines. """
    if verbose: print("\n--- Fitting Anisotropic Penalized B-Spline Surface ---");
    if not (len(x_data) == len(y_data) == len(z_data)): return None # Simplified check
    if verbose: print("Calculating knot vectors...");
    unique_x = np.unique(x_data); unique_y = np.unique(y_data)
    t_x = calculate_knot_vector(unique_x, degree_x, **knot_params_x)
    t_y = calculate_knot_vector(unique_y, degree_y, **knot_params_y)
    if t_x is None or t_y is None: print("Error: Failed to calculate knot vectors."); return None

    num_basis_x = len(t_x) - degree_x - 1; num_basis_y = len(t_y) - degree_y - 1
    can_penalize_x = num_basis_x >= penalty_order + 1; can_penalize_y = num_basis_y >= penalty_order + 1
    actual_lambda_x = lambda_x if can_penalize_x else 0.0; actual_lambda_y = lambda_y if can_penalize_y else 0.0
    if (lambda_x > 0 and not can_penalize_x) or (lambda_y > 0 and not can_penalize_y): print(f"Warning: Basis size too small for penalty order {penalty_order}. Setting lambda to 0.")

    if verbose: print(f"Knot vectors calculated (Basis funcs: X={num_basis_x}, Y={num_basis_y})")
    A = build_design_matrix(x_data, y_data, t_x, t_y, degree_x, degree_y)
    if A is None: print("Error: Failed to build design matrix."); return None

    # --- Call solver and unpack the tuple result (EXPECTING 3 VALUES) ---
    coeffs_flat, lsqr_status, lsqr_iterations = solve_penalized_bspline_coeffs(
        A, z_data, num_basis_x, num_basis_y,
        lambda_x=actual_lambda_x, lambda_y=actual_lambda_y, penalty_order=penalty_order,
        lsqr_iter_lim=lsqr_iter_lim, lsqr_tol=lsqr_tol
    )

    # <<< --- ADDED DEBUG PRINT --- >>>
    if coeffs_flat is None: print("Error: Failed to solve for penalized coefficients."); return None # Check if solver failed

    # Reshape coefficients
    coeffs_2d = None; expected_coeffs = num_basis_x * num_basis_y
    try:
        if len(coeffs_flat) == expected_coeffs: coeffs_2d = coeffs_flat.reshape((num_basis_x, num_basis_y))
        else: print(f"Warning: Coeff length mismatch ({len(coeffs_flat)} vs {expected_coeffs})."); coeffs_2d = coeffs_flat
    except ValueError as reshape_e: print(f"Warning: Coeff reshape failed: {reshape_e}."); coeffs_2d = coeffs_flat

    # --- Create spline object and add convergence info ---
    spline_object = {
        'knots_x': t_x, 'knots_y': t_y, 'coeffs': coeffs_2d,
        'degree_x': degree_x, 'degree_y': degree_y,
        'lsqr_status': lsqr_status, 'lsqr_iterations': lsqr_iterations
    }
    if verbose: print("--- Anisotropic Penalized B-Spline Fitting Complete ---");
    return spline_object

# --- Surface Evaluation ---
def evaluate_bspline_surface(x_eval, y_eval, spline_object):
    """ Evaluates the B-spline surface. """
    # (Code unchanged)
    if spline_object is None: print("Error: Invalid spline_object."); return np.nan
    try: knots_x=spline_object['knots_x']; knots_y=spline_object['knots_y']; coeffs=spline_object['coeffs']; degree_x=spline_object['degree_x']; degree_y=spline_object['degree_y']
    except KeyError as e: print(f"Error: Missing key {e}."); return np.nan
    x_eval=np.atleast_1d(x_eval); y_eval=np.atleast_1d(y_eval); original_shape=x_eval.shape; x_eval_flat=x_eval.ravel(); y_eval_flat=y_eval.ravel()
    if x_eval_flat.shape!=y_eval_flat.shape: print("Error: Mismatched x/y shapes."); return np.full(original_shape, np.nan)
    Bx_eval=evaluate_basis_matrix(x_eval_flat, knots_x, degree_x); By_eval=evaluate_basis_matrix(y_eval_flat, knots_y, degree_y)
    if Bx_eval is None or By_eval is None: print("Error: Basis matrix eval failed."); return np.full(original_shape, np.nan)
    num_basis_x=Bx_eval.shape[1]; num_basis_y=By_eval.shape[1]; coeffs_2d=None
    if coeffs.ndim==1:
        expected_len=num_basis_x*num_basis_y
        if len(coeffs)==expected_len:
            try: coeffs_2d=coeffs.reshape((num_basis_x, num_basis_y))
            except ValueError: print(f"Error: Could not reshape flat coeffs during evaluation."); return np.full(original_shape, np.nan)
        else: print(f"Error: Flat coeff length mismatch during evaluation."); return np.full(original_shape, np.nan)
    elif coeffs.ndim==2:
        if coeffs.shape==(num_basis_x, num_basis_y): coeffs_2d=coeffs
        else: print(f"Error: 2D coeff shape mismatch during evaluation."); return np.full(original_shape, np.nan)
    else: print("Error: Invalid coeff dimensions during evaluation."); return np.full(original_shape, np.nan)
    if coeffs_2d is None: return np.full(original_shape, np.nan)
    try: temp_matrix=Bx_eval@coeffs_2d; z_eval_flat=np.sum(temp_matrix*By_eval.toarray(), axis=1); return z_eval_flat.reshape(original_shape)
    except Exception as e: print(f"Error during final surface eval calculation: {e}"); traceback.print_exc(limit=1, file=sys.stdout); return np.full(original_shape, np.nan)

# --- Example Usage ---
if __name__ == '__main__':
    # (Example usage unchanged)
    np.random.seed(42); num_points = 200; x_data = np.random.rand(num_points) * 100 + 50; y_data = np.random.rand(num_points) * 1.0
    z_true = np.sin(x_data / 20.0) * np.cos(y_data * np.pi) + 0.1 * x_data/100 - 0.2 * y_data
    noise = np.random.randn(num_points) * 0.05; z_data = z_true + noise
    degree_x = 3; degree_y = 3
    knot_params_x = {'num_internal_knots': 15, 'strategy': 'uniform', 'min_separation': 1e-6}
    knot_params_y = {'num_internal_knots': 10, 'strategy': 'uniform', 'min_separation': 1e-6}
    test_lambda_x = 1.0; test_lambda_y = 10.0
    print(f"--- Testing Anisotropic P-Spline (LambdaX={test_lambda_x}, LambdaY={test_lambda_y}) ---")
    spline_object = fit_bspline_surface_penalized(
        x_data, y_data, z_data, degree_x, degree_y, knot_params_x, knot_params_y,
        lambda_x=test_lambda_x, lambda_y=test_lambda_y, verbose=True
    )
    if spline_object is not None:
        print("\nSpline Object Convergence Info:")
        print(f"  LSQR Status Code (istop): {spline_object.get('lsqr_status', 'N/A')}")
        print(f"  LSQR Iterations (itn): {spline_object.get('lsqr_iterations', 'N/A')}")
        x_eval_pts=np.array([75, 100, 125]); y_eval_pts=np.array([0.25, 0.5, 0.75])
        z_evaluated=evaluate_bspline_surface(x_eval_pts, y_eval_pts, spline_object)
        print(f"\nExample evaluated Z values: {z_evaluated}")
        grid_res=30; x_grid=np.linspace(x_data.min(), x_data.max(), grid_res); y_grid=np.linspace(y_data.min(), y_data.max(), grid_res)
        X_grid, Y_grid=np.meshgrid(x_grid, y_grid); x_flat=X_grid.ravel(); y_flat=Y_grid.ravel()
        Z_evaluated_flat=evaluate_bspline_surface(x_flat, y_flat, spline_object)
        if Z_evaluated_flat is not None and not np.all(np.isnan(Z_evaluated_flat)):
             Z_grid = Z_evaluated_flat.reshape(X_grid.shape)
             try:
                 import matplotlib.pyplot as plt; fig=plt.figure(figsize=(10, 7)); ax=fig.add_subplot(111, projection='3d')
                 ax.scatter(x_data, y_data, z_data, c='r', marker='.', label='Data', alpha=0.5)
                 ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', alpha=0.7, edgecolor='none'); ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z"); ax.set_title(f"Anisotropic P-Spline Fit (Lx={test_lambda_x}, Ly={test_lambda_y})"); ax.legend()
                 save_path = f"anisotropic_pspline_test_fit_Lx{test_lambda_x}_Ly{test_lambda_y}.png"; print(f"Saving test plot to {save_path}"); plt.savefig(save_path); plt.close(fig)
             except ImportError: print("\nMatplotlib not found.")
             except Exception as plot_e: print(f"\nError plotting: {plot_e}")
    else: print("Anisotropic P-Spline fitting failed.")