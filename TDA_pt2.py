"""
TDA processing and analysis functions.
Converts persistent homology data and compares persistence diagrams.
"""

import pandas as pd
import numpy as np
import ast
import warnings
from scipy.stats import median_abs_deviation, entropy, wasserstein_distance
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy as stats_entropy
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

# Save directory for output
save_dir = '/home/ll16598/Documents/POSTDOC/THERAPY/derived_data/'


# =============================================================================
# Helper Functions
# =============================================================================

def _parse_list(cell):
    """
    Convert one cell containing a repr of a list (e.g. '[1.1, 2.3]')
    into a NumPy float array. Robust to NA-like strings or bad input.
    """
    if pd.isna(cell):
        return np.array([], dtype=float)

    if isinstance(cell, (list, tuple, np.ndarray)):
        out = np.asarray(cell, dtype=float)
        return out

    try:
        parsed = ast.literal_eval(str(cell))
        if isinstance(parsed, (list, tuple, np.ndarray)):
            out = np.asarray(parsed, dtype=float)
        else:
            out = np.asarray([parsed], dtype=float)
        return out
    except Exception:
        return np.array([], dtype=float)


def _subtract_arrays(deaths, births):
    """
    Compute deaths - births, element-wise, NaN-aware.
    If arrays differ in length, subtract on overlapping part
    and keep extra elements from longer array.
    """
    len_min = min(len(deaths), len(births))
    if len_min == 0:
        return np.array([], dtype=float)

    out = deaths[:len_min] - births[:len_min]

    if len(deaths) > len_min:
        out = np.concatenate([out, deaths[len_min:]])
    elif len(births) > len_min:
        out = np.concatenate([out, -births[len_min:]])

    return out


def mads_tau(pers_flat, zcut=2.5):
    """Compute tau threshold using MAD (Median Absolute Deviation)."""
    med = np.median(pers_flat)
    mad = median_abs_deviation(pers_flat, scale="normal")
    sig = (pers_flat - med) / mad > zcut
    tau = med + zcut * mad
    prop_sig = sig.mean()
    print('mads kept:', prop_sig)
    return tau


def get_tau_quart(pers_flat, scale=1.5):
    """Compute tau threshold using quartiles."""
    q1, q3 = np.quantile(pers_flat, [0.25, 0.75])
    tau = q3 + scale * (q3 - q1)
    sig = pers_flat > tau
    prop_sig = float(sig.mean())
    print('quart kept:', prop_sig)
    return tau


def percentile_threshold(pers, top_frac=0.10):
    """Keep the top `top_frac` fraction of `pers`."""
    tau = np.quantile(pers, 1.0 - top_frac)
    sig = pers > tau
    prop_sig = float(sig.mean())
    print(f'percentile kept: {prop_sig:.3f}')
    return tau


def pers_vals(pers, Tau):
    """Return the subset of persistence values >= Tau."""
    return [p for p in pers if p >= Tau]


def pers_idx(pers, Tau):
    """Return the *indices* whose values >= Tau."""
    return [i for i, p in enumerate(pers) if p >= Tau]


def subset_array(arr, idx):
    """
    Return arr[idx] or an empty float array if either side is
    missing/empty.
    """
    if not isinstance(arr, np.ndarray) or len(arr) == 0 or len(idx) == 0:
        return np.array([], dtype=float)
    return arr[np.asarray(idx, dtype=int)]


def gini_coefficient(x):
    """Compute the Gini coefficient of a 1D array x."""
    x = np.asarray(x, dtype=float)
    if x.size == 0 or x.mean() == 0:
        return np.nan
    diffs = np.abs(x[:, None] - x[None, :])
    return diffs.sum() / (2 * x.size**2 * x.mean())


def _persistent_entropy(pers):
    """Compute normalized persistent entropy."""
    pers = np.asarray(pers, dtype=float)
    if pers.size == 0 or pers.sum() == 0:
        return 0.0
    p = pers / pers.sum()
    H = -np.sum(p * np.log(p))
    return H / np.log(len(p))


def _coeff_var(arr):
    """Compute coefficient of variation."""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return 0.0
    mean = arr.mean()
    return 0.0 if mean == 0 else arr.std(ddof=0) / mean


def get_rt_features(rt, births, deaths):
    """Count features with birth < rt < death."""
    return len([i for i in range(len(births))
                if births[i] < rt and deaths[i] > rt])


def _metric_block(b, d, p):
    """
    Compute a block of metrics from births, deaths, and persistence values.
    """
    if len(p) == 0:
        keys = ['births', 'deaths', 'birth_rate', 'death_rate', 'sum_persistence',
                'mean_persistence', 'max_persistence', 'std_persistence',
                'number', 'median_persistence', 'iqr', 'cv_persistence',
                'median_birth', 'std_birth', 'median_death', 'std_death', 'gini']
        return dict.fromkeys(keys, np.nan)

    b = np.asarray(b, dtype=float)
    d = np.asarray(d, dtype=float)
    p = np.asarray(p, dtype=float)

    birth_rate = b.mean()
    death_rate = d.mean()
    mean_persistence = p.mean()
    max_persistence = p.max()
    std_persistence = p.std(ddof=1)
    number = len(p)

    median_persistence = np.median(p)
    q25, q75 = np.percentile(p, [25, 75])
    iqr = q75 - q25
    cv = std_persistence / mean_persistence

    median_birth = np.median(b)
    std_birth = b.std(ddof=1)
    median_death = np.median(d)
    std_death = d.std(ddof=1)
    gini = gini_coefficient(p)

    return {
        'births': b.tolist(),
        'deaths': d.tolist(),
        'birth_rate': birth_rate,
        'death_rate': death_rate,
        'sum_persistence': p.sum(),
        'mean_persistence': mean_persistence,
        'max_persistence': max_persistence,
        'std_persistence': std_persistence,
        'number': number,
        'median_persistence': median_persistence,
        'iqr': iqr,
        'cv_persistence': cv,
        'median_birth': median_birth,
        'std_birth': std_birth,
        'median_death': median_death,
        'std_death': std_death,
        'gini': gini
    }


def _to_1d_float_array(cell):
    """
    Convert various input formats to flat 1D float numpy array.
    Handles lists, arrays, nested lists/arrays, etc.
    """
    if cell is None:
        return np.array([], dtype=float)
    if isinstance(cell, np.ndarray):
        return cell.ravel().astype(float)
    try:
        if any(isinstance(x, (list, np.ndarray)) for x in cell):
            parts = []
            for x in cell:
                arr = np.asarray(x, dtype=float)
                parts.append(arr.ravel())
            return np.concatenate(parts) if parts else np.array([], dtype=float)
        else:
            return np.asarray(cell, dtype=float)
    except Exception:
        return np.array([], dtype=float)


def find_match(feature, diag2, tol=1e-6, metric="chebyshev"):
    """
    Find matching point in diag2 for a feature point from diag1.
    
    Parameters:
        feature: (birth, death) tuple
        diag2: array of shape (N, 2)
        tol: matching tolerance
        metric: "chebyshev" or "euclid"
    
    Returns:
        (found, match_idx, dist)
    """
    if diag2.size == 0:
        return False, None, np.nan

    diffs = diag2 - feature
    if metric == "chebyshev":
        dists = np.max(np.abs(diffs), axis=1)
    else:
        dists = np.linalg.norm(diffs, axis=1)

    idx = np.argmin(dists)
    if dists[idx] <= tol:
        return True, idx, float(dists[idx])
    else:
        return False, None, float(dists[idx])


def mean_cross_nn_distance(pd1, pd2, metric="euclidean"):
    """
    Compute mean nearest-neighbor distance between two persistence diagrams.
    """
    if pd1.size == 0 or pd2.size == 0:
        return np.nan

    if metric == "chebyshev":
        p = np.inf
    else:
        p = 2

    tree1, tree2 = cKDTree(pd1), cKDTree(pd2)
    d12, _ = tree1.query(pd2, k=1, p=p)
    d21, _ = tree2.query(pd1, k=1, p=p)

    dists = np.concatenate([d12, d21])
    return float(dists.mean())


def compare_diagrams(df, idx1, idx2, tau_name, dim, tol=1e-6, metric="chebyshev"):
    """
    Compare persistence diagrams between two rows.
    
    Returns:
        prop_match, dist_stats, wdist, centroid_dist, bottleneck_dist, mean_nn
    """
    if not tau_name:
        col_b = f"births_dim{dim}"
        col_d = f"deaths_dim{dim}"
    else:
        col_b = f"{tau_name}_births_dim{dim}"
        col_d = f"{tau_name}_deaths_dim{dim}"

    b1 = _to_1d_float_array(df.loc[idx1, col_b])
    d1 = _to_1d_float_array(df.loc[idx1, col_d])
    pd1 = np.vstack([b1, d1]).T
    b2 = _to_1d_float_array(df.loc[idx2, col_b])
    d2 = _to_1d_float_array(df.loc[idx2, col_d])
    pd2 = np.vstack([b2, d2]).T

    found, dists = [], []
    for pt in pd1:
        ok, _, dist = find_match(tuple(pt), pd2, tol, metric)
        found.append(ok)
        dists.append(dist)

    prop_match = float(np.mean(found)) if found else np.nan

    if dists:
        dist_stats = {
            'mean_dist': float(np.mean(dists)),
            'std_dist': float(np.std(dists, ddof=1)) if len(dists) > 1 else 0.0,
            'entropy_dist': float(entropy(dists) / np.log(len(dists))) if len(dists) > 1 else 0.0,
            'coeff_dist': 0.0 if np.mean(dists) == 0 else np.std(dists, ddof=0) / np.mean(dists),
            'min_dist': float(np.min(dists)),
            'max_dist': float(np.max(dists)),
        }
    else:
        dist_stats = dict.fromkeys(
            ['mean_dist', 'std_dist', 'entropy_dist', 'coeff_dist', 'min_dist', 'max_dist'],
            np.nan
        )

    if pd1.size and pd2.size:
        wdist = WD(pd1, pd2)
    else:
        wdist = np.nan

    if pd1.size and pd2.size:
        c1 = pd1.mean(axis=0)
        c2 = pd2.mean(axis=0)
        centroid_dist = float(np.linalg.norm(c1 - c2))
    else:
        centroid_dist = np.nan

    if pd1.size and pd2.size:
        bottleneck_dist = float(_BNDIST(pd1, pd2))
    else:
        bottleneck_dist = np.nan

    mean_nn = mean_cross_nn_distance(pd1, pd2, metric)

    return prop_match, dist_stats, wdist, centroid_dist, bottleneck_dist, mean_nn


def get_abs_rel_diff(prev_val, curr_val):
    """Compute absolute relative difference."""
    if prev_val != 0:
        return abs((curr_val - prev_val) / abs(prev_val))
    else:
        return float('nan')


def get_spectral_difference(df_spec, idx_prev, idx_curr):
    """Compute spectral differences between two rows."""
    prev_laplace = df_spec.at[idx_prev, f"largest_laplacian_eigenvalue"]
    curr_laplace = df_spec.at[idx_curr, f"largest_laplacian_eigenvalue"]
    prev_fiedler = df_spec.at[idx_prev, f"fiedler_value"]
    curr_fiedler = df_spec.at[idx_curr, f"fiedler_value"]
    prev_sg = df_spec.at[idx_prev, f"spectral_gap"]
    curr_sg = df_spec.at[idx_curr, f"spectral_gap"]

    lap_diffs = [
        get_abs_rel_diff(prev_laplace, curr_laplace),
        get_abs_rel_diff(prev_fiedler, curr_fiedler),
        get_abs_rel_diff(prev_sg, curr_sg)
    ]

    if any(np.isnan(lap_diffs)):
        return float('nan'), float('nan')

    return np.mean(lap_diffs), np.sum(lap_diffs)


def plot_pd_with_tau(df, idx, birth_col, death_col, tau, ax=None):
    """
    Plot persistence diagram with threshold line y = x + tau.
    Points with death <= birth + tau are greyed out.
    """
    b = df.loc[idx, birth_col]
    d = df.loc[idx, death_col]

    if not (isinstance(b, np.ndarray) and isinstance(d, np.ndarray)):
        raise ValueError(f"Row {idx}: expected NumPy arrays in {birth_col}/{death_col}")

    pts = np.vstack([b, d]).T
    pts = pts[~np.isnan(pts).any(axis=1)]
    births, deaths = pts[:, 0], pts[:, 1]

    colors = np.where(deaths <= births + tau, "lightgrey", "C0")

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(births, deaths, c=colors, edgecolor="k", alpha=0.8)

    mn = min(births.min(), deaths.min())
    mx = max(births.max(), deaths.max())
    ax.plot([mn, mx], [mn, mx], "--", color="gray", linewidth=1)

    x_line = np.linspace(mn, mx, 100)
    y_line = x_line + tau
    ax.plot(x_line, y_line, "--", color="red", linewidth=1)

    ax.set_xlim(mn, max(mx, mn + tau))
    ax.set_ylim(mn, max(mx, mn + tau))

    ax.set_aspect("equal", "box")
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title(f"Persistence diagram (row {idx}), τ={tau:.3g}")

    plt.show()


# =============================================================================
# Main Function: Process TDA results
# =============================================================================

def process_tda_results(df_with_tda, MEL=3):
    """
    Process TDA results from CSV file and compute persistence-based features.
    
    Parameters:
        df_texts: Input dataframe (not directly used in this version, 
                  but kept for compatibility)
        dim_reduction: Dimension for reduction (default: 100)
        step_sparse: Step sparsity factor (default: 0.1)
        tuned: Whether tuned parameters are used (default: True)
        MEL: Minimum expected lifetime threshold (default: 2)
        pooling: Pooling strategy suffix (default: '_tunedsplit_02')
        span: Span parameter (default: 110)
    
    Returns:
        DataFrame with all computed TDA features
    """


    # Process dimensions
    for dim in [0, 1]:
        death_col = f"deaths_dim{dim}"
        birth_col = f"births_dim{dim}"
        number_col= f'number_dim{dim}'
        pers_col = f"persistence_dim{dim}"
        entropy_col = f"H_dim{dim}"
        cv_col = f"CV_dim{dim}"
        df_with_tda[death_col] = df_with_tda[death_col].apply(_parse_list)
        df_with_tda[birth_col] = df_with_tda[birth_col].apply(_parse_list)
        df_with_tda[number_col]=df_with_tda[death_col].apply(len)

        # def find_less_ind(births, MEL=2.5):
        #     return [idx for idx, i in enumerate(births) if i < MEL]

        # def filter_indices(births, kept_indices):
        #     return np.array([i for idx, i in enumerate(births) if idx in kept_indices])

        if dim == 0:
            df_with_tda[f'rt_dim{dim}'] = df_with_tda[death_col].apply(max)

        # df_with_tda['kept_indices'] = df_with_tda.apply(
        #     lambda row: find_less_ind(row[birth_col], MEL=MEL),
        #     axis=1
        # )
        # df_with_tda[birth_col] = df_with_tda.apply(
        #     lambda row: filter_indices(row[birth_col], row['kept_indices']),
        #     axis=1
        # )
        # df_with_tda[death_col] = df_with_tda.apply(
        #     lambda row: filter_indices(row[death_col], row['kept_indices']),
        #     axis=1
        #)
        df_with_tda[f'rt_num_dim{dim}'] = df_with_tda.apply(
            lambda row: get_rt_features(row['rt_dim0'], row[birth_col], row[death_col]),
            axis=1
        )
        df_with_tda[pers_col] = [
            _subtract_arrays(d, b) for d, b in zip(df_with_tda[death_col],
                                                   df_with_tda[birth_col])
        ]
        df_with_tda[entropy_col] = df_with_tda[pers_col].apply(_persistent_entropy)
        df_with_tda[cv_col] = df_with_tda[pers_col].apply(_coeff_var)

        pers_flat = np.concatenate(
            [a for a in df_with_tda[pers_col]
             if isinstance(a, np.ndarray) and a.size]
        ).tolist()

        m_tau = mads_tau(pers_flat)
        q_tau = get_tau_quart(pers_flat)
        iqr_tau = percentile_threshold(pers_flat, top_frac=0.10)

        tau_names = ['m_tau', 'q_tau', 'iqr_tau']
        taus = [m_tau, q_tau, iqr_tau]

        for t, tau_name in enumerate(tau_names):
            pers_col_sig = f"{tau_name}_persistence_dim{dim}"
            pers_col_sig_ind = f"{tau_name}_persistence_dim{dim}_indices"
            Tau_to_use = taus[t]

            df_with_tda[pers_col_sig] = df_with_tda[pers_col].apply(
                lambda arr, t=Tau_to_use: pers_vals(arr, t)
            )

            df_with_tda[pers_col_sig_ind] = df_with_tda[pers_col].apply(
                lambda arr, t=Tau_to_use: pers_idx(arr, t)
            )

            col_birth = f"{tau_name}_births_dim{dim}"
            col_death = f"{tau_name}_deaths_dim{dim}"

            df_with_tda[col_birth] = [
                subset_array(b, idx)
                for b, idx in zip(df_with_tda[birth_col], df_with_tda[pers_col_sig_ind])
            ]
            df_with_tda[col_death] = [
                subset_array(d, idx)
                for d, idx in zip(df_with_tda[death_col], df_with_tda[pers_col_sig_ind])
            ]

            def _row_stats(row):
                stats = _metric_block(row[col_birth], row[col_death], row[pers_col_sig])
                return pd.Series({f"{tau_name}_{k}_dim{dim}": v for k, v in stats.items()})

            stats_df = df_with_tda.apply(_row_stats, axis=1)
            df_with_tda = pd.concat([df_with_tda, stats_df], axis=1)

    return df_with_tda



def process_tda_results_local(df_with_tda, MEL=3):
    """
    Process TDA results from CSV file and compute persistence-based features.
    
    Parameters:
        df_texts: Input dataframe (not directly used in this version, 
                  but kept for compatibility)
        dim_reduction: Dimension for reduction (default: 100)
        step_sparse: Step sparsity factor (default: 0.1)
        tuned: Whether tuned parameters are used (default: True)
        MEL: Minimum expected lifetime threshold (default: 2)
        pooling: Pooling strategy suffix (default: '_tunedsplit_02')
        span: Span parameter (default: 110)
    
    Returns:
        DataFrame with all computed TDA features
    """


    # Process dimensions
    for dim in [1, 2]:
        death_col = f"deaths_dim{dim}"
        birth_col = f"births_dim{dim}"
        pers_col = f"persistence_dim{dim}"
        entropy_col = f"H_dim{dim}"
        cv_col = f"CV_dim{dim}"
        def parse_bracket_list(s):
            """Parse space-separated numbers in square brackets, e.g. '[5 6 1]' -> [5, 6, 1]"""
            if pd.isna(s) or s == '[]':
                return np.array([])
            # Remove brackets and split by whitespace
            s = str(s).strip('[]')
            if not s.strip():
                return np.array([])
            return np.array([float(x) for x in s.split()])

        death_col = f"deaths_dim{dim}"
        birth_col = f"births_dim{dim}"
        df_with_tda[death_col] = df_with_tda[death_col].apply(parse_bracket_list)
        df_with_tda[birth_col] = df_with_tda[birth_col].apply(parse_bracket_list)
        print(df_with_tda[death_col])
       # df_with_tda[death_col] = df_with_tda[death_col].apply(_parse_list)
        #df_with_tda[birth_col] = df_with_tda[birth_col].apply(_parse_list)
        # def find_less_ind(births, MEL=2.5):
        #     return [idx for idx, i in enumerate(births) if i < MEL]

        # def filter_indices(births, kept_indices):
        #     return np.array([i for idx, i in enumerate(births) if idx in kept_indices])

        # if dim == 0:
        #     df_with_tda[f'rt_dim{dim}'] = df_with_tda[death_col].apply(max)

        # df_with_tda['kept_indices'] = df_with_tda.apply(
        #     lambda row: find_less_ind(row[birth_col], MEL=MEL),
        #     axis=1
        # )
        # df_with_tda[birth_col] = df_with_tda.apply(
        #     lambda row: filter_indices(row[birth_col], row['kept_indices']),
        #     axis=1
        # )
        # df_with_tda[death_col] = df_with_tda.apply(
        #     lambda row: filter_indices(row[death_col], row['kept_indices']),
        #     axis=1
        #)
        df_with_tda[f'rt_num_dim{dim}'] = df_with_tda.apply(
            lambda row: get_rt_features(row['rt_dim0'], row[birth_col], row[death_col]),
            axis=1
        )
        df_with_tda[pers_col] = [
            _subtract_arrays(d, b) for d, b in zip(df_with_tda[death_col],
                                                   df_with_tda[birth_col])
        ]

        df_with_tda[pers_col] = [
            _subtract_arrays(d, b) for d, b in zip(df_with_tda[death_col],
                                                   df_with_tda[birth_col])
        ]
        df_with_tda[entropy_col] = df_with_tda[pers_col].apply(_persistent_entropy)
        df_with_tda[cv_col] = df_with_tda[pers_col].apply(_coeff_var)

        pers_flat = np.concatenate(
            [a for a in df_with_tda[pers_col]
             if isinstance(a, np.ndarray) and a.size]
        ).tolist()

        m_tau = mads_tau(pers_flat)
        q_tau = get_tau_quart(pers_flat)
        iqr_tau = percentile_threshold(pers_flat, top_frac=0.10)

        tau_names = ['m_tau', 'q_tau', 'iqr_tau']
        taus = [m_tau, q_tau, iqr_tau]

        for t, tau_name in enumerate(tau_names):
            pers_col_sig = f"{tau_name}_persistence_dim{dim}"
            pers_col_sig_ind = f"{tau_name}_persistence_dim{dim}_indices"
            Tau_to_use = taus[t]

            df_with_tda[pers_col_sig] = df_with_tda[pers_col].apply(
                lambda arr, t=Tau_to_use: pers_vals(arr, t)
            )

            df_with_tda[pers_col_sig_ind] = df_with_tda[pers_col].apply(
                lambda arr, t=Tau_to_use: pers_idx(arr, t)
            )

            col_birth = f"{tau_name}_births_dim{dim}"
            col_death = f"{tau_name}_deaths_dim{dim}"

            df_with_tda[col_birth] = [
                subset_array(b, idx)
                for b, idx in zip(df_with_tda[birth_col], df_with_tda[pers_col_sig_ind])
            ]
            df_with_tda[col_death] = [
                subset_array(d, idx)
                for d, idx in zip(df_with_tda[death_col], df_with_tda[pers_col_sig_ind])
            ]

            def _row_stats(row):
                stats = _metric_block(row[col_birth], row[col_death], row[pers_col_sig])
                return pd.Series({f"{tau_name}_{k}_dim{dim}": v for k, v in stats.items()})

            stats_df = df_with_tda.apply(_row_stats, axis=1)
            df_with_tda = pd.concat([df_with_tda, stats_df], axis=1)

    return df_with_tda

def process_tda_results2(df_with_tda, col_pers_thresh='final', MEL=3):
    """
    Process TDA results from CSV file and compute persistence-based features.
    
    Parameters:
        df_texts: Input dataframe (not directly used in this version, 
                  but kept for compatibility)
        dim_reduction: Dimension for reduction (default: 100)
        step_sparse: Step sparsity factor (default: 0.1)
        tuned: Whether tuned parameters are used (default: True)
        MEL: Minimum expected lifetime threshold (default: 2)
        pooling: Pooling strategy suffix (default: '_tunedsplit_02')
        span: Span parameter (default: 110)
    
    Returns:
        DataFrame with all computed TDA features
    """


    # Process dimensions
    for dim in [0, 1, 2]:
        death_col = f"deaths_dim{dim}"
        birth_col = f"births_dim{dim}"
        pers_col = f"persistence_dim{dim}"

        entropy_col = f"H_dim{dim}"
        cv_col = f"CV_dim{dim}"
        df_with_tda[death_col] = df_with_tda[death_col].apply(_parse_list)
        df_with_tda[birth_col] = df_with_tda[birth_col].apply(_parse_list)

        def find_less_ind(births, MEL=2.5):
            return [idx for idx, i in enumerate(births) if i < MEL]

        def filter_indices(births, kept_indices):
            return np.array([i for idx, i in enumerate(births) if idx in kept_indices])

        if dim == 0:
            df_with_tda[f'rt_dim{dim}'] = df_with_tda[death_col].apply(max)

        # df_with_tda['kept_indices'] = df_with_tda.apply(
        #     lambda row: find_less_ind(row[birth_col], MEL=MEL),
        #     axis=1
        # )
        # df_with_tda[birth_col] = df_with_tda.apply(
        #     lambda row: filter_indices(row[birth_col], row['kept_indices']),
        #     axis=1
        # )
        # df_with_tda[death_col] = df_with_tda.apply(
        #     lambda row: filter_indices(row[death_col], row['kept_indices']),
        #     axis=1
        # )
        df_with_tda[f'rt_num_dim{dim}'] = df_with_tda.apply(
            lambda row: get_rt_features(row['rt_dim0'], row[birth_col], row[death_col]),
            axis=1
        )
        df_with_tda[pers_col] = [
            _subtract_arrays(d, b) for d, b in zip(df_with_tda[death_col],
                                                   df_with_tda[birth_col])
        ]
        df_with_tda[entropy_col] = df_with_tda[pers_col].apply(_persistent_entropy)
        df_with_tda[cv_col] = df_with_tda[pers_col].apply(_coeff_var)
        if col_pers_thresh == 'final':
            df_pers=df_with_tda[df_with_tda['last']=='T']
            pers_flat = np.concatenate(
                [a for a in df_pers[pers_col]
                if isinstance(a, np.ndarray) and a.size]
            ).tolist()
        else:
            pers_flat = np.concatenate(
                [a for a in df_with_tda[pers_col]
                if isinstance(a, np.ndarray) and a.size]
            ).tolist()

        m_tau = mads_tau(pers_flat)
        q_tau = get_tau_quart(pers_flat)
        iqr_tau = percentile_threshold(pers_flat, top_frac=0.10)

        tau_names = ['m_tau', 'q_tau', 'iqr_tau']
        taus = [m_tau, q_tau, iqr_tau]

        for t, tau_name in enumerate(tau_names):
            pers_col_sig = f"{tau_name}_persistence_dim{dim}"
            pers_col_sig_ind = f"{tau_name}_persistence_dim{dim}_indices"
            Tau_to_use = taus[t]

            df_with_tda[pers_col_sig] = df_with_tda[pers_col].apply(
                lambda arr, t=Tau_to_use: pers_vals(arr, t)
            )

            df_with_tda[pers_col_sig_ind] = df_with_tda[pers_col].apply(
                lambda arr, t=Tau_to_use: pers_idx(arr, t)
            )

            col_birth = f"{tau_name}_births_dim{dim}"
            col_death = f"{tau_name}_deaths_dim{dim}"

            df_with_tda[col_birth] = [
                subset_array(b, idx)
                for b, idx in zip(df_with_tda[birth_col], df_with_tda[pers_col_sig_ind])
            ]
            df_with_tda[col_death] = [
                subset_array(d, idx)
                for d, idx in zip(df_with_tda[death_col], df_with_tda[pers_col_sig_ind])
            ]

            def _row_stats(row):
                stats = _metric_block(row[col_birth], row[col_death], row[pers_col_sig])
                return pd.Series({f"{tau_name}_{k}_dim{dim}": v for k, v in stats.items()})

            stats_df = df_with_tda.apply(_row_stats, axis=1)
            df_with_tda = pd.concat([df_with_tda, stats_df], axis=1)

    return df_with_tda