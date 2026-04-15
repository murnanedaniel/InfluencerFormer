"""
Diagnose why Product Loss (geometric mean Chamfer) fails at large N.

Runs side-by-side optimization at N_pred=7 vs N_pred=20, tracking:
- Gradient magnitudes (min/max/mean/std)
- Gradient direction diversity (cosine similarity between per-pred gradients)
- Loss values
- Assignment stability (which pred is closest to which target)
- Distance matrix statistics

Key hypothesis: The GM = exp(mean(log(D))) dilutes the signal from the
one close prediction among N-1 far predictions. As N grows, the gradient
contribution of the correctly-assigned pred shrinks as 1/N, making it
indistinguishable from noise.
"""

import sys
import os
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

# ------------------------------------------------------------------ #
#  Product Loss (inline, identical to ProductLoss.forward)
# ------------------------------------------------------------------ #

def product_loss(D, eps=1e-8):
    """L = coverage + precision via geometric mean."""
    log_D = torch.log(D + eps)
    coverage = torch.exp(log_D.mean(dim=0))   # GM over preds for each target
    precision = torch.exp(log_D.mean(dim=1))  # GM over targets for each pred
    return coverage.mean() + precision.mean(), coverage, precision


def sinkhorn_loss(D, tau=0.1, n_iters=30):
    """Sinkhorn baseline for comparison."""
    log_alpha = -D / tau
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)
    P = log_alpha.exp()
    return (P * D).sum()


# ------------------------------------------------------------------ #
#  Diagnostic run
# ------------------------------------------------------------------ #

def run_diagnostic(n_pred, n_target=5, n_steps=5000, lr=0.005, seed=42):
    """Run product loss optimization and collect diagnostics."""
    torch.manual_seed(seed)
    targets_real = torch.rand(n_target, 2)
    n_db = n_pred - n_target

    # Pad targets with dustbin at [-1, -1]
    db_pt = torch.tensor([-1.0, -1.0])
    padded_targets = torch.cat([targets_real, db_pt.unsqueeze(0).expand(n_db, -1)], dim=0)

    torch.manual_seed(seed + 100)
    preds = torch.rand(n_pred, 2, requires_grad=True)
    opt = torch.optim.Adam([preds], lr=lr)

    log = {
        "loss": [],
        "grad_mag_min": [], "grad_mag_max": [], "grad_mag_mean": [], "grad_mag_std": [],
        "grad_direction_similarity": [],
        "coverage_terms": [],   # per-target GM values
        "precision_terms": [],  # per-pred GM values
        "assignment_row": [],   # Hungarian assignment at this step
        "assignment_col": [],
        "D_min_per_target": [],  # min distance to each real target
        "D_mean_per_target": [],
        "D_assigned_ratio": [],  # ratio of assigned dist to mean dist per target
        "n_unique_assignments": [],  # how many unique targets are assigned
        "frac_close_to_real": [],  # fraction of preds closer to real targets than dustbin
    }

    snapshot_steps = [0, 500, 1000, 2000, 4999]

    snapshots = {}

    for step in range(n_steps):
        opt.zero_grad()
        D = torch.cdist(preds.unsqueeze(0), padded_targets.unsqueeze(0))[0]
        loss, cov, prec = product_loss(D)
        loss.backward()

        # ---- Collect diagnostics ----
        g = preds.grad.detach().clone()
        per_pred_mag = g.norm(dim=1)  # (n_pred,)

        log["loss"].append(loss.item())
        log["grad_mag_min"].append(per_pred_mag.min().item())
        log["grad_mag_max"].append(per_pred_mag.max().item())
        log["grad_mag_mean"].append(per_pred_mag.mean().item())
        log["grad_mag_std"].append(per_pred_mag.std().item())

        # Gradient direction diversity: mean pairwise cosine similarity
        g_norm = g / (g.norm(dim=1, keepdim=True) + 1e-12)
        cos_sim = (g_norm @ g_norm.T)
        # Upper triangle only (exclude diagonal)
        mask = torch.triu(torch.ones(n_pred, n_pred, dtype=torch.bool), diagonal=1)
        log["grad_direction_similarity"].append(cos_sim[mask].mean().item())

        # Coverage and precision terms
        log["coverage_terms"].append(cov.detach().cpu().numpy().tolist())
        log["precision_terms"].append(prec.detach().cpu().numpy().tolist())

        # Assignment via Hungarian on real targets only
        D_real = D[:, :n_target].detach()
        D_np = D_real.numpy()
        row, col = linear_sum_assignment(D_np)
        log["assignment_row"].append(row.tolist())
        log["assignment_col"].append(col.tolist())

        # Distance stats per real target
        log["D_min_per_target"].append(D_real.min(dim=0).values.numpy().tolist())
        log["D_mean_per_target"].append(D_real.mean(dim=0).numpy().tolist())

        # Ratio: for each target, what fraction of total pred distances does
        # the closest pred account for?
        min_d = D_real.min(dim=0).values
        mean_d = D_real.mean(dim=0)
        log["D_assigned_ratio"].append((min_d / (mean_d + 1e-8)).numpy().tolist())

        # How many unique targets are "claimed" by argmin?
        argmin_per_pred = D_real.argmin(dim=1)
        log["n_unique_assignments"].append(len(argmin_per_pred.unique().tolist()))

        # What fraction of preds are closer to some real target than to dustbin?
        D_db = D[:, n_target:].min(dim=1).values  # dist to nearest dustbin
        D_real_min = D_real.min(dim=1).values       # dist to nearest real target
        frac = (D_real_min < D_db).float().mean().item()
        log["frac_close_to_real"].append(frac)

        # Snapshots
        if step in snapshot_steps:
            snapshots[step] = {
                "preds": preds.detach().clone().numpy(),
                "grad": g.numpy(),
                "D": D.detach().numpy(),
                "per_pred_mag": per_pred_mag.numpy(),
                "coverage": cov.detach().numpy(),
                "precision": prec.detach().numpy(),
            }

        opt.step()

    # Final evaluation
    with torch.no_grad():
        D_final = torch.cdist(preds.unsqueeze(0), targets_real.unsqueeze(0))[0]
        D_db_final = torch.norm(preds - db_pt, dim=1)
        D_real_min_final = D_final.min(dim=1).values
        exist_pred = (D_real_min_final < D_db_final).float().numpy()

        row_f, col_f = linear_sum_assignment(D_final.numpy())
        true_exist = np.zeros(n_pred)
        true_exist[row_f] = 1.0

        tp = ((exist_pred == 1) & (true_exist == 1)).sum()
        fp = ((exist_pred == 1) & (true_exist == 0)).sum()
        fn = ((exist_pred == 0) & (true_exist == 1)).sum()
        prec_f = tp / max(tp + fp, 1)
        rec_f = tp / max(tp + fn, 1)
        f1 = 2 * prec_f * rec_f / max(prec_f + rec_f, 1e-8)

    return log, snapshots, {
        "f1": float(f1), "n_exist": int(exist_pred.sum()),
        "matched_l2": float(D_final.numpy()[row_f, col_f].mean()),
    }


# ------------------------------------------------------------------ #
#  Analysis and reporting
# ------------------------------------------------------------------ #

def analyze_gradient_landscape(snapshots, n_pred, n_target, label):
    """Analyze gradient landscape at snapshot steps."""
    print(f"\n{'='*70}")
    print(f"  GRADIENT LANDSCAPE: {label} (N_pred={n_pred}, N_target={n_target})")
    print(f"{'='*70}")

    for step, snap in sorted(snapshots.items()):
        print(f"\n--- Step {step} ---")
        mags = snap["per_pred_mag"]
        print(f"  Grad magnitude:  min={mags.min():.2e}  max={mags.max():.2e}  "
              f"mean={mags.mean():.2e}  std={mags.std():.2e}  ratio_max/min={mags.max()/(mags.min()+1e-15):.1f}")

        # Coverage analysis
        cov = snap["coverage"]
        print(f"  Coverage GM (per target): {['%.4f' % c for c in cov[:n_target]]}")
        if len(cov) > n_target:
            print(f"  Coverage GM (dustbin):    {['%.4f' % c for c in cov[n_target:n_target+3]]}...")

        # Distance matrix analysis
        D = snap["D"]
        D_real = D[:, :n_target]
        print(f"  D_real: min={D_real.min():.4f}  max={D_real.max():.4f}  mean={D_real.mean():.4f}")

        # Per-target: min distance vs mean distance
        for j in range(n_target):
            col = D_real[:, j]
            min_d = col.min()
            mean_d = col.mean()
            argmin_d = col.argmin()
            # log-mean = log of GM
            log_mean = np.log(col + 1e-8).mean()
            gm = np.exp(log_mean)
            print(f"    Target {j}: min={min_d:.4f} (pred {argmin_d})  "
                  f"mean={mean_d:.4f}  GM={gm:.4f}  "
                  f"min/GM={min_d/(gm+1e-8):.4f}  n_close(<0.1)={int((col<0.1).sum())}")

        # Gradient direction analysis
        g = snap["grad"]
        g_norms = np.linalg.norm(g, axis=1, keepdims=True) + 1e-12
        g_unit = g / g_norms

        # For each pred, check if gradient points toward its nearest real target
        preds = snap["preds"]
        targets_in_padded = D  # we don't have targets directly, use argmin
        nearest_target_idx = D_real.argmin(axis=1)

        # We'll need actual target positions; reconstruct from D is complex.
        # Instead, just report cosine sim stats
        cos_matrix = g_unit @ g_unit.T
        upper = cos_matrix[np.triu_indices(n_pred, k=1)]
        print(f"  Grad direction cos-sim: mean={upper.mean():.4f}  std={upper.std():.4f}")
        print(f"    (1.0 = all parallel, 0.0 = random, -1.0 = opposed)")


def print_summary_table(logs_7, logs_20, final_7, final_20):
    """Print comparative summary."""
    print(f"\n{'='*70}")
    print(f"  COMPARATIVE SUMMARY: N=7 vs N=20")
    print(f"{'='*70}")

    print(f"\n  Final F1:          N=7: {final_7['f1']:.3f}    N=20: {final_20['f1']:.3f}")
    print(f"  Final n_exist:     N=7: {final_7['n_exist']}        N=20: {final_20['n_exist']}")
    print(f"  Matched L2:        N=7: {final_7['matched_l2']:.4f}  N=20: {final_20['matched_l2']:.4f}")

    # Gradient magnitude evolution
    print(f"\n  Gradient magnitude (mean) at key steps:")
    for step_idx, step in enumerate([0, 500, 1000, 2000, 4999]):
        if step_idx < len(logs_7["grad_mag_mean"]):
            g7 = logs_7["grad_mag_mean"][step]
            g20 = logs_20["grad_mag_mean"][step]
            print(f"    Step {step:5d}:  N=7: {g7:.2e}   N=20: {g20:.2e}   ratio: {g7/(g20+1e-15):.2f}")

    # Gradient uniformity (std/mean)
    print(f"\n  Gradient uniformity (std/mean, lower = more uniform/less discriminative):")
    for step in [0, 500, 1000, 2000, 4999]:
        cv7 = logs_7["grad_mag_std"][step] / (logs_7["grad_mag_mean"][step] + 1e-15)
        cv20 = logs_20["grad_mag_std"][step] / (logs_20["grad_mag_mean"][step] + 1e-15)
        print(f"    Step {step:5d}:  N=7: {cv7:.4f}   N=20: {cv20:.4f}")

    # Direction similarity
    print(f"\n  Gradient direction similarity (mean pairwise cosine):")
    for step in [0, 500, 1000, 2000, 4999]:
        print(f"    Step {step:5d}:  N=7: {logs_7['grad_direction_similarity'][step]:.4f}   "
              f"N=20: {logs_20['grad_direction_similarity'][step]:.4f}")

    # Assignment stability
    print(f"\n  Unique targets claimed (out of {5}):")
    for step in [0, 500, 1000, 2000, 4999]:
        print(f"    Step {step:5d}:  N=7: {logs_7['n_unique_assignments'][step]}   "
              f"N=20: {logs_20['n_unique_assignments'][step]}")

    # Fraction close to real targets
    print(f"\n  Fraction of preds closer to real targets than dustbin:")
    for step in [0, 500, 1000, 2000, 4999]:
        print(f"    Step {step:5d}:  N=7: {logs_7['frac_close_to_real'][step]:.3f}   "
              f"N=20: {logs_20['frac_close_to_real'][step]:.3f}")

    # Coverage GM for real targets
    print(f"\n  Mean coverage GM over real targets (lower = better matched):")
    for step in [0, 500, 1000, 2000, 4999]:
        cov7 = np.mean(logs_7["coverage_terms"][step][:5])
        cov20 = np.mean(logs_20["coverage_terms"][step][:5])
        print(f"    Step {step:5d}:  N=7: {cov7:.6f}   N=20: {cov20:.6f}")


def analyze_gradient_math(n_pred_values=[7, 20], n_target=5):
    """Analyze the MATHEMATICAL cause of gradient dilution."""
    print(f"\n{'='*70}")
    print(f"  MATHEMATICAL ANALYSIS: WHY GM GRADIENTS DILUTE WITH N")
    print(f"{'='*70}")

    print("""
    Product Loss coverage for target j:
      L_j = GM_i(D_ij) = exp( (1/N) * sum_i log(D_ij) )

    Gradient for pred k toward target j:
      dL_j/dD_kj = (1/N) * GM_i(D_ij) / D_kj * (dD_kj/dpred_k)

    The critical factor is (1/N) * GM / D_kj:
    - The 1/N factor DIRECTLY dilutes the gradient as N grows
    - The GM is dominated by the MANY far predictions, not the ONE close one
    """)

    # Simulate typical distance distributions
    for N in n_pred_values:
        n_db = N - n_target
        print(f"\n  --- N_pred = {N} (n_dustbin = {n_db}) ---")

        # Typical scenario: 1 pred close to target (d=0.05), rest far (d~0.8)
        d_close = 0.05
        d_far = 0.8
        d_dustbin = 1.5  # distance from real target to dustbin region

        # Build distance column for one target
        dists = np.array([d_close] + [d_far] * (n_target - 1) + [d_dustbin] * n_db)

        log_mean = np.log(dists + 1e-8).mean()
        gm = np.exp(log_mean)

        # Gradient magnitude for the close prediction
        grad_close = (1.0 / N) * gm / d_close

        # Gradient magnitude for a far prediction
        grad_far = (1.0 / N) * gm / d_far

        print(f"    Distances: 1@{d_close}, {n_target-1}@{d_far}, {n_db}@{d_dustbin}")
        print(f"    GM = {gm:.6f}")
        print(f"    Grad for close pred (d={d_close}): {grad_close:.6f}")
        print(f"    Grad for far pred   (d={d_far}):   {grad_far:.6f}")
        print(f"    Ratio close/far grad:              {grad_close/grad_far:.4f}")
        print(f"    1/N factor:                        {1.0/N:.4f}")

        # The actual gradient in 2D would be (1/N) * GM/D_kj * (pred_k - target_j)/D_kj
        # The DIRECTION is correct (toward target) but MAGNITUDE is ~ 1/N * GM/D
        # As N grows: 1/N shrinks linearly, but GM doesn't compensate because
        # log-mean is dominated by log(d_far) and log(d_dustbin)

        # What is the effective "gradient signal to noise ratio"?
        # Signal: gradient pulling close pred toward target
        # Noise: gradients from precision term pulling pred in other directions
        print(f"    Signal (coverage grad for close pred): {grad_close:.6f}")

    # Compare the effective signal
    print(f"\n  CONCLUSION:")
    gm7 = np.exp(np.log(np.array([0.05] + [0.8]*4 + [1.5]*2) + 1e-8).mean())
    gm20 = np.exp(np.log(np.array([0.05] + [0.8]*4 + [1.5]*15) + 1e-8).mean())
    sig7 = (1.0/7) * gm7 / 0.05
    sig20 = (1.0/20) * gm20 / 0.05
    print(f"    Gradient signal at N=7:  {sig7:.6f}")
    print(f"    Gradient signal at N=20: {sig20:.6f}")
    print(f"    Signal ratio (N=7 / N=20): {sig7/sig20:.1f}x")
    print(f"    The signal at N=20 is {sig20/sig7*100:.1f}% of N=7")
    print(f"\n    Two compounding effects:")
    print(f"      1. 1/N dilution: 7/20 = {7/20:.2f}x")
    print(f"      2. GM inflation from dustbin: GM_7={gm7:.4f} vs GM_20={gm20:.4f} "
          f"(ratio={gm20/gm7:.2f}x)")
    print(f"    Combined: the useful gradient is {sig20/sig7*100:.1f}% as strong")
    print(f"    With Adam's adaptive LR this might partially compensate,")
    print(f"    but the DIRECTION becomes increasingly uniform (all preds")
    print(f"    get nearly equal gradient) so no pred is preferentially")
    print(f"    pulled toward its correct target.")


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    print("Product Loss Failure Diagnosis")
    print("=" * 70)

    # Run both configurations
    print("\nRunning N_pred=7, N_target=5 ...")
    logs_7, snaps_7, final_7 = run_diagnostic(n_pred=7, n_target=5, n_steps=5000, seed=42)

    print("Running N_pred=20, N_target=5 ...")
    logs_20, snaps_20, final_20 = run_diagnostic(n_pred=20, n_target=5, n_steps=5000, seed=42)

    # Mathematical analysis
    analyze_gradient_math()

    # Gradient landscape at snapshots
    analyze_gradient_landscape(snaps_7, 7, 5, "Product Loss")
    analyze_gradient_landscape(snaps_20, 20, 5, "Product Loss")

    # Comparative summary
    print_summary_table(logs_7, logs_20, final_7, final_20)

    # ---- CRITICAL: Direction analysis ----
    print(f"\n{'='*70}")
    print(f"  CRITICAL: GRADIENT DIRECTION UNIFORMITY AT N=20")
    print(f"{'='*70}")

    snap = snaps_20[0]  # Step 0
    D = snap["D"]
    D_real = D[:, :5]
    g = snap["grad"]

    print(f"\n  At initialization (step 0), N=20:")
    print(f"  Each target's coverage GM depends on ALL 20 predictions.")
    print(f"  The gradient for pred k from target j is proportional to:")
    print(f"    (1/20) * GM_j / D_kj * direction_kj")
    print(f"\n  Since GM_j involves product of 19 OTHER distances,")
    print(f"  ALL predictions get nearly equal GM factor.")
    print(f"  The ONLY discriminating factor is 1/D_kj, but this is")
    print(f"  summed over ALL targets, creating a ~uniform pull.")

    # Show actual gradient components for a few predictions
    print(f"\n  Grad magnitudes for first 10 preds: "
          f"{[f'{m:.4f}' for m in snap['per_pred_mag'][:10]]}")
    print(f"  Coefficient of variation: "
          f"{snap['per_pred_mag'].std()/snap['per_pred_mag'].mean():.4f}")

    # Compare with N=7
    snap7 = snaps_7[0]
    print(f"\n  Compare N=7:")
    print(f"  Grad magnitudes for all 7 preds: "
          f"{[f'{m:.4f}' for m in snap7['per_pred_mag']]}")
    print(f"  Coefficient of variation: "
          f"{snap7['per_pred_mag'].std()/snap7['per_pred_mag'].mean():.4f}")

    # ---- Run Sinkhorn comparison ----
    print(f"\n{'='*70}")
    print(f"  SINKHORN COMPARISON AT N=20")
    print(f"{'='*70}")

    torch.manual_seed(42)
    targets_real = torch.rand(5, 2)
    db_pt = torch.tensor([-1.0, -1.0])
    padded_targets = torch.cat([targets_real, db_pt.unsqueeze(0).expand(15, -1)], dim=0)

    torch.manual_seed(142)
    preds_sk = torch.rand(20, 2, requires_grad=True)
    D_sk = torch.cdist(preds_sk.unsqueeze(0), padded_targets.unsqueeze(0))[0]
    loss_sk = sinkhorn_loss(D_sk)
    loss_sk.backward()
    g_sk = preds_sk.grad.detach().clone()
    mags_sk = g_sk.norm(dim=1)
    g_sk_unit = g_sk / (g_sk.norm(dim=1, keepdim=True) + 1e-12)
    cos_sk = (g_sk_unit @ g_sk_unit.T)
    upper_sk = cos_sk[torch.triu(torch.ones(20, 20, dtype=torch.bool), diagonal=1)]

    print(f"  Sinkhorn grad magnitudes (first 10): "
          f"{[f'{m:.4f}' for m in mags_sk[:10]]}")
    print(f"  Sinkhorn CV: {mags_sk.std()/mags_sk.mean():.4f}")
    print(f"  Sinkhorn direction cos-sim: mean={upper_sk.mean():.4f}")

    # Product loss at same point
    preds_pl = preds_sk.detach().clone().requires_grad_(True)
    D_pl = torch.cdist(preds_pl.unsqueeze(0), padded_targets.unsqueeze(0))[0]
    loss_pl, _, _ = product_loss(D_pl)
    loss_pl.backward()
    g_pl = preds_pl.grad.detach().clone()
    mags_pl = g_pl.norm(dim=1)
    g_pl_unit = g_pl / (g_pl.norm(dim=1, keepdim=True) + 1e-12)
    cos_pl = (g_pl_unit @ g_pl_unit.T)
    upper_pl = cos_pl[torch.triu(torch.ones(20, 20, dtype=torch.bool), diagonal=1)]

    print(f"\n  Product grad magnitudes (first 10): "
          f"{[f'{m:.4f}' for m in mags_pl[:10]]}")
    print(f"  Product CV: {mags_pl.std()/mags_pl.mean():.4f}")
    print(f"  Product direction cos-sim: mean={upper_pl.mean():.4f}")

    print(f"\n  KEY: Sinkhorn gives DIFFERENTIATED gradients (high CV, low cos-sim)")
    print(f"       Product gives UNIFORM gradients (low CV, high cos-sim)")
    print(f"       Uniform gradients mean no pred is preferentially assigned.")

    # ---- Final diagnosis ----
    print(f"\n{'='*70}")
    print(f"  DIAGNOSIS SUMMARY")
    print(f"{'='*70}")
    print(f"""
    The Product Loss fails at large N due to TWO compounding mechanisms:

    1. GRADIENT MAGNITUDE DILUTION (1/N factor):
       The GM gradient for pred k toward target j contains a 1/N factor
       from the chain rule through mean(log(D)). At N=7 this is 0.143;
       at N=20 this is 0.050 -- a 2.86x reduction.

    2. GRADIENT DIRECTION UNIFORMITY (the killer):
       The GM for target j = exp(mean(log(D_ij))) depends on ALL N preds.
       At large N, the GM is dominated by the (N-1) far predictions,
       making it nearly identical for all targets. The gradient for each
       pred is a sum of terms (1/N)*GM_j/D_kj*(direction to j), where
       GM_j is nearly the same constant for all j.

       This means EVERY prediction gets pulled toward the CENTROID of
       all targets, not toward its specific assigned target. The gradient
       directions become highly correlated (cosine similarity -> 1).

       At N=7, there are only 2 dustbin slots, so the GM still has some
       discriminative power. At N=20, there are 15 dustbin slots, and
       the GM is overwhelmed by the dustbin distances.

    3. DUSTBIN AMPLIFICATION:
       Dustbin targets at [-1,-1] are far from all predictions. Their
       large distances dominate the log-mean, inflating GM_j for real
       targets. This further homogenizes the gradient landscape.

    WHY SINKHORN WORKS:
       Sinkhorn computes a SOFT PERMUTATION MATRIX P via iterative
       normalization. Each pred gets a clear assignment weight to one
       target. The gradient dL/dpred_k = sum_j P_kj * d(D_kj)/dpred_k
       is DOMINATED by the single target j where P_kj is large.
       This gives each pred a UNIQUE direction, enabling parallel
       convergence of all assignments.
    """)


if __name__ == "__main__":
    main()
