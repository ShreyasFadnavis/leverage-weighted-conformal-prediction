"""Master script: runs all LWCP experiments and saves figures.

Usage:
    python -m experiments.run_all          # run everything
    python -m experiments.run_all --quick  # quick mode (fewer reps)
"""

import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _print_hardware_info():
    """Print hardware and software environment information."""
    import platform
    import numpy as np
    import sklearn
    print("  Hardware / Software Environment:")
    print(f"    Platform:    {platform.platform()}")
    print(f"    Processor:   {platform.processor()}")
    print(f"    Python:      {platform.python_version()}")
    print(f"    NumPy:       {np.__version__}")
    print(f"    scikit-learn: {sklearn.__version__}")
    print()


def main():
    quick = "--quick" in sys.argv

    if quick:
        print("=" * 60)
        print("QUICK MODE: reduced repetitions for fast iteration")
        print("=" * 60)
        reps_coverage = 50
        reps_conditional = 30
        reps_width = 30
        reps_frontier = 30
        reps_gaussian = 30
        reps_hetero = 30
        reps_baselines = 20
        reps_approx = 30
        reps_scaling = 20
        reps_nonlinear = 10
        reps_feature_scaling = 20
        reps_ridge_ext = 20
        reps_weight_sel = 20
        reps_real_data = 10
    else:
        reps_coverage = 1000
        reps_conditional = 200
        reps_width = 200
        reps_frontier = 200
        reps_gaussian = 200
        reps_hetero = 200
        reps_baselines = 100
        reps_approx = 200
        reps_scaling = 100
        reps_nonlinear = 100
        reps_feature_scaling = 100
        reps_ridge_ext = 100
        reps_weight_sel = 50
        reps_real_data = 30

    _print_hardware_info()
    total_start = time.perf_counter()

    # Experiment 1: Marginal Coverage
    print("\n\n" + "#" * 60)
    print("# EXPERIMENT 1: Marginal Coverage")
    print("#" * 60)
    from experiments.run_coverage import run_coverage_experiment
    run_coverage_experiment(n_reps=reps_coverage)

    # Experiment 2: Conditional Coverage
    print("\n\n" + "#" * 60)
    print("# EXPERIMENT 2: Conditional Coverage")
    print("#" * 60)
    from experiments.run_conditional import run_conditional_experiment
    run_conditional_experiment(n_reps=reps_conditional)

    # Experiment 3: Width Comparison
    print("\n\n" + "#" * 60)
    print("# EXPERIMENT 3: Width Comparison")
    print("#" * 60)
    from experiments.run_width import run_width_experiment
    run_width_experiment(n_reps=reps_width)

    # Experiment 4: Efficiency Frontier
    print("\n\n" + "#" * 60)
    print("# EXPERIMENT 4: Efficiency Frontier")
    print("#" * 60)
    from experiments.run_efficiency_frontier import run_efficiency_frontier
    run_efficiency_frontier(n_reps=reps_frontier)

    # Experiment 5: Gaussian Recovery
    print("\n\n" + "#" * 60)
    print("# EXPERIMENT 5: Gaussian Recovery")
    print("#" * 60)
    from experiments.run_gaussian_recovery import run_gaussian_recovery
    run_gaussian_recovery(n_reps=reps_gaussian)

    # Experiment 6: Heteroscedasticity Sweep
    print("\n\n" + "#" * 60)
    print("# EXPERIMENT 6: Heteroscedasticity Sweep")
    print("#" * 60)
    from experiments.run_hetero_sweep import run_hetero_sweep
    run_hetero_sweep(n_reps=reps_hetero)

    # Experiment 7: Baselines (CQR, Studentized)
    print("\n\n" + "#" * 60)
    print("# EXPERIMENT 7: Baselines Comparison")
    print("#" * 60)
    try:
        from experiments.run_baselines import run_baselines_comparison
        run_baselines_comparison(n_reps=reps_baselines)
    except ImportError as e:
        print(f"  SKIPPED (missing dependency): {e}")
        print("  Install with: pip install quantile-forest")

    # Experiment 8: Approximate Leverage
    print("\n\n" + "#" * 60)
    print("# EXPERIMENT 8: Approximate Leverage")
    print("#" * 60)
    from experiments.run_approximate import run_approximate_experiment
    run_approximate_experiment(n_reps=reps_approx)

    # Experiment 9: Scaling
    print("\n\n" + "#" * 60)
    print("# EXPERIMENT 9: Scaling")
    print("#" * 60)
    from experiments.run_scaling import run_scaling_experiment
    run_scaling_experiment(n_reps=reps_scaling)

    # Experiment 10: Non-Linear Predictors
    print("\n\n" + "#" * 60)
    print("# EXPERIMENT 10: Non-Linear Predictors")
    print("#" * 60)
    from experiments.run_nonlinear import run_nonlinear_experiment
    run_nonlinear_experiment(n_reps=reps_nonlinear)

    # Experiment 11: Feature Scaling Sensitivity
    print("\n\n" + "#" * 60)
    print("# EXPERIMENT 11: Feature Scaling Sensitivity")
    print("#" * 60)
    from experiments.run_feature_scaling import run_feature_scaling_experiment
    run_feature_scaling_experiment(n_reps=reps_feature_scaling)

    # Experiment 12: Ridge Extended (LWCP+, lambda sweep, heteroscedastic)
    print("\n\n" + "#" * 60)
    print("# EXPERIMENT 12: Ridge Extended")
    print("#" * 60)
    from experiments.run_ridge import run_ridge_extended
    run_ridge_extended(n_reps=reps_ridge_ext)

    # Experiment 13: Weight Selection Sensitivity
    print("\n\n" + "#" * 60)
    print("# EXPERIMENT 13: Weight Selection Sensitivity")
    print("#" * 60)
    from experiments.run_weight_selection import run_weight_selection_experiment
    run_weight_selection_experiment(n_reps=reps_weight_sel)

    # Experiment 14: Real Data (with MSCE/WSC metrics)
    print("\n\n" + "#" * 60)
    print("# EXPERIMENT 14: Real Data")
    print("#" * 60)
    from experiments.run_real_data import run_real_data_experiment
    run_real_data_experiment(n_reps=reps_real_data)

    total_time = time.perf_counter() - total_start
    print("\n\n" + "=" * 60)
    print(f"ALL EXPERIMENTS COMPLETE. Total time: {total_time:.1f}s")
    print(f"Figures saved to: {Path(__file__).parent / 'figures'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
