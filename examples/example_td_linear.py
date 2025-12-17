"""
TD-Linear evaluation for the 5x5 GridWorld assignment.

This script:
1. Computes the ground-truth state values for the uniform random policy.
2. Trains TD(0) with linear approximation using polynomial and Fourier features.
3. Renders the 5x5 table of estimated values.
4. Outputs 3D surfaces and RMSE curves for comparison with the ground truth.
"""
import argparse
from pathlib import Path

import numpy as np

from src.grid_world import GridWorld
from src.td_linear import (
    TDLinearAgent,
    fourier_feature_factory,
    plot_error_curve,
    plot_state_values_surface,
    policy_evaluation,
    polynomial_feature_factory,
    visualize_state_values_on_grid,
)


def run_experiment(feature_name: str, feature_fn, env, ground_truth, output_dir: Path,
                   alpha: float = 0.05):
    agent = TDLinearAgent(env, feature_fn, alpha=alpha, gamma=0.9)
    errors = agent.train(num_episodes=500, max_steps=500, ground_truth=ground_truth)
    est_values = agent.estimate_state_values()

    surface_path = output_dir / f"{feature_name}_surface.png"
    error_path = output_dir / f"{feature_name}_rmse.png"
    table_path = output_dir / f"{feature_name}_values.png"

    plot_state_values_surface(est_values, env.env_size, f"TD-Linear ({feature_name})", surface_path)
    plot_error_curve(errors, f"TD-Linear ({feature_name})", error_path)
    visualize_state_values_on_grid(env, est_values, precision=3, output_path=table_path)

    return est_values, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/td_linear"))
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    env = GridWorld()
    ground_truth = policy_evaluation(env, gamma=0.9)

    print("Ground-truth state values (row-major):")
    print(ground_truth.reshape(env.env_size[1], env.env_size[0]))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("poly_order1", polynomial_feature_factory(1, env.env_size)),
        ("poly_order2", polynomial_feature_factory(2, env.env_size)),
        ("poly_order3", polynomial_feature_factory(3, env.env_size)),
        ("fourier_q1", fourier_feature_factory(1, env.env_size)),
        ("fourier_q2", fourier_feature_factory(2, env.env_size)),
        ("fourier_q3", fourier_feature_factory(3, env.env_size)),
    ]

    for name, feat_fn in experiments:
        print(f"\nRunning TD-Linear with features: {name}")
        est_values, errors = run_experiment(
            name, feat_fn, env, ground_truth, output_dir, alpha=args.alpha
        )
        print("Estimated values (row-major):")
        print(np.round(est_values.reshape(env.env_size[1], env.env_size[0]), 3))
        print(f"Final RMSE: {errors[-1]:.4f}")

    # Also visualize the ground-truth surface for reference
    plot_state_values_surface(ground_truth, env.env_size, "Ground Truth", output_dir / "ground_truth_surface.png")
    visualize_state_values_on_grid(env, ground_truth, precision=3, output_path=output_dir / "ground_truth_values.png")


if __name__ == "__main__":
    main()
