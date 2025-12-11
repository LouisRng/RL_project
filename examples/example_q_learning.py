import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt

from src.grid_world import GridWorld
from src.q_learning import QLearningAgent, run_value_iteration


def plot_error_curve(errors, epsilon, filename):
    plt.figure(figsize=(7, 4))
    steps = np.arange(1, len(errors) + 1)
    plt.plot(steps, errors, linewidth=1.0, color="tab:blue")
    plt.xlabel("Step")
    plt.ylabel("Mean |V* - V_hat|")
    plt.title(f"Q-learning state-value error (ε={epsilon})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def visualize_episode(env, policy, values, epsilon, filename):
    # Render stored trajectory and overlay policy & state values
    env.render(animation_interval=0.001)
    env.add_policy(policy)
    env.add_state_values(values, precision=2)
    env.render(animation_interval=0.001)
    env.save_graphics(filename)
    plt.close("all")


def main():
    epsilons = [1.0, 0.5, 0.1]
    num_steps = 100_000
    gamma = 0.9
    alpha = 0.1

    # Ground-truth optimal values/policy (Bellman optimality)
    gt_env = GridWorld()
    ground_truth_values, optimal_policy = run_value_iteration(gt_env, gamma=gamma)

    for eps in epsilons:
        env = GridWorld()
        agent = QLearningAgent(
            env,
            alpha=alpha,
            gamma=gamma,
            epsilon=eps,
            ground_truth_values=ground_truth_values,
        )

        errors = agent.learn(num_steps=num_steps, track_error=True)
        learned_policy = agent.get_greedy_policy()
        learned_values = agent.get_state_values()

        traj_file = f"q_learning_traj_eps{eps}.png"
        error_file = f"q_learning_error_eps{eps}.png"

        visualize_episode(env, learned_policy, learned_values, eps, traj_file)
        plot_error_curve(errors, eps, error_file)

        print(
            f"ε={eps}: trajectory saved to {traj_file}, error curve saved to {error_file}"
        )


if __name__ == "__main__":
    main()
