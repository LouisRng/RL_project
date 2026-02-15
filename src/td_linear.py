"""
TD(0) with linear function approximation utilities for the GridWorld environment.

This module supports both polynomial and Fourier feature representations and
provides helpers to train on episodes generated from a fixed stochastic policy
(the uniform policy used in the assignment) and to visualize estimated state
values.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for 3D plotting


def state_to_index(state: tuple[int, int], env_size: tuple[int, int]) -> int:
    """Convert (x, y) to a flattened state index."""
    return state[1] * env_size[0] + state[0]


def policy_evaluation(env, gamma: float = 0.9, tol: float = 1e-8, max_iter: int = 10_000) -> np.ndarray:
    """
    Evaluate the given environment under the uniform random policy.

    The policy matches the assignment description where each of the five actions
    has probability 0.2 regardless of the state.
    """
    num_states = env.num_states
    action_prob = 1.0 / len(env.action_space)
    values = np.zeros(num_states)

    for _ in range(max_iter):
        delta = 0.0
        new_values = np.zeros_like(values)
        for idx in range(num_states):
            x = idx % env.env_size[0]
            y = idx // env.env_size[0]
            expected_return = 0.0
            for action in env.action_space:
                next_state, reward = env._get_next_state_and_reward((x, y), action)
                next_idx = state_to_index(next_state, env.env_size)
                expected_return += action_prob * (reward + gamma * values[next_idx])
            new_values[idx] = expected_return
            delta = max(delta, abs(values[idx] - expected_return))
        values = new_values
        if delta < tol:
            break
    return values


def normalize_state(state: tuple[int, int], env_size: tuple[int, int]) -> np.ndarray:
    """Scale state coordinates to [0, 1] for stable feature computation."""
    x, y = state
    return np.array([x / (env_size[0] - 1), y / (env_size[1] - 1)])


def polynomial_feature_factory(order: int, env_size: tuple[int, int]) -> Callable[[tuple[int, int]], np.ndarray]:
    """
    Create a polynomial feature function up to the requested order.

    The resulting dimensionality matches the assignment:
        order=1 -> [1, x, y]
        order=2 -> [1, x, y, x^2, y^2, xy]
        order=3 -> [1, x, y, x^2, y^2, xy, x^3, y^3, x^2 y, x y^2]
    where x and y are normalized to [0, 1].
    """
    def features(state: tuple[int, int]) -> np.ndarray:
        x, y = normalize_state(state, env_size)
        if order == 1:
            return np.array([1.0, x, y], dtype=float)
        if order == 2:
            return np.array([1.0, x, y, x ** 2, y ** 2, x * y], dtype=float)
        if order == 3:
            return np.array([
                1.0,
                x,
                y,
                x ** 2,
                y ** 2,
                x * y,
                x ** 3,
                y ** 3,
                (x ** 2) * y,
                x * (y ** 2),
            ], dtype=float)
        raise ValueError(f"Unsupported polynomial order: {order}")

    return features


def fourier_feature_factory(q: int, env_size: tuple[int, int]) -> Callable[[tuple[int, int]], np.ndarray]:
    """
    Create Fourier basis features with order q.

    The dimensionality follows (q+1)^2, matching q=1 -> 4, q=2 -> 9, q=3 -> 16.
    """
    coeffs = list(itertools.product(range(q + 1), repeat=2))

    def features(state: tuple[int, int]) -> np.ndarray:
        s_scaled = normalize_state(state, env_size)
        return np.array([
            np.cos(np.pi * (c[0] * s_scaled[0] + c[1] * s_scaled[1])) for c in coeffs
        ], dtype=float)

    return features


@dataclass
class TDLinearAgent:
    env: any
    feature_fn: Callable[[tuple[int, int]], np.ndarray]
    alpha: float = 0.01
    gamma: float = 0.9

    def __post_init__(self):
        self.weights = np.zeros_like(self.feature_fn((0, 0)), dtype=float)

    def _sample_action(self):
        idx = np.random.randint(len(self.env.action_space))
        return self.env.action_space[idx]

    def _randomize_start(self):
        x = np.random.randint(self.env.env_size[0])
        y = np.random.randint(self.env.env_size[1])
        self.env.agent_state = (x, y)
        return self.env.agent_state

    def estimate_state_values(self) -> np.ndarray:
        """Estimate V(s) for every state using the learned weights."""
        values = []
        for idx in range(self.env.num_states):
            x = idx % self.env.env_size[0]
            y = idx // self.env.env_size[0]
            values.append(float(np.dot(self.weights, self.feature_fn((x, y)))))
        return np.array(values)

    def train(self, num_episodes: int = 500, max_steps: int = 500,
              ground_truth: np.ndarray | None = None) -> list[float]:
        errors = []
        for _ in range(num_episodes):
            state = self._randomize_start()
            _ = self.env.reset()
            self.env.agent_state = state
            for _ in range(max_steps):
                action = self._sample_action()
                next_state, reward, done, _ = self.env.step(action)

                phi_s = self.feature_fn(state)
                phi_next = self.feature_fn(next_state)
                td_error = reward + self.gamma * np.dot(self.weights, phi_next) - np.dot(self.weights, phi_s)
                self.weights += self.alpha * td_error * phi_s
                state = next_state
                if done:
                    # continue stepping from the terminal state to keep episode length consistent
                    pass
            if ground_truth is not None:
                est_values = self.estimate_state_values()
                rmse = float(np.sqrt(np.mean((est_values - ground_truth) ** 2)))
                errors.append(rmse)
        return errors


def plot_state_values_surface(values: Iterable[float], env_size: tuple[int, int],
                              title: str, output_path: Path | None = None) -> None:
    grid = np.array(values).reshape(env_size[1], env_size[0])
    x = np.arange(1, env_size[0] + 1)
    y = np.arange(1, env_size[1] + 1)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, grid, cmap="viridis", edgecolor="k", linewidth=0.5, antialiased=True)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_title(title)
    ax.view_init(elev=30, azim=-120)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_error_curve(errors: Iterable[float], title: str, output_path: Path | None = None) -> None:
    fig, ax = plt.subplots()
    ax.plot(errors)
    ax.set_xlabel("Episode index")
    ax.set_ylabel("State value error (RMSE)")
    ax.set_title(title)
    ax.grid(True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def visualize_state_values_on_grid(env, values: Iterable[float], precision: int = 3,
                                   output_path: Path | None = None) -> None:
    """Use the existing GridWorld rendering utility to annotate values in a 5x5 table."""
    env.reset()
    env.render(animation_interval=0.001)
    env.add_state_values(values, precision=precision)
    env.canvas.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        env.save_graphics(str(output_path))
    plt.close(env.canvas)
