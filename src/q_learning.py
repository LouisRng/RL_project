"""
Q-learning (off-policy) agent for the GridWorld environment.

Provides:
- `run_value_iteration` to compute ground-truth optimal state values/policy.
- `QLearningAgent` implementing tabular Q-learning with epsilon-greedy behavior.
"""
import numpy as np


def state_to_index(state, env_size):
    """Convert (x, y) -> linear index for a grid with shape env_size."""
    return state[1] * env_size[0] + state[0]


def run_value_iteration(env, gamma=0.9, max_iterations=10_000, tol=1e-6):
    """
    Compute the optimal state values and greedy policy for the given GridWorld.

    This uses the environment's transition/reward logic so the result is
    consistent with training dynamics (including walls/forbidden/target/stay).
    Returns:
        values: np.ndarray of shape (num_states,)
        policy: one-hot matrix of shape (num_states, num_actions)
    """
    num_states = env.num_states
    num_actions = len(env.action_space)
    values = np.zeros(num_states)

    for _ in range(max_iterations):
        delta = 0.0
        new_values = np.zeros_like(values)
        for idx in range(num_states):
            x = idx % env.env_size[0]
            y = idx // env.env_size[0]
            q_candidates = []
            for action in env.action_space:
                next_state, reward = env._get_next_state_and_reward((x, y), action)
                next_idx = state_to_index(next_state, env.env_size)
                q_candidates.append(reward + gamma * values[next_idx])
            best_q = max(q_candidates)
            new_values[idx] = best_q
            delta = max(delta, abs(best_q - values[idx]))
        values = new_values
        if delta < tol:
            break

    # Greedy policy from converged values
    policy = np.zeros((num_states, num_actions))
    for idx in range(num_states):
        x = idx % env.env_size[0]
        y = idx // env.env_size[0]
        q_candidates = []
        for action in env.action_space:
            next_state, reward = env._get_next_state_and_reward((x, y), action)
            next_idx = state_to_index(next_state, env.env_size)
            q_candidates.append(reward + gamma * values[next_idx])
        best_action = int(np.argmax(q_candidates))
        policy[idx, best_action] = 1.0

    return values, policy


class QLearningAgent:
    """Tabular off-policy Q-learning with epsilon-greedy behavior."""

    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1, ground_truth_values=None):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((env.num_states, len(env.action_space)))
        self.ground_truth_values = ground_truth_values

    def _state_index(self, state):
        return state_to_index(state, self.env.env_size)

    def select_action(self, state):
        """Epsilon-greedy action selection on current Q-values."""
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(len(self.env.action_space))
        else:
            action_idx = int(np.argmax(self.Q[self._state_index(state)]))
        return self.env.action_space[action_idx], action_idx

    def learn(self, num_steps=100_000, track_error=True):
        """
        Run a single long episode of Q-learning updates.

        Returns:
            errors: list of mean absolute state-value error per step (if tracked)
        """
        errors = []
        state, _ = self.env.reset()

        for _ in range(num_steps):
            action, action_idx = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)

            s_idx = self._state_index(state)
            ns_idx = self._state_index(next_state)
            td_target = reward + self.gamma * np.max(self.Q[ns_idx])
            td_error = td_target - self.Q[s_idx, action_idx]
            self.Q[s_idx, action_idx] += self.alpha * td_error

            if track_error and self.ground_truth_values is not None:
                est_values = np.max(self.Q, axis=1)
                errors.append(np.mean(np.abs(self.ground_truth_values - est_values)))

            state = next_state

            # Continue stepping even if done=True to keep the episode length fixed
            if done:
                # Do not reset trajectory; keep walking from the terminal state
                pass

        return errors

    def get_greedy_policy(self):
        """Return deterministic greedy policy matrix."""
        policy = np.zeros_like(self.Q)
        best_actions = np.argmax(self.Q, axis=1)
        for s, a in enumerate(best_actions):
            policy[s, a] = 1.0
        return policy

    def get_state_values(self):
        """State value estimates V(s) = max_a Q(s,a)."""
        return np.max(self.Q, axis=1)
