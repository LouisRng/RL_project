"""
Proximal Policy Optimization agent for the GridWorld environment.
The implementation uses a simple tabular-style state encoding (one-hot)
to suit the small, discrete grid layout.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class ActorCritic(nn.Module):
    def __init__(self, num_states: int, num_actions: int, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(num_states, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, num_actions)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, state_one_hot: torch.Tensor):
        x = F.relu(self.fc1(state_one_hot))
        x = F.relu(self.fc2(x))
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value


class PPOAgent:
    def __init__(
        self,
        env,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        lr: float = 3e-4,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        steps_per_epoch: int = 1000,
        train_iters: int = 10,
        batch_size: int = 128,
        device: str | torch.device = "cpu",
    ):
        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.steps_per_epoch = steps_per_epoch
        self.train_iters = train_iters
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.num_states = env.num_states
        self.num_actions = len(env.action_space)
        self.model = ActorCritic(self.num_states, self.num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.state_identity = torch.eye(self.num_states, device=self.device)

    def state_to_index(self, state):
        return state[1] * self.env.env_size[0] + state[0]

    def _state_one_hot(self, state_idx: int) -> torch.Tensor:
        return self.state_identity[state_idx]

    @torch.no_grad()
    def select_action(self, state):
        state_idx = self.state_to_index(state)
        state_tensor = self._state_one_hot(state_idx)
        logits, value = self.model(state_tensor)
        dist = Categorical(logits=logits)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        action = self.env.action_space[action_idx.item()]
        return action, action_idx.item(), log_prob.item(), value.item()

    def _gather_rollout(self):
        states = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []

        state, _ = self.env.reset()
        for _ in range(self.steps_per_epoch):
            action, action_idx, log_prob, value = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)

            states.append(self.state_to_index(state))
            actions.append(action_idx)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value)

            state = next_state if not done else self.env.reset()[0]

        with torch.no_grad():
            last_state_idx = self.state_to_index(state)
            last_logits, last_value = self.model(self._state_one_hot(last_state_idx))
            last_value = last_value.item()

        return (
            np.array(states, dtype=np.int64),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(log_probs, dtype=np.float32),
            np.array(values, dtype=np.float32),
            last_value,
        )

    def _compute_gae(self, rewards, dones, values, last_value):
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def _ppo_update(self, states, actions, old_log_probs, returns, advantages):
        num_samples = len(states)
        indices = np.arange(num_samples)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.train_iters):
            np.random.shuffle(indices)
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_states = self.state_identity[states[batch_idx]].to(self.device)
                batch_actions = torch.tensor(actions[batch_idx], device=self.device)
                batch_old_log_probs = torch.tensor(old_log_probs[batch_idx], device=self.device)
                batch_returns = torch.tensor(returns[batch_idx], device=self.device)
                batch_advantages = torch.tensor(advantages[batch_idx], device=self.device)

                logits, values = self.model(batch_states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratios = (new_log_probs - batch_old_log_probs).exp()
                clipped_ratios = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps)
                policy_loss = -torch.min(ratios * batch_advantages, clipped_ratios * batch_advantages).mean()

                value_loss = F.mse_loss(values, batch_returns)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

    def train(self, epochs: int = 200):
        for epoch in range(epochs):
            states, actions, rewards, dones, log_probs, values, last_value = self._gather_rollout()
            advantages, returns = self._compute_gae(rewards, dones, values, last_value)
            self._ppo_update(states, actions, log_probs, returns, advantages)

            if (epoch + 1) % 10 == 0:
                avg_return = np.mean(returns)
                print(f"Epoch {epoch + 1}: mean return {avg_return:.3f}")

    def get_policy_matrix(self):
        policy_matrix = np.zeros((self.num_states, self.num_actions))
        with torch.no_grad():
            logits, _ = self.model(self.state_identity)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            policy_matrix[:] = probs
        return policy_matrix

    def get_state_values(self):
        with torch.no_grad():
            _, values = self.model(self.state_identity)
        return values.cpu().numpy()
