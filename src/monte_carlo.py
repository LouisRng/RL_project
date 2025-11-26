'''
Monte Carlo ε-greedy algorithm for Grid World
Implementation based on standard RL textbook pseudocode
'''
import numpy as np
import pickle
import os

class MonteCarloAgent:
    def __init__(self, env, epsilon=0.1, gamma=0.95):
        """
        Initialize Monte Carlo ε-greedy agent
        
        Args:
            env: GridWorld environment instance
            epsilon: exploration parameter ε ∈ (0,1]
            gamma: discount factor γ ∈ [0,1]
        """
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        
        # Initialize Q(s,a) - value function
        self.Q = np.zeros((env.num_states, len(env.action_space)))
        
        # Initialize Returns(s,a) - sum of returns
        self.Returns = np.zeros((env.num_states, len(env.action_space)))
        
        # Initialize Num(s,a) - count of visits
        self.Num = np.zeros((env.num_states, len(env.action_space)))
        
        # Initialize policy π(a|s)
        self.policy = np.ones((env.num_states, len(env.action_space))) / len(env.action_space)
        
    def state_to_index(self, state):
        """Convert (x, y) state to linear index"""
        return state[1] * self.env.env_size[0] + state[0]
    
    def select_action(self, state):
        """
        Select action according to current policy π(a|s)
        
        Args:
            state: current state (x, y)
            
        Returns:
            action: selected action tuple
            action_idx: index of selected action
        """
        state_idx = self.state_to_index(state)
        action_idx = np.random.choice(len(self.env.action_space), p=self.policy[state_idx])
        return self.env.action_space[action_idx], action_idx
    
    def generate_episode(self, max_steps=1000, render=False):
        """
        Episode generation: Generate an episode following current policy
        
        Args:
            max_steps: maximum steps per episode
            render: whether to visualize the episode generation
            
        Returns:
            episode: [(s0,a0,r1), (s1,a1,r2), ..., (sT-1,aT-1,rT)]
        """
        episode = []
        state, _ = self.env.reset()
        
        for step in range(max_steps):
            if render:
                self.env.render(0.01)
            
            action, action_idx = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            
            episode.append((state, action_idx, reward))
            
            # if done:
            #     if render:
            #         self.env.render()
            #     break

            state = next_state
        
        return episode
    
    def update_policy(self, state_idx):
        """
        Policy improvement: Update ε-greedy policy for state s
        
        π(a|s) = 1 - ε·(|A(s)|-1)/|A(s)|,  if a = a*
                 ε/|A(s)|,                  if a ≠ a*
        
        where a* = arg max_a Q(s,a)
        """
        num_actions = len(self.env.action_space)
        
        # Find best action: a* = arg max_a Q(s,a)
        best_action = np.argmax(self.Q[state_idx])
        
        # Update policy according to ε-greedy formula
        for a in range(num_actions):
            if a == best_action:
                self.policy[state_idx, a] = 1 - self.epsilon * (num_actions - 1) / num_actions
            else:
                self.policy[state_idx, a] = self.epsilon / num_actions
    
    def train(self, num_episodes=1000, max_steps=1000, method='every_visit', verbose=True, 
              render_interval=None, render_last=False):
        """
        Monte Carlo ε-greedy algorithm
        
        Args:
            num_episodes: number of episodes to train
            method: 'first_visit' or 'every_visit'
            verbose: print training progress
            render_interval: visualize every N episodes (None = no visualization)
            render_last: visualize the last episode
        """
        episode_rewards = []
        
        for ep in range(num_episodes):
            # Determine if we should render this episode
            should_render = False
            if render_interval is not None and (ep + 1) % render_interval == 0:
                should_render = True
            if render_last and ep == num_episodes - 1:
                should_render = True
            
            # Episode generation
            episode = self.generate_episode(max_steps, render=should_render)
            
            # Track total reward for this episode
            total_reward = sum([r for _, _, r in episode])
            episode_rewards.append(total_reward)
            
            # Initialization for each episode: g ← 0
            G = 0
            
            # Track visited state-action pairs for first-visit
            visited = set() if method == 'first_visit' else None
            
            # For each step t = T-1, T-2, ..., 0
            T = len(episode)
            for t in range(T - 1, -1, -1):
                state, action_idx, reward = episode[t]
                state_idx = self.state_to_index(state)
                
                # g ← γ·g + r_{t+1}
                G = self.gamma * G + reward
                
                # First-visit check
                if method == 'first_visit':
                    if (state_idx, action_idx) in visited:
                        continue
                    visited.add((state_idx, action_idx))
                
                # Returns(s_t, a_t) ← Returns(s_t, a_t) + g
                self.Returns[state_idx, action_idx] += G
                
                # Num(s_t, a_t) ← Num(s_t, a_t) + 1
                self.Num[state_idx, action_idx] += 1
                
                # Policy evaluation: Q(s_t, a_t) ← Returns(s_t, a_t) / Num(s_t, a_t)
                self.Q[state_idx, action_idx] = self.Returns[state_idx, action_idx] / self.Num[state_idx, action_idx]
                
                # Policy improvement: Update π(a|s_t)
                self.update_policy(state_idx)
            
            # Print progress
            if verbose and (ep + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_length = np.mean([len(self.generate_episode()) for _ in range(10)])
                
                print(f"Episode {ep + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}")
        
        return episode_rewards
    
    def get_greedy_policy(self):
        """
        Extract deterministic greedy policy (for visualization)
        Returns policy matrix where π(a*|s) = 1.0
        """
        policy_matrix = np.zeros((self.env.num_states, len(self.env.action_space)))
        for s in range(self.env.num_states):
            best_action = np.argmax(self.Q[s])
            policy_matrix[s, best_action] = 1.0
        return policy_matrix
    
    def get_state_values(self):
        """Extract state values: V(s) = max_a Q(s,a)"""
        return np.max(self.Q, axis=1)
    
