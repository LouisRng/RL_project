import sys 
sys.path.append("..")
from src.grid_world import GridWorld
from src.monte_carlo import MonteCarloAgent
import numpy as np
import matplotlib.pyplot as plt

def train_agent(epsilon, num_episodes=500):
    """训练agent并返回策略和状态值"""
    env = GridWorld()
    agent = MonteCarloAgent(env, epsilon=epsilon, gamma=0.9)
    
    print(f"Training ε={epsilon}...")
    agent.train(
        num_episodes=num_episodes, 
        method='first_visit',
        verbose=True,
        render_last=False  
    ) 
    policy = agent.get_greedy_policy()
    values = agent.get_state_values()
    
    return policy, values

def visualize_policy(epsilon, policy, values):
    """可视化策略和状态值"""
    env = GridWorld()
    env.reset()
    env.render(animation_interval=0.01)
    env.add_policy(policy)
    env.add_state_values(values, precision=1)
    env.render(animation_interval=0.1)
    env.save_graphics(f'policy_eps{epsilon}.png')
    plt.close('all')
    print(f"Saved: policy_eps{epsilon}.png")

def print_state_values(epsilon, values, env_size):
    """打印状态值"""
    print(f"\nε={epsilon} State Values:")
    value_grid = values.reshape(env_size)
    print(value_grid)

if __name__ == "__main__":
    epsilon_values = [0.0, 0.1, 0.2, 0.5]
    env = GridWorld()
    
    for eps in epsilon_values:
        policy, values = train_agent(eps, num_episodes=5000)
        print_state_values(eps, values, env.env_size)
        visualize_policy(eps, policy, values)
    
    print("\nAll policies saved!")
