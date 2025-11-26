import sys 
sys.path.append("..")
from src.grid_world import GridWorld
from src.monte_carlo import MonteCarloAgent
import numpy as np

if __name__ == "__main__":
    # Create environment
    env = GridWorld()
    
    # Create Monte Carlo agent
    agent = MonteCarloAgent(env, epsilon=0, gamma=0.9)
        
    # Train with periodic visualization
    agent.train(
        num_episodes=1, 
        method='first_visit',
        verbose=True,
        render_interval=1,  # 每100个episode可视化一次
        render_last=True      # 最后一个episode也可视化
    )
    
    
    # Get learned policy and values
    policy_matrix = agent.get_greedy_policy()
    state_values = agent.get_state_values()
    
    # Visualize learned policy
    env.reset()
    env.render()
    env.add_policy(policy_matrix)
    env.add_state_values(state_values)
    env.render()
    env.save_graphics("monte_carlo_policy.png")
    
    
    
