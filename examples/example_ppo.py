import sys 
sys.path.append("..")
from src.grid_world import GridWorld
from src.ppo_agent import PPOAgent
import numpy as np

if __name__ == "__main__": 
    # Create environment
    env = GridWorld()
    
    # Create PPO agent
    agent = PPOAgent(
        env, 
        lr=3e-4,           # Learning rate
        gamma=0.99,        # Discount factor
        eps_clip=0.2,      # Clipping parameter
        K_epochs=10,       # Update epochs
        gae_lambda=0.95,   # GAE parameter
        c1=0.5,            # Value loss coefficient
        c2=0.01            # Entropy coefficient
    )
    
    # Train agent
    print("Training PPO agent...")
    agent.train(
        num_iterations=100,
        steps_per_iteration=2048,
        verbose=True,
        eval_interval=10
    )
    
    # Get learned policy and values
    policy_matrix = agent.get_greedy_policy()
    state_values = agent.get_state_values()
    
    # Visualize learned policy
    env.reset()
    env.render()
    env.add_policy(policy_matrix)
    env.add_state_values(state_values, precision=2)
    env.render()
    env.save_graphics("ppo_policy.png")
    
    # Save model
    agent.save_model("ppo_model.pth")
    
    print("\nTraining completed!  Check 'ppo_policy.png' for results.")
