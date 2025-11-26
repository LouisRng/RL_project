import sys 
sys.path.append("..")
from src.grid_world import GridWorld
from src.monte_carlo import MonteCarloAgent
import numpy as np
import matplotlib.pyplot as plt

def analyze_episode(env, epsilon, max_steps, train_first=True):
    """生成一个episode并统计状态-动作对的访问次数"""
    agent = MonteCarloAgent(env, epsilon=epsilon, gamma=0.9)
    
    # must train first to populate the Q-table
    if train_first:
        agent.train(num_episodes=1000, method='first_visit', verbose=False)
    
    visit_counts = np.zeros((env.num_states, len(env.action_space)))
    
    state, _ = env.reset()
    for step in range(max_steps):
        action, action_idx = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        state_idx = agent.state_to_index(state)
        visit_counts[state_idx, action_idx] += 1
        
        state = next_state
        
        # if done:
        #     state, _ = env.reset()
    
    return visit_counts.flatten()

def plot_coverage(visit_counts, epsilon, max_steps):
    """绘制状态-动作对访问次数的散点图"""
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(visit_counts))
    plt.scatter(x, visit_counts, alpha=0.7, s=50, color='tab:orange')
    
    plt.xlabel('State-Action Pair Index', fontsize=14)
    plt.ylabel('Visit Count', fontsize=14)
    plt.title(f'ε={epsilon}, Episode Length={max_steps:,} steps', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'coverage_eps{epsilon}_steps{max_steps}.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    env = GridWorld()
    
    # 测试参数
    epsilon_values = [1.0, 0.5]
    step_values = [100, 1000, 10000, 1000000]
    
    for epsilon in epsilon_values:
        for max_steps in step_values:
            print(f"Analyzing ε={epsilon}, steps={max_steps}...")
            visit_counts = analyze_episode(env, epsilon, max_steps)
            plot_coverage(visit_counts, epsilon, max_steps)
            print(f"  Visited state-action pairs: {np.sum(visit_counts > 0)}/{len(visit_counts)}")
            print(f"  Average visits per pair: {np.mean(visit_counts[visit_counts > 0]):.1f}")
