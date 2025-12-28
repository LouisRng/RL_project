"""
TD-Linear实验主程序
实现作业要求的所有功能：
1. 计算ground truth（策略评估）
2. 使用TD-Linear算法近似状态值
3. 比较多项式和傅立叶特征
4. 可视化结果（表格和3D图）
"""
import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src. grid_world import GridWorld
from src.td_linear import TDLinear
from src.episode_generator import EpisodeGenerator

def create_uniform_policy(env):
    """创建均匀随机策略:  π(a|s) = 0.2 for all a"""
    num_actions = len(env. action_space)
    policy = np.ones((env.num_states, num_actions)) / num_actions
    return policy

def compute_rmse(estimated_values, true_values):
    """计算RMSE误差"""
    return np.sqrt(np.mean((estimated_values - true_values) ** 2))

def print_value_table(values, title="状态值"):
    """以5×5表格形式打印状态值"""
    print(f"\n{title}:")
    print("=" * 60)
    value_matrix = values.reshape(5, 5)
    for i in range(5):
        row_str = "  ".join([f"{value: 7.3f}" for value in value_matrix[i]])
        print(f"Row {i+1}: {row_str}")
    print("=" * 60)

def plot_3d_surface(value_matrix, title="State Values", ax=None):
    """绘制3D状态值曲面"""
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    
    rows, cols = np.meshgrid(range(1, 6), range(1, 6))
    surf = ax.plot_surface(cols, rows, value_matrix. T, cmap='viridis', 
                          edgecolor='black', linewidth=0.5, alpha=0.8)
    
    ax.set_xlabel('column')
    ax.set_ylabel('row')
    ax.set_zlabel('State Value')
    ax.set_title(title)
    ax.invert_yaxis()
    
    return ax

def plot_error_curve(errors, title="TD-Linear", alpha=0.0005, ax=None):
    """绘制RMSE误差曲线"""
    if ax is None: 
        fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(range(len(errors)), errors, linewidth=2, 
            label=f'TD-Linear:  α={alpha}')
    ax.set_xlabel('Episode index', fontsize=12)
    ax.set_ylabel('State value error (RMSE)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax

def run_experiment(env, policy, true_values, episodes, 
                   feature_type, feature_param, alpha=0.0005):
    """
    运行单个TD-Linear实验
    
    Returns:
        estimated_values:  估计的状态值
        errors: 每个episode后的RMSE误差列表
    """
    # 初始化TD-Linear
    td_learner = TDLinear(env, feature_type=feature_type, 
                         feature_param=feature_param, alpha=alpha, gamma=0.9)
    
    # 训练并记录每个episode后的误差
    errors = []
    # print(f"开始训练...")
    
    for i, episode in enumerate(episodes):
        # 在单个episode上训练
        td_learner.train_on_episode(episode)
        
        # 计算当前误差
        estimated_values = td_learner.get_all_state_values()
        error = compute_rmse(estimated_values, true_values)
        errors.append(error)
        
        # if (i + 1) % 100 == 0:
        #     print(f"  Episode {i+1}/{len(episodes)}, RMSE={error:.4f}")
    
    final_values = td_learner.get_all_state_values()
    final_error = compute_rmse(final_values, true_values)
    
    print(f"最终RMSE误差:  {final_error:.4f}")
    
    return final_values, errors

def main():
    """主函数"""
    
    # 1. 初始化环境和策略
    env = GridWorld()
    policy = create_uniform_policy(env)
    print(f"环境大小: {env.env_size}")
    print(f"状态数: {env.num_states}")
    print(f"动作数:  {len(env.action_space)}")
    print(f"策略: 均匀随机 π(a|s) = {1/len(env. action_space):.1f}")
    
    # 2. 计算ground truth
    true_values, iterations = env.policy_evaluation(policy, gamma=0.9)
    print_value_table(true_values, "Ground Truth状态值")
    
    # 可视化Ground Truth的3D图
    fig_gt = plt.figure(figsize=(10, 8))
    ax_gt = fig_gt.add_subplot(111, projection='3d')
    true_value_matrix = true_values.reshape(5, 5)
    plot_3d_surface(true_value_matrix, 'Ground Truth State Values', ax_gt)
    plt.savefig('ground_truth_3d.png', dpi=300, bbox_inches='tight')
    
    # 3. 生成episodes
    generator = EpisodeGenerator(env, policy)
    episodes = generator.generate_episodes(num_episodes=500, max_steps=500)
    
    # 4. 多项式特征实验
    
    poly_configs = [
        (1, 3, r'$\phi(s) \in \mathbb{R}^3$'),
        (2, 6, r'$\phi(s) \in \mathbb{R}^6$'),
        (3, 10, r'$\phi(s) \in \mathbb{R}^{10}$')
    ]
    
    fig_poly = plt.figure(figsize=(18, 10))
    
    for idx, (order, dim, label) in enumerate(poly_configs):
        print(f"\n--- 多项式特征 (阶数={order}, 维度={dim}) ---")
        
        values, errors = run_experiment(env, policy, true_values, episodes,
                                       'polynomial', order, alpha=0.0005)
        
        print_value_table(values, f"多项式(阶数={order})估计状态值")
        
        # 3D曲面图
        ax1 = fig_poly. add_subplot(2, 3, idx+1, projection='3d')
        value_matrix = values.reshape(5, 5)
        plot_3d_surface(value_matrix, 'TD-Linear', ax1)
        
        # 误差曲线
        ax2 = fig_poly.add_subplot(2, 3, idx+4)
        plot_error_curve(errors, 'TD-Linear', 0.0005, ax2)
        ax2.text(0.95, 0.95, label, transform=ax2.transAxes,
                fontsize=14, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('td_linear_polynomial_results.png', dpi=300, bbox_inches='tight')
    
    # 5. 傅立叶特征实验
    
    fourier_configs = [
        (1, 4, r'$\phi(s) \in \mathbb{R}^4$ (q=1)'),
        (2, 9, r'$\phi(s) \in \mathbb{R}^9$ (q=2)'),
        (3, 16, r'$\phi(s) \in \mathbb{R}^{16}$ (q=3)')
    ]
    
    fig_fourier = plt.figure(figsize=(18, 10))
    
    for idx, (q, dim, label) in enumerate(fourier_configs):
        print(f"\n--- 傅立叶特征 (q={q}, 维度={dim}) ---")
        
        values, errors = run_experiment(env, policy, true_values, episodes,
                                       'fourier', q, alpha=0.0005)
        
        print_value_table(values, f"傅立叶(q={q})估计状态值")
        
        # 3D曲面图
        ax1 = fig_fourier.add_subplot(2, 3, idx+1, projection='3d')
        value_matrix = values.reshape(5, 5)
        plot_3d_surface(value_matrix, 'TD-Linear', ax1)
        
        # 误差曲线
        ax2 = fig_fourier.add_subplot(2, 3, idx+4)
        plot_error_curve(errors, 'TD-Linear', 0.0005, ax2)
        ax2.text(0.95, 0.95, label, transform=ax2.transAxes,
                fontsize=14, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('td_linear_fourier_results.png', dpi=300, bbox_inches='tight')
    
    # 6. 绘制ground truth的网格世界可视化
    env.render(0.01)
    env.add_policy(policy)
    env.add_state_values(true_values, precision=2)
    env.save_graphics("ground_truth_visualization.png")
    
    plt.show()    

if __name__ == "__main__":
    main()
