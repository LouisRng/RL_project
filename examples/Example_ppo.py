import sys
sys.path.append("..")
from src.grid_world import GridWorld
from src.ppo_agent import PPOAgent


def visualize(env: GridWorld, agent: PPOAgent):
    policy_matrix = agent.get_policy_matrix()
    state_values = agent.get_state_values()

    env.reset()
    env.render()
    env.add_policy(policy_matrix)
    env.add_state_values(state_values, precision=2)
    env.render()
    env.save_graphics("ppo_policy.png")


def main():
    env = GridWorld()
    agent = PPOAgent(
        env,
        gamma=0.98,
        lam=0.95,
        clip_eps=0.2,
        lr=3e-4,
        entropy_coef=0.01,
        value_coef=0.5,
        steps_per_epoch=500,
        train_iters=10,
        batch_size=128,
    )

    agent.train(epochs=150)
    visualize(env, agent)


if __name__ == "__main__":
    main()
