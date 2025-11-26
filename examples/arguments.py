'''
Specify parameters of the env
'''
import numpy as np
import argparse
from ast import literal_eval

parser = argparse.ArgumentParser("Grid World Environment")

## ==================== User settings ====================
# specify the number of columns and rows of the grid world
parser.add_argument("--env-size", type=lambda x: literal_eval(x), default=(5,5))   

# specify the start state
parser.add_argument("--start-state", type=lambda x: literal_eval(x), default=(0,0))

# specify the target state
parser.add_argument("--target-state", type=lambda x: literal_eval(x), default=(2,3))

# specify the forbidden states
parser.add_argument("--forbidden-states", type=lambda x: literal_eval(x), default=[(1,1), (2,1), (2,2), (1,3), (3,3), (1,4)])

# specify the reward when reaching target
parser.add_argument("--reward-target", type=float, default=1)

# specify the reward when entering into forbidden area
parser.add_argument("--reward-forbidden", type=float, default=-2)

# specify the reward when hitting the wall
parser.add_argument("--reward-wall", type=float, default=-1)

# specify the reward for each step
parser.add_argument("--reward-step", type=float, default=0)

## ==================== End of User settings ====================


## ==================== Monte Carlo Settings ====================
parser.add_argument("--mc-epsilon", type=float, default=0.1, help="Exploration rate for epsilon-greedy policy (0-1)")
parser.add_argument("--mc-gamma", type=float, default=0.9, help="Discount factor for future rewards (0-1)")
parser.add_argument("--mc-alpha", type=float, default=0.1, help="Learning rate for Q-value updates (0-1)")
parser.add_argument("--mc-num-episodes", type=int, default=1000, help="Number of training episodes")
parser.add_argument("--mc-method", type=str, default='first_visit', choices=['first_visit', 'every_visit'], 
                    help="Monte Carlo method: first_visit or every_visit")
parser.add_argument("--mc-model-path", type=str, default='monte_carlo_model.pkl', help="Path to save/load trained model")
## ==================== End of Monte Carlo Settings ====================


## ==================== Advanced Settings ====================
parser.add_argument("--action-space", type=lambda x: literal_eval(x), default=[(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)])  # down, right, up, left, stay           
parser.add_argument("--debug", type=lambda x: x.lower() in ['true', '1', 'yes'], default=False)
parser.add_argument("--animation-interval", type=float, default=0.1)

parser.add_argument("--save-graphics", type=lambda x: x.lower() in ['true', '1', 'yes'], default=False)
## ==================== End of Advanced settings ====================


args = parser.parse_args()

def validate_environment_parameters(
    env_size: tuple | list | np.ndarray,  # 使用 | 替代 Union
    start_state: tuple | list | np.ndarray,
    target_state: tuple | list | np.ndarray,
    forbidden_states: list
) -> None:
    if not (isinstance(env_size, (tuple, list, np.ndarray)) and len(env_size) == 2):
        raise ValueError("Invalid environment size. Expected a tuple (rows, cols) with positive dimensions.")
    
    for i in range(2):
        assert start_state[i] < env_size[i], f"start_state[{i}] must be < env_size[{i}]"
        assert target_state[i] < env_size[i], f"target_state[{i}] must be < env_size[{i}]"
        for j in range(len(forbidden_states)):
            assert forbidden_states[j][i] < env_size[i], f"forbidden_states[{j}][{i}] must be < env_size[{i}]"

try:
    validate_environment_parameters(args.env_size, args.start_state, args.target_state, args.forbidden_states)
except (ValueError, AssertionError) as e:
    print("Error:", e)   
