from unityagents import UnityEnvironment
from hyperparameters import *
from maddpg import start_maddpg_tennis



start_maddpg_tennis()


def explore_unity_env():
    env = UnityEnvironment(file_name='./Tennis_Windows_x86_64/Tennis.exe')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=TRAINMODE)[brain_name]
    # size of each action
    action_dim = brain.vector_action_space_size
    print('# dimensions of each action:', action_dim)
    # examine the state space
    states = env_info.vector_observations
    state_dim = states.shape[1]
    print('# dimensions of each state:', state_dim)
    # examine number of agents
    nr_agents=states.shape[0]
    print('# agents:', nr_agents)
