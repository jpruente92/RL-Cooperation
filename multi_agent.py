

import numpy as np


from agent import Agent
from hyperparameters import *
from replay_buffer import ReplayBuffer


class Multi_agent:
    def __init__(self, num_agents, state_size, action_size):
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size

        self.agents = [Agent(state_size, action_size, i) for i in range(num_agents)]

        # Replay buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)

        self.t_step = 0


    def save_networks(self):
        for agent in self.agents:
            agent.save_networks()



    # return the action for both actors concatenated
    # input: numpy array-> matrix with shape (number agents,state dimension)
    # output numpy array-> vector with shape(number agents, action dimension)
    def act(self, observations):
        actions=[]
        for i,agent in enumerate(self.agents):
            state = torch.from_numpy(observations[i]).float().unsqueeze(0).to(DEVICE)
            action=agent.policy_network_local.evaluate(state,False).squeeze().cpu().data.numpy()
            action=np.clip(action, -1, 1)
            actions.append(action)
        actions=np.array(actions)
        return actions



    def learn(self):
        for agent in self.agents:
            agent.learn(self.memory,self.agents)



