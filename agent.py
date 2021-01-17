

import torch.nn.functional as F
import torch.optim as optim

from hyperparameters import *
from neural_networks import Policy_network, Q_network




class Agent:

    def __init__(self, state_size, action_size, id):

        self.state_size = state_size
        self.action_size = action_size
        self.id = id

        # Policy/Actor Network (w/ Target Network)
        self.policy_network_local = Policy_network(state_size, action_size,FILENAME_FOR_LOADING+str(self.id)+"_policy_local.pth").to(DEVICE)
        self.policy_network_target = Policy_network(state_size, action_size,FILENAME_FOR_LOADING+str(self.id)+"_policy_target.pth").to(DEVICE)
        self.policy_network_optimizer = optim.Adam(self.policy_network_local.parameters(), lr=LR_ACTOR)

        # Qvalue/Critic Network (w/ Target Network)
        self.qvalue_network_local = Q_network(state_size, action_size,FILENAME_FOR_LOADING+str(self.id)+"_qvalue_local.pth").to(DEVICE)
        self.qvalue_network_target = Q_network(state_size, action_size,FILENAME_FOR_LOADING+str(self.id)+"_qvalue_target.pth").to(DEVICE)
        self.qvalue_network_optimizer = optim.Adam(self.qvalue_network_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)





    def save_networks(self):
        self.policy_network_local.save(FILENAME_FOR_SAVING+str(self.id)+"_policy_local.pth")
        self.policy_network_target.save(FILENAME_FOR_SAVING+str(self.id)+"_policy_target.pth")
        self.qvalue_network_local.save(FILENAME_FOR_SAVING+str(self.id)+"_qvalue_local.pth")
        self.qvalue_network_target.save(FILENAME_FOR_SAVING+str(self.id)+"_qvalue_target.pth")



    def learn(self,replay_buffer,agents):
        states, actions, rewards, next_states, dones = replay_buffer.sample()

        # observations are for the actor
        # states are for the critic
        observations = states.reshape(-1, len(agents), self.state_size)
        next_observations =next_states.reshape(-1, len(agents), self.state_size)

        # update critic
            # compute next actions
        actions_next=[]
        for i,agent in enumerate(agents):
            action= agent.policy_network_target.evaluate(next_observations.index_select(1, torch.tensor([i])),False).squeeze()
            actions_next.append(action)
        actions_next=torch.cat(actions_next,dim =1).to(DEVICE)
            # compute next Q values
        Q_next_states = self.qvalue_network_target.evaluate(next_states, actions_next,False)
            # Compute Q targets for current states
        Q_targets = rewards.index_select(1, torch.tensor([self.id])) + (GAMMA * Q_next_states * (1 - dones.index_select(1, torch.tensor([self.id]))))
            # Compute current Q values
        Q_current_states = self.qvalue_network_local(states, actions)
            # compute loss
        qvalue_loss = F.mse_loss(Q_current_states, Q_targets)
            # Minimize the loss
        self.qvalue_network_optimizer.zero_grad()
        qvalue_loss.backward()
        self.qvalue_network_optimizer.step()

        # update actor
            # compute the action predicted by the actors
        actions_predicted=[]
        for i,agent in enumerate(agents):
            action= agent.policy_network_local(observations.index_select(1, torch.tensor([i]))).squeeze()
            actions_predicted.append(action)
        actions_predicted=torch.cat(actions_predicted,dim =1).to(DEVICE)
            # compute the policy loss, which is the mean of the Q values times -1
        policy_loss = -self.qvalue_network_local(states, actions_predicted).mean()
            # Minimize the loss
        self.policy_network_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_network_optimizer.step()

        # update target networks
        self.soft_update(self.qvalue_network_local, self.qvalue_network_target, TAU)
        self.soft_update(self.policy_network_local, self.policy_network_target, TAU)



    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

