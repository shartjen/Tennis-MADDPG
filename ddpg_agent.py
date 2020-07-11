import numpy as np
# import random
# from collections import deque

from model import Actor, Critic
from colorama import  Back, Style
from OUNoise import OUNoise

import torch
import torch.nn.functional as F
import torch.optim as optim

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
# Conversion from numpy to tensor
def ten(x): return torch.from_numpy(x).float().to(device)

class ddpg_agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, config):
        """Initialize an Agent object.
        
        Params
        ======
            config : configuration given a variety of parameters
        """
        self.config = config

        # set parameter for ML
        self.set_parameters(config)
        # Q-Network
        self.create_networks()
        # Noise process
        self.noise = OUNoise(self.action_size, self.seed, sigma = self.sigma)
    
    def set_parameters(self, config):
        # Base agent parameters
        self.gamma = config['gamma']                    # discount factor 
        self.tau = config['tau']
        self.state_size = config['state_size']
        self.action_size = config['action_size']
        self.hidden_size = config['hidden_size']
        self.batch_size = config['batch_size']
        self.seed = config['seed']
        self.dropout = config['dropout']
        self.critic_learning_rate = config['critic_learning_rate']
        self.actor_learning_rate = config['actor_learning_rate']
        self.sigma = config['sigma']
        self.noise_scale = 1
        # Some debug flags
        self.Do_debug_values = False
        self.Do_debug_qr = False
        
    def create_networks(self):
        # Actor Network (local & Target Network)
        input_size = self.state_size*2
        action_input_size = 2*self.action_size
            
        self.actor_local = Actor(input_size, self.hidden_size, self.action_size, self.seed, self.dropout).to(device)
        self.actor_target = Actor(input_size, self.hidden_size, self.action_size, self.seed, self.dropout).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.actor_learning_rate)

        # Critic Network (local & Target Network)
        self.critic_local = Critic(input_size, self.hidden_size, action_input_size, self.seed, self.dropout).to(device)
        self.critic_target = Critic(input_size, self.hidden_size, action_input_size, self.seed, self.dropout).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.critic_learning_rate)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(state)
        self.actor_local.train()
        if add_noise:
            actions += self.noise_scale * self.noise.sample()
        actions = np.clip(actions, -1, 1)
        return actions
    
    def learn(self, states, actions, rewards, next_states, actions_next, dones):
        """Update policy and value parameters using given batch of experience tuples.
        q_target = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value """

        # print('agent learn shapes : ',states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape)
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        # actions_next = self.actor_target(next_states)
        # print('learn : Next States : ',next_states.shape)
        q_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        q_target = rewards[:,self.id].unsqueeze(1) + (self.gamma * q_next * (1 - dones[:,self.id].unsqueeze(1)))
        # Compute critic loss
        q_expected = self.critic_local(states, actions)
        # print('Agent learn shapes : ',actions_next.shape, q_next.shape, q_target.shape, q_expected.shape, rewards.shape)

        td_error = q_target - q_expected
        # if max_reward > 0.01:
        self.debug_values(q_next, q_target, q_expected, td_error, rewards, dones)
        self.debug_update_qr(q_expected, rewards,td_error)

        critic_loss = F.mse_loss(q_expected, q_target)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        # print('Agent learn Shapes : ',self.id,actions_pred.shape, actions.shape)
        # Add actions pred in the agent specific part of actions_next
        # print('pred : ',actions_pred[1:4,:])
        if self.id == 0:
            actions_pred = torch.cat([actions_pred,actions[:,self.action_size:2*self.action_size]],dim=1)
        else:
            actions_pred = torch.cat([actions[:,0:self.action_size], actions_pred],dim=1)
        # print('actions : ',actions[1:4,:])
        # print('cat : ',actions_pred[1:4,:])
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        td_error = td_error.detach().numpy()
        return al,cl

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    # Some debug functions
    def debug_values(self, q_next, q_target, q_expected, td_error, rewards, dones):
        if self.Do_debug_values and (self.episode % 20 == 0):
            print('Found reward :')
            # print(type(q_next))
            # print(type(q_target))
            # print(type(q_expected))
            # print(q_next.shape)
            # print(q_target.shape)
            # print(q_expected.shape)
            # print(td_error.shape)
            # print(rewards.shape)
            # print(dones.shape)
            torch.set_printoptions(precision=4,threshold=10_000, sci_mode = False, linewidth = 120)
            print('q_next, q_target, q_expected, td_error')
            qlist = torch.cat([q_next,q_target,q_expected,td_error,rewards,dones],1)
            print(qlist)
        
    def debug_update_qr(self, q_expected, rewards,td_error):
        if self.Do_debug_qr and ((self.episode % 10 == 0) or (torch.max(rewards) > 0.01)):
            print('--------------Agent Learn------------------------')
            print('Agent {} and episode {} '.format(self.id, self.episode))
            print('update - q expected : mean : {:6.4f} - sd : {:6.4f} min-max {:6.4f}|{:6.4f}'.format(torch.mean(q_expected),torch.std(q_expected),torch.min(q_expected),torch.max(q_expected)))
            if torch.max(rewards) > 0.01:
                print(Back.GREEN+'update - reward : mean : {:6.4f} - sd : {:6.4f} min-max {:6.4f}|{:6.4f}'.format(torch.mean(rewards),torch.std(rewards),torch.min(rewards),torch.max(rewards)))
                print(Style.RESET_ALL, end='')                
            else:
                print('update - reward : mean : {:6.4f} - sd : {:6.4f} min-max {:6.4f}|{:6.4f}'.format(torch.mean(rewards),torch.std(rewards),torch.min(rewards),torch.max(rewards)))
            
            abs_td_error = torch.abs(td_error)
            print('update - TD-Error : mean : {:6.4f} - sd : {:6.4f} min-max {:6.4f}|{:6.4f}'.format(torch.mean(td_error),torch.std(td_error),torch.min(td_error),torch.max(td_error)))
            print('update - abs-TD-Error : mean : {:6.4f} - sd : {:6.4f} min-max {:6.4f}|{:6.4f}'.format(torch.mean(abs_td_error),torch.std(abs_td_error),torch.min(abs_td_error),torch.max(abs_td_error)))

            
    def save_agent(self,i_episode):
        filename = 'trained_reacher_e'+str(i_episode)+'.pth'
        torch.save({
            'critic_local': self.critic_local.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_local': self.actor_local.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            }, filename)
        
        print('Saved Networks in ',filename)
        return
        
    def load_agent(self,filename):
        savedata = torch.load(filename)
        self.critic_local.load_state_dict(savedata['critic_local'])
        self.critic_target.load_state_dict(savedata['critic_target'])
        self.actor_local.load_state_dict(savedata['actor_local'])
        self.actor_target.load_state_dict(savedata['actor_target'])
        return