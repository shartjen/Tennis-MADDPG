import numpy as np
import progressbar as pb
from collections import deque
import time
import random

from colorama import Back, Style
import torch
from ddpg_agent import ddpg_agent
from Buffer import ReplayBuffer

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
# Conversion from numpy to tensor
def ten(x): return torch.from_numpy(x).float().to(device)

# empty class to use like Matlab struct
class struct_class(): pass
    
class maddpg():
    """Interacts with and learns from the environment."""
    
    def __init__(self, env, config):
        """Initialize an Agent object.
        
        Params
        ======
            env : environment to be handled
            config : configuration given a variety of parameters
        """
        
        
        self.env = env
        self.config = config
        # self.seed = (config['seed'])

        # set parameter for ML
        self.set_parameters(config)
        # Replay memory
        self.memory = ReplayBuffer(config)
        # Q-Network
        self.create_agents(config)
        # load agent
        if self.load_model:
            self.load_agent('trained_tennis_2k86.pth')
    
    def set_parameters(self, config):
        # Base agent parameters
        self.gamma = config['gamma']                    # discount factor 
        self.tau = config['tau']
        self.max_episodes = config['max_episodes']      # max numbers of episdoes to train
        self.env_file_name = config['env_file_name']    # name and path for env app
        self.brain_name = config['brain_name']          # name for env brain used in step
        self.train_mode = config['train_mode']
        self.load_model = config['load_model']
        self.save_model = config['save_model']
        self.num_agents = config['num_agents']
        self.state_size = config['state_size']
        self.action_size = config['action_size']
        self.hidden_size = config['hidden_size']
        self.buffer_size = config['buffer_size']
        self.batch_size = config['batch_size']
        self.learn_every = config['learn_every']
        self.learn_num = config['learn_num']
        self.critic_learning_rate = config['critic_learning_rate']
        self.actor_learning_rate = config['actor_learning_rate']
        self.noise_decay = config['noise_decay']
        self.seed = (config['seed'])
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.noise_scale = 1
        self.results = struct_class()
        # Some Debug flags
        self.debug_show_memory_summary = False
        
    def create_agents(self, config):
        self.maddpg_agent = [ddpg_agent(config), 
                             ddpg_agent(config)]
        
        for a_i in range(self.num_agents):
            self.maddpg_agent[a_i].id = a_i
        
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        # print('Step adding types') # : ,states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape)
        actions = np.reshape(actions,(1,2*self.action_size))
        self.memory.add(states, actions, rewards, next_states, dones)
                    

    def act(self, state):
        """Returns actions for given state as per current policy 
        shuold only get single or single joined states from train"""
        state = ten(state)
        actions = np.vstack([agent.act(state) for agent in self.maddpg_agent])
        return actions
    
    def actor_target(self, states):
        """Returns actions for given state as per current target_policy without noise.
           should only get batch_size states from learn"""
        actions = np.hstack([agent.act(states) for agent in self.maddpg_agent])
        return ten(actions)

    def init_results(self):
        """ Keeping different results in list in self.results, being initializd here"""
        self.results.reward_window = deque(maxlen=100)
        self.results.all_rewards = []
        self.results.avg_rewards = []
        self.results.critic_loss = []
        self.results.actor_loss = []

    def episode_reset(self, i_episode):
        self.noise_reset()
        self.episode = i_episode
        self.noise_scale *= self.noise_decay
        for agent in self.maddpg_agent:
            agent.noise_scale = self.noise_scale
            agent.episode = self.episode
        
    def noise_reset(self):
        for agent in self.maddpg_agent:
            agent.noise.reset() 

    def train(self):
        print('Running on device : ',device)
        # if False:
        #     filename = 'trained_reacher_a_e100.pth'
        #     self.load_agent(filename)
        self.init_results()
        # training loop
        # show progressbar
        widget = ['episode: ', pb.Counter(),'/',str(self.max_episodes),' ', 
                  pb.Percentage(), ' ', pb.ETA(), ' ', pb.Bar(marker=pb.RotatingMarker()), ' ' ]
        
        timer = pb.ProgressBar(widgets=widget, maxval=self.max_episodes).start()

        for i_episode in range(self.max_episodes): 
            timer.update(i_episode)
            tic = time.time()

            # per episode resets
            self.episode_reset(i_episode)
            total_reward = np.zeros(self.num_agents)
            # Reset the enviroment
            env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
            states = self.get_states(env_info)
            t = 0
            dones = np.zeros(self.num_agents, dtype = bool)

            # loop over episode time steps
            while not any(dones):
                # act and collect data
                actions = self.act(states)
                env_info = self.env.step(actions)[self.brain_name]
                next_states = self.get_states(env_info)
                rewards = env_info.rewards
                dones = env_info.local_done
                # increment stuff
                t += 1
                total_reward += rewards
                # np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
                # print('Episode {} step {} taken action {} reward {} and done is {}'.format(i_episode,t,actions,rewards,dones))
                # Proceed agent step
                self.step(states, actions, rewards, next_states, dones)
                # prepare for next round
                states = next_states
            #:while not done
            # Learn, if enough samples are available in memory
            if (i_episode % self.learn_every == 0):
                if len(self.memory) > self.batch_size:
                    for l in range(self.learn_num):
                        experiences = self.memory.sample()
                        self.learn(experiences)
                    
            toc = time.time()
            # keep track of rewards:
            self.results.all_rewards.append(total_reward)
            self.results.avg_rewards.append(np.mean(self.results.reward_window))
            self.results.reward_window.append(np.max(total_reward))
            # Output Episode info : 
            self.print_episode_info(total_reward,t,tic,toc)
        # for i_episode
        
        if self.save_model:        
            filename = 'trained_tennis'+str(self.seed)+'.pth'
            self.save_agent(filename)
            
        return self.results
    
    def get_states(self, env_info):
        return np.reshape(env_info.vector_observations,(1,2*self.state_size))
    
    def print_episode_info(self,total_reward, num_steps, tic, toc):
        if (self.episode % 20 == 0) or (np.max(total_reward) > 0.01):
            if np.max(total_reward) > 0.01:
                if np.sum(total_reward) > 0.15:
                    if np.sum(total_reward) > 0.25:
                        StyleString = Back.GREEN
                        print('Double Hit')
                    else:
                        StyleString = Back.BLUE
                else:
                    StyleString = Back.RED
            else:
                StyleString = ''
            print(StyleString + 'Episode {} with {} steps || Reward : {} || avg reward : {:6.3f} || Noise {:6.3f} || {:5.3f} seconds, mem : {}'.format(self.episode,num_steps,total_reward,np.mean(self.results.reward_window),self.noise_scale,toc-tic,len(self.memory)))
            print(Style.RESET_ALL, end='')                
        
            
    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        q_target = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value """

        states, actions, rewards, next_states, dones = experiences
        # print('Learning shape : ',states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape)
        # print('Learning state & reward shape : ',states[0].shape,rewards[0].shape)
        
        actor_loss = []
        critic_loss = []
        both_next_actions = self.actor_target(next_states)
            
        # print('Learn both',both_next_actions.shape)
        for agent in self.maddpg_agent:
            # In case of joined_states, we want actions_next from both agents for learning
            al, cl = agent.learn(states, actions, rewards, next_states, both_next_actions, dones)
            actor_loss.append(al)
            critic_loss.append(cl)
            
        self.results.actor_loss.append(actor_loss)
        self.results.critic_loss.append(critic_loss)

    def save_agent(self,filename):
        states, actions, rewards, next_states, dones = self.memory.save_buffer()
        print('save agent : ',states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape)
        torch.save({
            'critic_local0': self.maddpg_agent[0].critic_local.state_dict(),
            'critic_target0': self.maddpg_agent[0].critic_target.state_dict(),
            'actor_local0': self.maddpg_agent[0].actor_local.state_dict(),
            'actor_target0': self.maddpg_agent[0].actor_target.state_dict(),
            'critic_local1': self.maddpg_agent[1].critic_local.state_dict(),
            'critic_target1': self.maddpg_agent[1].critic_target.state_dict(),
            'actor_local1': self.maddpg_agent[1].actor_local.state_dict(),
            'actor_target1': self.maddpg_agent[1].actor_target.state_dict(),
            'memory': (states, actions, rewards, next_states, dones),
            }, filename)
        print('Saved Networks and ER-memory in ',filename)
        return
        
    def load_agent(self,filename):
        savedata = torch.load(filename)
        self.maddpg_agent[0].critic_local.load_state_dict(savedata['critic_local0'])
        self.maddpg_agent[0].critic_target.load_state_dict(savedata['critic_target0'])
        self.maddpg_agent[0].actor_local.load_state_dict(savedata['actor_local0'])
        self.maddpg_agent[0].actor_target.load_state_dict(savedata['actor_target0'])
        self.maddpg_agent[1].critic_local.load_state_dict(savedata['critic_local1'])
        self.maddpg_agent[1].critic_target.load_state_dict(savedata['critic_target1'])
        self.maddpg_agent[1].actor_local.load_state_dict(savedata['actor_local1'])
        self.maddpg_agent[1].actor_target.load_state_dict(savedata['actor_target1'])
        states, actions, rewards, next_states, dones = savedata['memory']
        self.memory.load_buffer(states, actions, rewards, next_states, dones)
        print('Memory loaded with length : ',len(self.memory))
        return
