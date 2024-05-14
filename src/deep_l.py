import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
import csv
from stable_baselines3 import DQN
from stable_baselines3.common.envs import Monitor
from stable_baselines3.dqn.policies import CnnPolicy
from stable_baselines3.common.callbacks import CheckpointCallback


class deep_rl:

    def __init__(self, env):
        self.env = env
        self.height, self.width, self.channels = self.env.observation_space.shape
        self.actions = self.env.action_space.n
        self.model = self.build_model(self.height, self.width, self.channels, self.actions)
        self.dqn = self.build_agent(self.model, self.actions)
        self.dqn.compile(Adam(lr=1e-4))

    def build_model(height, width, channels, actions):
        model = Sequential() #initialize the model
        # Convolutional layers of the model
        model.add(Convolution2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(3,height, width, channels)))
        model.add(Convolution2D(64, (4,4), strides=(2,2), activation='relu'))
        model.add(Convolution2D(64, (3,3), activation='relu'))
        # Flatten the model to convert it to a 1D array
        model.add(Flatten())
        # Dense layers of the model, add fully connected layers
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        # Final layer with the actions
        model.add(Dense(actions, activation='linear'))
        return model
    
    def build_agent(model, actions):
        #create the policy
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=10000)
        #create the memory
        memory = SequentialMemory(limit=1000, window_length=3)
        #create the deep Q network agent
        dqn = DQNAgent(model=model, memory=memory, policy=policy,
                    enable_dueling_network=True, dueling_type='avg', 
                    nb_actions=actions, nb_steps_warmup=1000
                    )
        return dqn
    
    def train(self, iterations=1000):
        self.dqn.fit(self.env, nb_steps=iterations, visualize=False, verbose=2)

    def print_score(self):
        self.dqn.test(self.env, nb_episodes=5, visualize=True)
        self.dqn.save_weights('dqn_weights.h5f')

    def save_scores(self, filename):
        scores = self.dqn.test(self.env, nb_episodes=5, visualize=True)
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Row Number', 'Score'])
            for i, score in enumerate(scores):
                writer.writerow([i+1, score])

    def __del__(self):
        self.env.close()

class DeepRL:
    def __init__(self, env):
        self.env = Monitor(env)
        self.model = DQN(CnnPolicy, self.env, verbose=1, buffer_size=10000, learning_rate=1e-4, exploration_final_eps=0.1, exploration_initial_eps=1.0, target_update_interval=1000)

    def train(self, iterations=1000):
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./weights/', name_prefix='dqn_weights')
        self.model.learn(total_timesteps=iterations, callback=checkpoint_callback)

    def print_score(self):
        mean_reward, std_reward = self.model.evaluate_policy(self.model.policy, self.env, n_eval_episodes=5)
        print(f"Mean reward: {mean_reward} +/- {std_reward}")

    def save_scores(self, filename):
        mean_reward, std_reward = self.model.evaluate_policy(self.model.policy, self.env, n_eval_episodes=5)
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Row Number', 'Score'])
            for i, score in enumerate([mean_reward, std_reward]):
                writer.writerow([i+1, score])

    def __del__(self):
        self.env.close()