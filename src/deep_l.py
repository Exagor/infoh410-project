import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D
from keras.optimizers import Adam
import csv
import os

class deep_QN:

    def __init__(self, env, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.1, epsilon_max=1.0, 
                 epsilon_interval=0.0001, batch_size=32, max_episodes=1000,
                 filename="deep_QN.weights.h5",
                 score_file="score.csv"):
        self.env = env
        self.height, self.width = self.env.observation_space.shape[0], self.env.observation_space.shape[1]
        self.channels = 1 #because grayscale
        self.actions = self.env.action_space.n
        self.filename = filename
        self.score_file = score_file
        self.model = self.build_model()
        self.model_target = self.build_model()
        self.max_steps_per_episode = 10000
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_interval = epsilon_interval
        self.batch_size = batch_size
        self.max_episodes = max_episodes
        

    def build_model(self):
        model = Sequential() #initialize the model
        # Convolutional layers of the model
        model.add(Convolution2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(self.height, self.width, self.channels)))
        model.add(Convolution2D(64, (4,4), strides=(2,2), activation='relu'))
        model.add(Convolution2D(64, (3,3), activation='relu'))
        # Flatten the model to convert it to a 1D array
        model.add(Flatten())
        # Dense layers of the model, add fully connected layers
        model.add(Dense(512, activation='relu'))
        # Final layer with the actions
        model.add(Dense(6, activation='linear'))

        if os.path.exists(self.filename):
            model.load_weights(self.filename)

        return model
    
    def save_scores(self, filename, score):
        with open(filename, mode='a',newline='') as file:
            writer = csv.writer(file)
            writer.writerow([score])

    def train(self):
        # Used to facilitate the backpropagation
        optimizer = Adam(learning_rate=0.00025, clipnorm=1.0) 

        # Experience replay buffers
        action_history = []
        state_history = []
        state_next_history = []
        rewards_history = []
        done_history = []
        episode_reward_history = []
        running_reward = 0
        episode_count = 0
        frame_count = 0
        # Number of frames to take random action and observe output
        epsilon_random_frames = 5000
        # Number of frames for exploration
        epsilon_greedy_frames = 100000.0
        # Maximum replay length
        # Note: The Deepmind paper suggests 1000000 however this causes memory issues
        max_memory_length = 100000
        # Train the model after 4 actions
        update_after_actions = 4
        # How often to update the target network
        update_target_network = 10000
        # Using huber loss for stability
        loss_function = keras.losses.Huber()

        while True:
            observation, _ = self.env.reset()
            state = np.array(observation)
            episode_reward = 0

            for timestep in range(1, self.max_steps_per_episode):
                frame_count += 1

                # Use epsilon-greedy for exploration
                if frame_count < epsilon_random_frames or self.epsilon > np.random.rand(1)[0]:
                    # Take random action
                    action = np.random.choice(self.actions)
                else:
                    # Predict action Q-values
                    # From environment state
                    state_tensor = keras.ops.convert_to_tensor(state)
                    state_tensor = keras.ops.expand_dims(state_tensor, 0)
                    action_probs = self.model(state_tensor, training=False)
                    # Take best action
                    action = keras.ops.argmax(action_probs[0]).numpy()

                # Decay probability of taking random action
                self.epsilon -= self.epsilon_interval / epsilon_greedy_frames #decrease of small value
                self.epsilon = max(self.epsilon, self.epsilon_min)

                # Apply the sampled action in our environment
                state_next, reward, done, _, _ = self.env.step(action)
                state_next = np.array(state_next)

                episode_reward += reward

                # Save actions and states in replay buffer
                action_history.append(action)
                state_history.append(state)
                state_next_history.append(state_next)
                done_history.append(done)
                rewards_history.append(reward)
                state = state_next

                # Update every fourth frame and once batch size is over 32
                if frame_count % update_after_actions == 0 and len(done_history) > self.batch_size:
                    # Get indices of samples for replay buffers
                    indices = np.random.choice(range(len(done_history)), size=self.batch_size)

                    # Using list comprehension to sample from replay buffer
                    state_sample = np.array([state_history[i] for i in indices])
                    state_next_sample = np.array([state_next_history[i] for i in indices])
                    rewards_sample = [rewards_history[i] for i in indices]
                    action_sample = [action_history[i] for i in indices]
                    done_sample = keras.ops.convert_to_tensor(
                        [float(done_history[i]) for i in indices]
                    )

                    # Build the updated Q-values for the sampled future states
                    # Use the target model for stability
                    future_rewards = self.model_target.predict(state_next_sample,verbose=0) #to silent the output
                    # Q value = reward + discount factor * expected future reward
                    updated_q_values = rewards_sample + self.gamma * keras.ops.amax(
                        future_rewards, axis=1
                    )

                    # If final frame set the last value to -1
                    # because there's no reward after the last frame
                    # to avoid the model to go to the end of episode
                    updated_q_values = updated_q_values * (1 - done_sample) - done_sample 

                    # Create a mask so we only calculate loss on the updated Q-values
                    masks = keras.ops.one_hot(action_sample, self.actions)

                    with tf.GradientTape() as tape:
                        # Train the model on the states and updated Q-values
                        q_values = self.model(state_sample)

                        # Apply the masks to the Q-values to get the Q-value for action taken
                        q_action = keras.ops.sum(keras.ops.multiply(q_values, masks), axis=1)
                        # Calculate loss between new Q-value and old Q-value
                        loss = loss_function(updated_q_values, q_action)

                        # Backpropagation #TODO: check if the indentation is correct
                        grads = tape.gradient(loss, self.model.trainable_variables)
                        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                if frame_count % update_target_network == 0:
                    # update the the target network with new weights
                    self.model_target.set_weights(self.model.get_weights())
                    # Log details
                    template = "running reward: {:.2f} at episode {}, frame count {}"
                    print(template.format(running_reward, episode_count, frame_count))

                # Limit the state and reward history
                if len(rewards_history) > max_memory_length:
                    del rewards_history[:1]
                    del state_history[:1]
                    del state_next_history[:1]
                    del action_history[:1]
                    del done_history[:1]

                if done:
                    break

            # Save the score of the episode
            self.save_scores(self.score_file, episode_reward)

            # Save weights of the model after each episode
            self.model_target.save_weights(self.filename)

            # Update running reward to check condition for solving
            episode_reward_history.append(episode_reward)
            if len(episode_reward_history) > 100:
                del episode_reward_history[:1]
            running_reward = np.mean(episode_reward_history)

            episode_count += 1

            print("Episode: ", episode_count, "Epsilon: ", self.epsilon)
            if (
                self.max_episodes > 0 and episode_count >= self.max_episodes
            ):  # Maximum number of episodes reached
                print("Stopped at episode {}!".format(episode_count))
                break

    def __del__(self):
        self.env.close()