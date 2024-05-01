import random as rng

class agent:
    """
    Represents an agent that interacts with the game

    Attributes:
    - Q Table
    - the environment
    - epsilon
    - environment
    - gamma (discount factor)
    - alpha (learning rate)
    """
    def __init__(self, epsilon, env, gamma=0.95, alpha=0.1):
        self.epsilon = epsilon #epsilon greedy
        self.gamma = gamma #discount factor
        self.alpha = alpha #learning rate
        self.env = env
        self.Q_table = dict()

    def convert_state(self, state):
        """
        Convert the state to a string

        Parameters:
        - state: the state

        Returns:
        - the state as a string
        """
        str_state = ""
        for i in state:
            for j in i:
                str_state += str(i)+" "
        return str_state
    
    def get_action(self, state):
        """
        Get the action based on the state

        Parameters:
        - state: the state
        - env: the environment

        Returns:
        - the action
        """
        if rng.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            str_state = self.convert_state(state)
            return max(self.Q_table[str_state])
        
    def update_Q_table(self, state, action, reward, next_state):
        """
        Update the Q table based on the state, action, reward, and next state

        Parameters:
        - state: the state
        - action: the action
        - reward: the reward
        - next_state: the next state
        - env: the environment
        """
        str_state = self.convert_state(state)
        str_next_state = self.convert_state(next_state)
        # Initialize the Q table if the state or next state is not in the Q table
        if str_state not in self.Q_table:
            self.Q_table[str_state] = dict()
        if str_next_state not in self.Q_table:
            self.Q_table[str_next_state] = dict()
        if action not in self.Q_table[str_state]:
            self.Q_table[str_state][action] = 0.0
        if action not in self.Q_table[str_next_state]:
            self.Q_table[str_next_state][action] = 0.0
        
        self.Q_table[str_state][action] += self.alpha * (reward + self.gamma * max(self.Q_table[str_next_state].values()) - self.Q_table[str_state][action])

    def train(self, episodes=1000):
        """
        Train the agent

        Parameters:
        - env: the environment
        - episodes: the number of episodes
        """
        for episode in range(episodes):
            self.state = self.env.reset()
            self.play_a_game(train=True)
    
    def play_a_game(self, train=False):
        """
        Play a game
        """
        done = False
        while not done:
            action = self.get_action(self.state)
            next_state, reward, done, _, _ = self.env.step(action)
            print(self.state)
            print(next_state)
            if train:
                self.update_Q_table(self.state, action, reward, next_state)
            self.state = next_state

    def __del__(self):
        self.env.close()