from gameEnv import CarRacingEnv
from ple import PLE
import time
import numpy as np
import random
from keras.layers import Dense, Reshape
from keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from collections import deque

from utils import process_state

learning_rate = 0.001
epsilon = 0.99
DISCOUNT               = 0.97
REPLAY_MEMORY_SIZE     = 3000   # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 500
MINIBATCH_SIZE = 128
target_update_steps = 10

render = True

game = CarRacingEnv()
possible_actions = game.actions

p = PLE(game, fps=30, state_preprocessor=process_state, display_screen=render)
p.init()

# Take default action in the game
p.act(p.NOOP)
observation = p.getGameState()

def select_action(epsilon, observation):
    # return np.random.choice(2, 1)[0]
    r = np.random.random()
    if r > epsilon:
        a_probs = agent.model.predict(np.reshape(observation, (1, 7, 1))).flatten()
        action = np.argmax(a_probs)
    else:
        # Get random action between 0,1
        action = np.random.choice(3, 1)[0]
    return action

# https://github.com/ultronify/cartpole-tf-dqn/blob/master/dqn_agent.py
class DQNAgent:
    def __init__(self, env):
        self.env = env

        # Main Model
        self.model = self.build_model()

        # Target network
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0


    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=7, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(3, activation="softmax"))
        model.compile(loss="mse", metrics=["accuracy"], optimizer= Adam(learning_rate=learning_rate))
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal_state, step ):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        X = []
        Y = []
        
        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # Except for the last episode, look into the future
            
            current_q = self.model.predict_on_batch(np.reshape(current_state, (1, 7, 1))) # This has 2 q-values (for both the action was taken and not)
            
            # target = reward
            if not done:
            #     # max_future_q = np.max(future_qs_list[index])
                val = self.target_model.predict_on_batch(np.reshape(new_current_state, (1, 7, 1))).flatten()
                target = reward + DISCOUNT * np.amax(val)
            else:
                target = reward
            # The DQN agent outputs q values for all the actions, select the q value for action that is taken
            current_q[0][action] = np.array(target) # update the q value for the action that was taken

            X.append(current_state)
            Y.append(current_q[0])

        states_batch = np.array(X)
        q_values_batch = np.array(Y)


        self.model.train_on_batch(states_batch, q_values_batch)
        
        # , sample_weight=advantage.flatten())
        # , epochs=1, verbose=0)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1
            
    def load(self, path):
        self.model = load_model(path)
        self.target_model.set_weights(self.model.get_weights()) 


agent = DQNAgent(p)
# agent.load("best_model_pole.h5")

EPISODES=1000
MIN_EPSILON=0.001
EPSILON_DECAY=0.995


step = 1 
reward_collected = []
#  Stats to be collected each episode
total_score = 0
timestep = 0
# Iterate over episodes
for episode in range(1, EPISODES+1):
    
    p.reset_game()
    p.act(p.NOOP)
    current_state = p.getGameState()
    timestep = 0
    
    count = 0
    game_over = p.game_over()
    while not game_over:
        
        action = select_action(epsilon, current_state)
        p.act(possible_actions[action])
        new_state = p.getGameState()
        game_over = p.game_over()
        r = p._getReward()
        score = p.score()

        # Every step we update replay memory and train main network
        agent.update_replay_memory((observation, action, r, new_state, game_over))
        if count%20 == 0:
            agent.train(game_over, step)

        current_state = new_state
        step += 1
        timestep+=1 

        # reward_sum = 0
        if episode % 10 == 0:
            agent.model.save('model_pole.h5')

    # Copy weights of dqn agent to target agent every target_update_steps
    if agent.target_update_counter % target_update_steps == 0: 
        agent.target_model.set_weights(agent.model.get_weights()) 

    
    # Decay epsilon
    print(f"Exploration rate = {round(epsilon,3)},", end = "\t")
    epsilon = max(MIN_EPSILON, epsilon*EPSILON_DECAY)
  
    print(f"Episode {episode} completed with {timestep} timesteps, score = {score}", end = '\n')
    reward_collected.append(score)
    

    # if len(last_n_scores) >= 1:
        # print("Average score: " + str(np.average(last_n_scores[0])))
    with open("history_racing.txt", "a+") as data:
        data.write(str(episode) + "," +  str(epsilon) + ", " + str(score) + ", " + str(time.time()) + "\n")
    