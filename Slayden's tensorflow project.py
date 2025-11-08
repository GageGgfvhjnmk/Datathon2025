'''
-----SOURCES-----
https://www.analyticsvidhya.com/blog/2019/03/deep-learning-frameworks-comparison/
https://www.youtube.com/watch?v=-TTziY7EmUA
https://www.tensorflow.org/api_docs/python/tf
https://medium.com/mind-magazines/how-policies-are-represented-by-neural-networks-98677f35bad8
https://www.tensorflow.org/tensorboard/image_summaries


-----ABOUT NEURAL NETWORKS-----
TensorFlow is the most popular deep learning framework, it is used in Python, C++, JavaScript, and R. 
TensorBoard helps in data visualization using data flow graphs. This can help you see 
what your model looks like.
TensorFlow is useful in the following areas:
    Text-based applications: Language detection, text summarization
    Image recognition: Image captioning, face recognition, object detection
    Sound recognition
    Time series analysis
    Video analysis

Keras runs on top of TensorFlow, it easily manages what TensorFlow does not. Since TensorFlow
is such a low-level platform, it is difficult to learn, but Keras takes care of that. It abstracts 
the low level libraries, namely Theano and TensorFlow. It
handles CNNs and RNNs well. 
Keras has many architectures:
    VGG16
    VGG19
    InceptionV3
    Mobilenet, and many more

Sequential models the architecture of the networks, since most networks contain similar 
layers, they are stored easily with the Sequential library

Gym is an environment suite; a library that represents the game. Related libraries are:
    Atari
    Mujoco
    PyBullet
    DM_Control
These are often used to set up a parallel environment for efficient training

A general guideline for creating a neural network:
    Check if it is a problem where Neural Network gives you uplift over traditional algorithms (other machine learning algorithms or programs)
    Do a survey of which Neural Network architecture is most suitable for the required problem
    Define Neural Network architecture through whichever language / library you choose.
    Convert data to right format and divide it in batches
    Pre-process the data according to your needs
    Augment Data to increase size and make better trained models
    Feed batches to Neural Network
    Train and monitor changes in training and validation data sets
    Test your model, and save it for future use

Many types of networks include:
    Regressors (Classifiers)
    Generators
    Policies
Here, a policy network is the best fit because it takes the state of a game and outputs an action.
Policy networks use reinforcement learning (RL) to promote good actions and punish bad actions.
By contrast, regression models try to minimize a loss function by outputting a continuous value, whereas
    RL uses distinct actions and its consequential rewards.
How this looks in the network: The state of the game is expressed in the input layer, where each neuron holds a variable
    such as (x/y position, enemy position, score, etc.) The neurons in the output layer represent each action (w, a, s, d).
With RL, we often define an agent (the character), which holds the main RL algorithms
    (training and batch reading), holds a policy, and can use many algorithms:
    DQN, DDQN, DQN-RNN
    DDPG, TD3
    PPO, PPO-RNN
    REINFORCE
    SAC
    Behavioral Cloning
The system runs in a loop:
    Environment (driver.run() OR TFPyEnvironment(ParallelPyEnvironment([...]))) 
    Driver (driver = DynamicStepDriver())
    Trajectory
    Observer (rb.add_batch())
    Replay Buffer (rb = TfUniformReplayBuffer())
    tf.data.Dataset (rb.as_dataset())
    Batch
    Agent (agent = DqnAgent())
    Neural Network (agent.train(), QNetwork()) 
    Collect Policy (agent.collect_policy)
    back to Driver
The Policy is the last half of that list (Data -> Collect Policy). Policies take in
    parameters from the observation space, and emits a distribution of actions. 
    The fundamental method is _distribution() 
The Environment is the basis of everything, the Driver and Policy interact with
    that Environment, those interactions are stored in the Replay Buffer, which is
    read with tf.data.Dataset, the Agent is trained with batches from that 
    dataset, the Network underlying the Policy is updated from the Agent


-----QUESTIONS-----
What is verbose?
    Means wordy, or referring to going into more detail. Typically has an index to 
    specify how much output you want.
What is a batch? 
    A subset of data. Within an epoch. The number of 
    batches in an epoch is determined by the batch size. (1-N)
What is an epoch? 
    One full pass through the entire dataset. (1-inf)
What is an iteration? 
    The process of passing one batch of data through the model, calculating 
    the loss, and updating the model's parameters
Why use more than 1 epoch, shouldn't it be enough to see each sample once? 
    Learns iteratively, 
How does a model file differ in format?
    *.keras is generally what is used here
How do I save and load models?
    model.save(filename)
    loadmodel(filename)
What is model.fit()?
    This method is how it learns, it uses the optimizer and backpropagation to update 
    the values in the network
What are all the variables that I could build this with?
    (Comments section)
What is a tensor? 
    An algebraic object that describes a multi-linear relationship between 
    sets of algebraic objects related to a vector space
What is a config?
    Contains the optimizer, loss, metrics. 
What is a model?
    Contains many things:
    1) Architecture (Configuration) - Layer structure
    2) Weights - Specific values (State of the model)
    3) Optimizer - How it learns: Here it is Adam
    4) Losses and Metrics - 
What is argmax()?
    It finds the max value along the specified axis of an array.
    So axis 0 (down) would return the indices (size: num columns) of the max values in
    each column
What is model.predict()?
    It just runs through the model, as I'd expect the main intention is.
    Computation is done in batches. This method is designed for batch processing of large numbers 
    of inputs. It is not intended for use inside of loops that iterate over your data and 
    process small numbers of inputs at a time.
    Takes in an array of states
How do I batch process?
    model.predict is designed for batch processing from the replay buffer, just store all states and
    Q-values in a np array.
What does loss represent?
    Loss is how well it performs
What does reward represent and how does it affect training?
    Reward is given based on the state.
How do I effectively give rewards?
    Rewards should be given positively anywhere and everywhere you want to "congratulate" the agent.
    Punishments (negative rewards) should be given anytime and every time I want to say "NO, BAD" to the agent
    These should each have their own values based on the severity of the state. The rewards will be summed and 
    transferred to the model via model.fit().
    Note: Because they are simply summed, severity is important. If the agent were to go out of bounds, get 
    stuck, but also kill a bunch of enemies, the reward might sum to zero. At which point is thinks "this is 
    fine, don't change anything." Which might be bad if it happens often
What is Q-Learning?
    Q-Learning creates an exact matrix for the working agent which it can “refer to” to maximize its 
    reward in the long run. This is only practical in small environments because each value only has
    important with respect to other values. 
    Determines the optimal action in any given state.
What is Deep Q-Learning (DQN - networks)?
    Uses a deep neural network to approximate the Q-function. The initial state is fed into the 
    neural network and it returns the Q-value of all possible actions as an output.
    One of the key challenges in implementing Deep Q-Learning is that the Q-function 
    is typically non-linear and can have many local minima. This can make it difficult for 
    the neural network to converge to the correct Q-function. To address this, several 
    techniques have been proposed, such as experience replay and target networks.
What do Q-values represent?
    The action probabilities taken for a specific state
How do I make a policy?
    The policy here is the whole goal, it is the neural network specifically learning from its 
    environment to take in a state, and spit out an action
What is a replay buffer?
    A buffer used to track actions for the network and agent to learn from later. This 
    helps to decorrelate the data and make the learning process more stable
How do I decide the target values?
    The target value for a given state-action pair is computed with the Bellman equation:
        y = r + g * max(Q(s', a'))
        r is the immediate reward
        g is gamma, the discount factor
        Q(s', a') is the Q-value
        max(Q(s', a')) is the highest expected Q-value for the next state s'
    However, there's a missing piece: you're not selecting the Q-value of the action
        actually taken. Instead, you're always taking the max Q-value of the next state, 
        which is correct for DQN, but not for general Q-learning.
How do I fix getting stuck in a loop:
    If the RL agent is getting stuck in a loop, it likely means it's learning a suboptimal 
    policy that cycles between two (or a few) states
    Solutions:
        Penalize looping
        Penalize exceeding the time constraint
        Increase exploration
        Reward progress
        Use a target network - Update a secondary network every few episodes to smooth learning


-----COMMENTS-----
Variables:
    input size
    output size
    each hidden layer size
    num hidden layers
    num episodes
    num epochs
    batch size
    learning rate
    memory size
    gamma
    epsilon
    minimum epsilon
    epsilon decay
    reward/penalty
    verbose
    validation loss tolerance?
    observation space
    action space
    activation type (linear, relu)
    loss function type

Overfitting is when a model is so tuned to its training data
    it preforms poorly when exposed to new data
Keras is a standalone library as well as an extension to TensorFlow


-----GOALS-----
Answer everything with a question mark on this page - CHECK!
Save model to file - CHECK!
Apply model to more complex game
Search for more optimized methods
    Test out a tf.keras.layers.Conv2D as input layer (maybe a stupid idea)
Apply model to PacMan


-----INSTALLATIONS AND SYSTEM CHECKS-----
Run these in the terminal:
    pip install gym
    pip install tensorflow==2.18.0
    where python
    pip --version
    pip show tensorflow


-----MODULES-----
tf.keras - Keras designed to work alongside TensorFlow
tf.eager - For development and debugging
tf.function - Speeds everything up
gym - The environment the agent runs in
'''

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import gym
import numpy as np
import random
import datetime
from collections import deque
from gym import spaces
import tensorflow as tf

# Ignore the YELLOW Pylance errors they are lying!!!!
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.utils import plot_model # type: ignore

# Create a callback that saves the model's weights
checkpoint_path = "training_1/cp1.keras"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=False, verbose=1)

# STEP 1: Create the Number Game (Same as Before)
class NumberGame(gym.Env):
    def __init__(self):
        super(NumberGame, self).__init__()
        
        self.target = 50  # AI needs to reach this number
        self.state = np.random.randint(0, 100)  # Define as simple list. Start at a random number
        
        self.action_space = spaces.Discrete(2)  # Actions: [0] -1, [1] +1
        self.observation_space = spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32)
    
    def step(self, action):
        # PROBLEM: It has to reach the goal in order to review it, what if it never achieves the goal?
        # SOLUTION: Run a check on every possible way it can mess up, punish harshly and terminate if 
        #           it fails its task (i.e. going out of bounds, taking too long)
        if action == 0:
            self.state -= 1
        else:
            self.state += 1
        
        reward = 1 - abs(self.state - self.target) / 100  # Closer gets higher reward
        done = self.state == self.target  # Game ends if target is reached
        
        return np.array([self.state]), reward, done, {}
    
    def reset(self):
        self.state = np.random.randint(0, 100)
        return np.array([self.state])
    
    def render(self):
        print(f"Current Number: {self.state}")
    
    def close(self):
        pass

# STEP 2: Create a Neural Network for AI
def build_model():
    '''
    # Define a simple model
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
    '''
    
    run_from_config = 1
    
    if run_from_config:
        model = load_model(checkpoint_path)
    else:
        model = Sequential([ # Architecture of the network
            Dense(24, input_shape=(1,), activation="relu"),  # Hidden layer 1
            Dense(24, activation="relu"),  # Hidden layer 2
            Dense(2, activation="linear")  # Output: Predicts the best action (0 or 1)
        ])
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.1)) # How it is trained. What does mse stand for? Mean squared error, a type of loss for regression models
    model.summary() # Visual of the network
    # plot_model(model, to_file=checkpoint_path) # Figure out how to display visually
    
    return model

def train_ai(episodes=1000):
    env = NumberGame()  # Create the game
    model = build_model()  # Build the AI model
    target_model = build_model()
    
    memory = deque(maxlen=2000)  # Experience replay memory
    gamma = 0.9  # Discount factor
    
    # Exploration parameters
    epsilon = 1.0  # Initial exploration rate
    epsilon_min = 0.01  # Minimum exploration
    epsilon_decay = 0.995  # Decay rate
    
    batch_size = 32  # Training batch size
    
    # 🔹 Initialize TensorBoard
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        prev_state = None
        
        while not done:
            # 🔹 STEP 1: Choose Action (Explore or Exploit)
            if np.random.rand() < epsilon:  
                action = env.action_space.sample()  # Random action
            else:
                q_values = model.predict(np.array([state]), verbose=0)[0]  # Batch predict
                action = np.argmax(q_values)  # Choose best action
            
            # 🔹 STEP 2: Take Action and Observe Reward
            next_state, reward, done, _ = env.step(action)
            
            # Reward based on movement toward goal
            if abs(next_state - env.target) < abs(state - env.target):
                reward += 5  # Encourage getting closer
            elif not -50 < next_state < 150: # Out of bounds
                reward -= 20  # Extra penalty for going out of bounds
                print("OUT OF BOUNDS")
            else:
                reward -= 2  # Discourage moving away
            
            # Punishments (if applicable)
            if next_state == prev_state:
                reward -= 20  # Penalize being stuck in a loop
            
            memory.append((state, action, reward, next_state, done))  # Store experience
            
            prev_state = state
            state = next_state
            total_reward += reward
            
            # Exit if out of bounds or taken too long
            if (not -50 < next_state < 150) or len(memory) > 150:
                break
        
        # 🔹 STEP 3: Train AI Using Experience Replay
        loss = 0  # Track loss per episode
        if len(memory) > batch_size:  
            batch = random.sample(memory, batch_size)
            
            states, actions, rewards, next_states, dones = zip(*batch)  # Unzip batch into arrays
            states = np.array(states)
            next_states = np.array(next_states)
            
            # Copy model weights every few episodes
            if episode % 10 == 0:
                target_model.set_weights(model.get_weights())
            
            # Batch predict Q-values for both current & next states. Use the target model for stable learning
            q_values = target_model.predict(states, verbose=0)  
            next_q_values = target_model.predict(next_states, verbose=0)            
            
            # Compute target Q-values
            targets = q_values.copy()
            for i in range(batch_size):
                target = rewards[i]
                if not dones[i]:
                    target += gamma * np.max(next_q_values[i])
                targets[i][actions[i]] = target
            
            # Train model on batch & record loss
            history = model.fit(states, targets, epochs=50, verbose=0, batch_size=batch_size)
            loss = history.history['loss'][0]  # Extract loss
        
        # 🔹 STEP 4: Decay Exploration
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        if episodes % 51 == 0: # Reset every so ofter
            epsilon = 1.0
        
        # 🔹 STEP 5: Log Metrics in TensorBoard
        with summary_writer.as_default():
            tf.summary.scalar("Total Reward", total_reward, step=episode)
            tf.summary.scalar("Loss", loss, step=episode)
            tf.summary.scalar("Epsilon", epsilon, step=episode)
        
        print(f"Episode {episode+1}: Total Reward = {total_reward}, Loss = {loss:.4f}, Epsilon = {epsilon:.4f}")
    
    env.close()
    return model

'''
# STEP 3: Train the AI using Deep Q-Learning
def train_ai(episodes=1000):
    env = NumberGame()  # Create the game
    model = build_model()  # Build the AI model
    
    memory = deque(maxlen=2000)  # Store past experiences, first-in-first-out
    gamma = 0.9  # Discount factor (importance of future rewards)
    
    # Move more strategically over time
    # Why have random moves? To start off with some data, because we don't yet know where to go
    epsilon = 1.0  # Exploration rate (AI tries random moves)
    epsilon_min = 0.01  # Minimum exploration (AI stops random moves)
    epsilon_decay = 0.9990  # Slowly reduce random moves
    
    for episode in range(episodes):
        state = env.reset()  # Start a new game
        done = False
        total_reward = 0
        
        prev_state = 0
        while not done: # Break when reached target
            # 🔹 STEP 3.1: AI Chooses Action (Explore or Exploit)
            if np.random.rand() < epsilon: # Why only less than epsilon? Percentage of the time to make a random move, start as 100%
                action = env.action_space.sample()  # Random action (explore)
            else:
                action_values = model.predict(np.array([state]), verbose=0)  # Predict best action. How does this work? Predict based on what?? Predict based on the neural network output
                action = np.argmax(action_values)  # Choose best action (exploit). What is this purpose? 
            
            # 🔹 STEP 3.2: Apply Action and Get New State
            next_state, reward, done, _ = env.step(action) # Take a step based on action choice
            # print("Initial Reward:", reward)
            
            # Punish out of bounds
            if not -50 < next_state < 150:
                reward *= 2
                print("OUT OF BOUNDS")
            
            # Punish if stuck in a loop
            if next_state == prev_state:
                reward -= 10
            
            print("Iteration Reward:", reward)
            memory.append((state, action, reward, next_state, done)) # Store previous moves
            
            prev_state = state
            state = next_state
            total_reward += reward
            
            if not -50 < next_state < 150: # Break if out of bounds for too long
                "--*--*--*--*--*--*--*--*-- Took too long --*--*--*--*--*--*--*--*--"
                break
        
        # 🔹 STEP 3.3: Train the AI from Memory
        if len(memory) > 32:  # Only start training after storing enough experiences
            batch = random.sample(memory, 32) # Random 32-element list in the memory
            for state, action, reward, next_state, done in batch: # Review each previous move
                target = reward
                if not done:
                    # Reward based on each move in relation to the next
                    target += gamma * np.max(model.predict(np.array([next_state]), verbose=0)) # Figure out what this means
                    # print(model.predict(np.array([next_state]), verbose=0), target)
                # If out of bounds, it needs to punish every action taken that made it further out of bounds
                target_q_values = model.predict(np.array([state]), verbose=0) # What does this represent?
                # print(target_q_values)
                target_q_values[0][action] = target # Store as needed
                log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
                model.fit(np.array([state]), target_q_values, epochs=10, verbose=2) # Prints loss
                # Why do I need a loss function? Does it measure how well I did after even though this is RL?
                # print(" State =", next_state)
        
        # 🔹 STEP 3.4: Reduce Randomness Over Time
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        print(f"Episode {episode+1}: Total Reward: {total_reward}")
    
    env.close()
    return model
'''

# STEP 4: Train the AI
trained_model = train_ai(episodes=00)
trained_model.save(checkpoint_path)

# STEP 5: Watch AI Play the Game!
def play_game(model, env):
    state = env.reset()
    done = False
    
    while not done:
        env.render()
        action = np.argmax(model.predict(np.array([state]), verbose=0))  # AI chooses best move
        state, _, done, _ = env.step(action)

env = NumberGame()
for i in range(5):
    play_game(trained_model, env)
env.close()


'''
# Train the model with the new callback
model.fit(train_images, 
            train_labels,  
            epochs=10,
            validation_data=(test_images, test_labels),
            callbacks=[cp_callback])  # Pass callback to training

model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),  # Hidden layer with 16 neurons
    Dense(16, activation='relu'),  # Another hidden layer
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])
# Input → [16 Neurons] → [16 Neurons] → Output
model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
'''

'''
# Create a callback that saves the model's weights
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)



checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)

# Train the model with the new callback
model.fit(train_images, 
            train_labels,  
            epochs=10,
            validation_data=(test_images, test_labels),
            callbacks=[cp_callback])  # Pass callback to training

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.
'''