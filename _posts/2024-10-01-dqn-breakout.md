---
layout: post
title: How to Build a DQN to Play Atari Breakout (and Actually Get Good Results!)
date: 2024-10-01 15:00:00
description: An easy-to-follow guide on implementing a Deep Q-Network (DQN) for Atari Breakout.
tags: DQN, Atari, Reinforcement Learning, Deep Learning
categories: tutorials
pseudocode: true
---

If you’ve ever wondered how an AI can learn to play games like Atari Breakout, you’re in the right place! In this post, we’ll break down the steps I took to build a Deep Q-Network (DQN) that can play Breakout and smash those blocks with skill.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0 d-flex align-items-center justify-content-center" style="height: 100%;">
        {% include video.liquid loading="eager" path="assets/video/ATARI_Breakout_Eval_model_21700_reward_357-speed.mp4" class="img-fluid rounded z-depth-1" width="200" controls=true autoplay=true%}
    </div>
</div>

For a deep dive into the code, feel free to check out my [GitHub repository](https://github.com/francesco-fortunato/DQN-breakout). Let's get started!

---

## The Environment: Atari Breakout

Atari Breakout is a timeless classic where you control a paddle to bounce a ball and break bricks. The objective is straightforward: break all the bricks without letting the ball slip past you. To make our lives easier, we’ll leverage OpenAI Gym, which offers a variety of built-in environments for simulating games. For our project, we’ll be using the BreakoutNoFrameskip-v4 environment.

## The DQN Model: Teaching the Agent to Play

Our DQN model is essentially a neural network that learns to predict the best action to take based on the current game state. In simpler terms, it tries to figure out whether it should move the paddle left, right, or just stay put.

### CNN Architecture: Why Convolutions Are Perfect for Games

For our agent to “see” the game properly, we use a Convolutional Neural Network (CNN). CNNs are amazing for image-based tasks because they can automatically pick out important patterns, like where the ball and bricks are.

Here’s the architecture we used:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Network.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    CNN Architecture
</div>

That translates in the following code:

```python
def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(84, 84, 4,))

    # Define the first convolutional layer
    # - 32 filters, each 8x8 in size
    # - Stride of 4, meaning the filter moves 4 pixels at a time
    # - ReLU activation function is applied to the output
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)

    # Define the second convolutional layer
    # - 64 filters, each 4x4 in size
    # - Stride of 2
    # - ReLU activation function
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)

    # Define the third convolutional layer
    # - 64 filters, each 3x3 in size
    # - Stride of 1
    # - ReLU activation function
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    # Flatten the output from the convolutional layers
    layer4 = layers.Flatten()(layer3)

    # Define a fully connected layer with 512 neurons
    # - ReLU activation function
    layer5 = layers.Dense(512, activation="relu")(layer4)

    # Output layer with num_actions neurons (4 in this case for the Breakout game)
    # - Linear activation function
    action = layers.Dense(num_actions, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)
```

Now, you may think: if I have just three moves inside Atari (left, right, still), why are there 4 `num_actions`? Well, the fourth action is needed for starting the game. Once the game has started, we don't want to use that button anymore. For this reason, we use **Atari wrappers** to handle such nuances in the environment.

---

## Atari Breakout Environment Wrappers

To help the agent interact with the environment effectively, we use several **Atari wrappers**. These wrappers preprocess the frames, manage game resets, and handle special cases like losing a life or needing to press "Fire" to start the game. Some of the most important wrappers we use to ensure the game environment includes EpisodicLifeEnv to treat life loss as episode end (we have three lifes), MaxAndSkipEnv to skip unnecessary frames, and FrameStack to maintain temporal continuity by stacking multiple frames.

```python
# Use the Baseline Atari environment because of Deepmind helper functions
env = make_atari_breakout("BreakoutNoFrameskip-v4")
# Warp the frames, grey scale, stake four frame and scale to smaller ratio
env = wrap(env, frame_stack=True, scale=True)
env.seed(seed)
```

---

### Pre-processing of the Images

To ensure our neural network can learn effectively from the game frames, we need to preprocess the images observed in the environment before they are fed into the network.

- **Frame Stacking**: The wrapper stacks multiple frames together (in this case, four frames) to provide the neural network with a sense of motion and temporal continuity. This helps the agent make decisions based on the recent history of observations.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0 d-flex align-items-center justify-content-center" style="height: 100%;">
        {% include figure.liquid loading="eager" path="assets/img/frame_00_delay-0.02s.jpg" class="img-fluid rounded z-depth-1" width="200" %}
    </div>
</div>
<div class="caption">
    With just one frame, it's impossible to tell where the ball is going.
</div>

Below, you can see how stacking four frames together provides a clearer understanding of the ball's movement:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <div style="display: flex; justify-content: space-around;">
            {% include figure.liquid loading="eager" path="assets/img/frame_00_delay-0.02s.jpg" class="img-fluid rounded z-depth-1" %}
            {% include figure.liquid loading="eager" path="assets/img/frame_01_delay-0.02s.jpg" class="img-fluid rounded z-depth-1" %}
            {% include figure.liquid loading="eager" path="assets/img/frame_02_delay-0.02s.jpg" class="img-fluid rounded z-depth-1" %}
            {% include figure.liquid loading="eager" path="assets/img/frame_03_delay-0.02s.jpg" class="img-fluid rounded z-depth-1" %}
        </div>
    </div>
</div>

<div class="caption">
    With a stack of four frames, we can easily understand where the ball is going.
</div>

- **Grayscale Conversion**: Grayscale conversion helps reduce the computational complexity of the neural network, making it faster to train.

- **Scaling to a Smaller Ratio**: It crops the images to remove unnecessary parts, like the score and other additional information that are not useful for the computation.

Below is the pre-processing flow showcasing the steps:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <div style="display: flex; justify-content: space-around; align-items: center; height: 100%;">
            <div style="width: 20%; text-align: center;">
                {% include figure.liquid loading="eager" path="assets/img/frame_00_delay-0.02s.jpg" class="img-fluid rounded z-depth-1" %}
                <div class="caption">Original Frame</div>
            </div>
            <div style="width: 20%; text-align: center;">
                {% include figure.liquid loading="eager" path="assets/img/cropped.png" class="img-fluid rounded z-depth-1" %}
                <div class="caption">Cropped Frame</div>
            </div>
            <div style="width: 20%; text-align: center;">
                {% include figure.liquid loading="eager" path="assets/img/grey.png" class="img-fluid rounded z-depth-1" %}
                <div class="caption">Grayscale Frame</div>
            </div>
        </div>
    </div>
</div>

<div class="caption">
    Pre-processing flow: from the original frame to cropped and grayscale versions.
</div>

---

## Training Process

### Exploration-Exploitation Trade-off

In reinforcement learning, agents face the dilemma of whether to explore new actions or exploit known ones to maximize rewards. This **exploration-exploitation trade-off** is crucial for balancing the discovery of potentially better actions and the exploitation of known optimal actions.

We define **$$\epsilon$$** as a function of the number of frames the agent has seen. For the first 50,000 frames, the agent only explores, setting **$$\epsilon = 1$$**. Over the next 1 million frames, **$$\epsilon$$** is linearly decreased to **0.1**, meaning the agent gradually starts to exploit its knowledge more. While DeepMind maintains **$$\epsilon = 0.1$$**, I chose to reduce it to **$$\epsilon = 0.01$$** over the remaining 24 million frames, as suggested by the [OpenAI Baselines for DQN](https://openai.com/research/openai-baselines-dqn) (note that in the accompanying plot, the maximum number of frames is shown as 2 million for demonstration purposes).

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/epsilon.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

---

### Configuration Parameters

To kick things off, let’s outline the key configuration parameters that will set the foundation for our training process. These parameters are essential for controlling various aspects of the learning algorithm:

```python
# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_final = 0.01  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
epsilon_interval_2 = (
    epsilon_min - epsilon_final
)  # Rate at which to reduce chance of random action being taken after 1 million frames

# Number of frames to take random action and observe output
epsilon_random_frames = 50000.0   # Number of frames with epsilon set to 1.0
# Number of frames for exploration
epsilon_greedy_frames = 1000000.0 # Number of frames to linearly decay epsilon from 1 to 0.1
epsilon_final_frames = 24000000.0 # Number of frames to linearly decay epsilon from 0.1 to 0.01
```

### Pseudocode

Now, let’s dive into the pseudocode for our Deep Q-learning algorithm with experience replay. This gives us a clear roadmap of the algorithm's structure and flow:

```pseudocode
\begin{algorithm}
\caption{Deep Q-learning with Experience Replay}
\begin{algorithmic}
\STATE Initialize replay memory $$D$$ to capacity $$N$$
\STATE Initialize action-value function $$Q$$ with random weights $$θ$$
\STATE Initialize target action-value function $$Q̂$$ with weights $$θ⁻ = θ$$
\FOR{episode $$= 1$$ \TO $$M$$}
    \STATE Initialize sequence $$s₁ = \{x₁\}$$ and preprocessed sequence $$ϕ₁ = ϕ(s₁)$$
    \FOR{$$t = 1$$ \TO $$T$$}
        \STATE With probability $$ε$$ select a random action $$aₜ$$
        \STATE otherwise select $$aₜ = $$ argmaxₐ$$ Q(ϕ(sₜ), a; θ)$$
        \STATE Execute action $$aₜ$$ in emulator and observe reward $$rₜ$$ and image $$xₜ₊₁$$
        \STATE Set $$sₜ₊₁ = sₜ, aₜ, xₜ₊₁$$ and preprocess $$ϕₜ₊₁ = ϕ(sₜ₊₁)$$
        \STATE Store transition $$(ϕₜ, aₜ, rₜ, ϕₜ₊₁)$$ in $$D$$
        \STATE Sample random minibatch of transitions $$(ϕₕ, aₕ, rₕ, ϕₕ₊₁)$$ from $$D$$
        \STATE Set $$yₕ = rₕ if episode terminates at step h + 1; otherwise rₕ + γ maxₐ Q̂(ϕₕ₊₁, a'; θ⁻)$$
        \STATE Perform a gradient descent step on $$(yₕ - Q(ϕₕ, aₕ; θ))²$$ with respect to the network parameters $$θ$$
        \STATE Every $$C$$ steps, set $$Q̂ = Q$$
    \ENDFOR
\ENDFOR
\end{algorithmic}
\end{algorithm}
```

This pseudocode outlines how we initialize our environment, select actions based on the epsilon-greedy strategy, and manage experiences in the replay buffer. Let's explain how does it works and which are the components we need.

---

### Replay Memory and Other Parameters

One crucial aspect of our training setup is the replay memory. Experiences are stored in a replay buffer, allowing the model to be trained periodically using sampled batches from this buffer. This approach not only helps in maintaining a diverse set of experiences but also stabilizes the learning process.

In the spirit of the DeepMind paper, I've defined a maximum memory size for the buffer, albeit a smaller one due to computational constraints. Using a replay buffer is beneficial because experiences are often highly correlated temporally. Training directly on consecutive experiences can lead to instability and slow convergence. Without this buffer, an agent risks overfitting to recent experiences, which hampers its ability to generalize effectively.

By implementing these configurations and leveraging replay memory, we set our agent up for success in mastering the task at hand.

#### Experience Replay Buffers

One of the foundational techniques in our reinforcement learning setup for implementing the replay memory is the use of experience replay buffers. These buffers store the agent’s experiences, allowing for more effective training by breaking the correlation between consecutive experiences. Below are the key components of our experience replay setup:

```python
# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000

# Maximum replay length
# Note: The Deepmind paper suggests 1,000,000; however, this causes memory issues
max_memory_length = 100000

# Train the model after 4 actions
update_after_actions = 4

# How often to update the target network
update_target_network = 10000

running_reward = 0
episode_count = 0
frame_count = 0
terminal_life_lost = False
```

---

### Huber Loss

One other interesting thing to notice: DeepMind uses the quadratic cost function with error clipping (see page 7 of [Mnih et al. 2015](https://www.nature.com/articles/nature14236/)).

> We also found it helpful to clip the error term from the update [...] to be between -1 and 1. Because the absolute value loss function `$$|x|$$` has a derivative of -1 for all negative values of `$$x$$` and a derivative of 1 for all positive values of `$$x$$`, clipping the squared error to be between -1 and 1 corresponds to using an absolute value loss function for errors outside of the (-1,1) interval. This form of error clipping further improved the stability of the algorithm.

Why does this improve the stability of the algorithm?

> In deep networks or recurrent neural networks, error gradients can accumulate during an update and result in very large gradients. These can lead to large updates to the network weights, resulting in an unstable network. In extreme cases, the values of weights can become so large as to overflow and result in NaN values. [Source](https://machinelearningmastery.com/exploding-gradients-in-neural-networks/)

This so-called **exploding gradient problem** can be avoided to some extent by clipping the gradients to a certain threshold value if they exceed it: _If the true gradient is larger than a critical value \(x\), just assume it is \(x\)._ The derivative of the green curve does not increase (or decrease) for \(x > 1\) (or \(x < -1\)).

Error clipping can be easily implemented in TensorFlow by using the Huber loss function `tf.losses.huber_loss`.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/huber.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

#### Implementing Huber Loss

```python
# Using huber loss for stability
loss_function = keras.losses.Huber()

# In the DeepMind paper, they use RMSProp; however, the Adam optimizer improves training time
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
```

---

### Q-Values and the Bellman Equation

One of the foundational concepts in reinforcement learning field that shapes how intelligent agents make decisions is the **Bellman equation**. This powerful equation allows agents to evaluate the quality of actions they can take in various states, essentially providing a roadmap for optimal decision-making.

The Bellman equation is expressed as follows:

$$
Q(s, a) = r + \gamma \cdot \max_{a'} Q(s', a')
$$

Let’s break down its components to understand what each term represents:

- **\(Q(s, a)\)**: This is the Q-value for a specific action \(a\) in a given state \(s\). It serves as a crucial indicator of how good an action is in a particular situation, guiding the agent toward better choices.
- **\(r\)**: This term signifies the immediate reward the agent receives after taking action \(a\) in state \(s\). It provides instant feedback from the environment, telling the agent how well it performed.

- **\(\gamma\)**: Known as the discount factor, this parameter ranges between 0 and 1. It helps determine how much importance the agent should give to future rewards compared to immediate ones, fostering a balance between short-term gains and long-term strategies.

- **\(\max\_{a'} Q(s', a')\)**: This part captures the maximum Q-value achievable in the next state \(s'\). It reflects the agent's estimate of the best possible future reward by considering all available actions.

The Bellman equation is a guiding principle for reinforcement learning algorithms, blending theoretical concepts with practical implementations.

#### Additional Consideration in Implementation

In our implementation, we make a slight modification to the Bellman equation to better accommodate the dynamics of the environment:

$$
Q(s, a) = r + \gamma \cdot \max_{a'} Q(s', a') \cdot (1 - \text{done})
$$

Here, the addition of the binary flag \(\text{done}\) indicates whether the episode has terminated after taking the current action. The term \((1 - \text{done})\) acts as a filter, allowing the update of Q-values only when the episode is ongoing. If the episode has ended (\(\text{done} = 1\)), future expected rewards are excluded from the update, as there’s no subsequent state (\(s'\)) to consider.

This adjustment is particularly important in scenarios where the termination of an episode doesn’t signify the end of the learning process. In games like Atari Breakout, losing a life does not mean the game is over; the agent continues to play. The binary flag \(\text{done}\) ensures that Q-values are updated appropriately based on the episodic dynamics, allowing the agent to learn from its experiences even after setbacks.

Let’s illustrate this in simpler terms:

$$
\text{Set } y_j =
\begin{cases}
    r_j & \text{if episode terminates at step } j+1 \\
    r_j + \gamma \max_{a'} Q(s_{j+1}, a'; \theta^-) & \text{otherwise}
\end{cases}
$$

#### Code Representation

In the Python implementation, this concept is reflected in the following line of code:

```python
updated_q_values = rewards_sample + (1 - done_sample) * gamma * tf.reduce_max(future_rewards, axis=1)
```

This line corresponds to our modified Bellman equation, where we effectively update the Q-values during the agent's learning process. By incorporating the discount factor and the done flag, we ensure that our Q-learning approach is both effective and adaptable to the nuances of the environment.

### Training Loop

In the heart of our Deep Q-Learning algorithm lies the training loop, where the agent learns from its interactions with the environment. To set this up effectively, we need to create two models: the **primary model** that will be trained and a **target model** for predicting future rewards. Here’s how we set them up:

```python
# The first model makes the predictions for Q-values which are used to
# make a action.
model = create_q_model()
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = create_q_model()
```

The target model plays a critical role in mitigating the oscillations and divergence that can occur during training. Instead of using the current model to predict future Q-values, we update the target model’s weights to match those of the training model every 10,000 time steps. This stability is essential for effective learning.

To track the training process and check the results, I decided to implement a script that creates or updates a CSV file containing essential statistics:

```python
csv_filename = "training_stats.csv"
# Check if the CSV file exists
if os.path.exists(csv_filename):
    with open(csv_filename, mode='r') as file:
        # CSV file already exists, read the header
        reader = csv.reader(file)
        header = next(reader)
else:
    # CSV file does not exist, create and write the header
    header = ["Episode", "Total Reward", "Epsilon", "Avg Reward (Last 100)", "Total Frames",
              "Frame Rate", "Model Updates", "Running Reward", "Training Time"]
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
```

### How the Training Loop Works

Now, let’s break down how the training loop operates. At each timestep, the agent selects an action based on the epsilon-greedy strategy, takes a step in the environment, and stores this transition in memory. A random batch of 32 transitions is then sampled from this replay buffer to train the neural network.

For each training sample \((s, a, r, s')\) in the mini-batch, the model is provided with a state (a stack of four frames). Using the next state and the Bellman equation, we derive the targets for our neural network. Specifically, if the next state is a terminal state (meaning the episode has ended), the target is simply the immediate reward \(r\). Otherwise, the Q-value should reflect both the immediate reward and the discounted value of the highest future Q-value.

This approach ensures that when an episode concludes, the next state is not considered in the update, which is facilitated by the `done_sample` array. By using the target network, we stabilize the learning process and avoid oscillations that could lead to divergence.

Here’s how this training loop is implemented in Python:

```python
starting = datetime.datetime.now()
while True:  # Run until solved
    start_time = time.time()
    state = np.array(env.reset())

    current_lives = 5

    episode_reward = 0

    for timestep in range(1, max_steps_per_episode):
        # env.render(); Adding this line would show the attempts
        # of the agent in a pop up window.
        frame_count += 1

        # Use epsilon-greedy for exploration
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(num_actions)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take the best action
            action = tf.argmax(action_probs[0]).numpy()

        if frame_count > epsilon_random_frames: # Decay epsilon only after exploring for first 50k frames
            if epsilon > epsilon_min:
                # Decay probability of taking random action
                epsilon -= epsilon_interval / epsilon_greedy_frames
                epsilon = max(epsilon, epsilon_min)
            else:
                # Continue decaying epsilon linearly over the remaining frames
                epsilon -= epsilon_interval_2 / (epsilon_final_frames)
                epsilon = max(epsilon, epsilon_final)


        # Apply the sampled action in our environment
        state_next, reward, done, info = env.step(action)
        state_next = np.array(state_next)

        episode_reward += reward

        # When a life is lost, we save terminal_life_lost = True in the replay memory
        # N.B. We don't modify directly done, since done is already used to break the loop
        num_lives = info['lives']

        if (num_lives < current_lives):
            terminal_life_lost = True
            current_lives = num_lives
        else:
            terminal_life_lost = False

        # Save actions and states in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(terminal_life_lost if not done else done) # If the game is not terminated, if life lost add true, else add done (False or true)
        rewards_history.append(reward)
        state = state_next

        # Update every fourth frame and once batch size is over 32
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:

            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            ) # turns True into 1.0 and False into 0.0.

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict(state_next_sample)
            # Q value = reward + discount factor * expected future reward
            # updated_q_values = rewards_sample + gamma * tf.reduce_max(
            #    future_rewards, axis=1
            # )

            # Our Implementation
            # If the game is over because the agent lost or won, there is no next state and the value is simply the reward

            updated_q_values = rewards_sample + (1 - done_sample) * gamma * tf.reduce_max(future_rewards, axis=1)

            # Create a mask so we only calculate loss on the updated Q-values (If action taken was 1, it create [0,1,0,0])
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                #  to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
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
            # print(info)
            break

    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    # Calculate additional statistics
    avg_reward_last_100 = np.mean(episode_reward_history[-100:])
    frame_rate = frame_count / (time.time() - start_time)
    training_time = time.time() - start_time

    # Append the episode statistics to the CSV file
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([episode_count, episode_reward, epsilon, avg_reward_last_100,
                            frame_count, frame_rate, len(done_history),
                            running_reward, training_time])

    if (episode_count%100 == 0):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"{current_time} - Episode {episode_count} reached. Saving model in saved_models/model_episode_{episode_count}. . .")
        model.save("saved_models/model_episode_{}".format(episode_count))
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{current_time} - Model saved.")
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{current_time} - Saving target model. . .")
        # Save the target model
        model_target.save("saved_models/target_model_episode_{}".format(episode_count))
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{current_time} - Target model saved in saved_models/target_model_episode_{episode_count}.")

    episode_count += 1
    if (num_lives==0):
        template = "running reward: {:.2f} at episode {}, frame count {}"
        print(template.format(running_reward, episode_count, frame_count))

    if running_reward > 40:  # 40 is the avg score of human beings
        print("Solved at episode {}!".format(episode_count))
        episode_count -= 1
        break
```

## Conclusion

Getting Deep Q-Learning to work can be a bit of a pain, right? There are so many little details to tweak, and if you miss one, things just don’t click. Plus, if you’re running experiments on a single GPU (or even without), get ready for those overnight waits! Debugging becomes a slow grind, and it can feel like you’re just spinning your wheels sometimes.

But honestly, it’s been totally worth it. I learned a ton about neural networks and reinforcement learning through all the debugging and fine-tuning. It’s amazing to see the progress your agent makes!

So, give it a shot! Implement your own DQN, tweak it, and watch your AI tackle Atari Breakout. It's a rewarding experience, and who knows? You might just create something impressive.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0 d-flex align-items-center justify-content-center" style="height: 100%;">
        {% include video.liquid loading="eager" path="assets/video/ATARI_Breakout_Eval_model_21700_reward_357-speed.mp4" class="img-fluid rounded z-depth-1" width="200" controls=true autoplay=true%}
    </div>
</div>

If you have any questions or want to share your journey, feel free to contact me. Happy coding, and enjoy the ride!
