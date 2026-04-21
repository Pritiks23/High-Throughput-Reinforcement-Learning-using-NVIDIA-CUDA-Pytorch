# FP8-Style RL Training Demo (Actor-Critic + Custom Environment)

---

## 1. Custom Environment Design

Our custom environment will feature:

- **State:** A small, continuous state space (e.g., position, velocity).
- **Actions:** Discrete actions (e.g., move left, move right).
- **Rewards:** A simple reward structure (e.g., +1 for staying upright).
- **Termination:** Episodes end after a fixed number of steps or if a failure condition is met.

The **CustomEnv** environment, which our agent interacts with, is a very simplified world with these main characteristics:

---

### What the Agent Sees (State)
The agent is given two pieces of information at all times: its current position and its current velocity (how fast and in what direction it's moving).

### What the Agent Can Do (Actions)
The agent has only two choices: it can apply a small push to the left or a small push to the right.

### The Agent's Goal (Rewards)
The environment rewards the agent for staying as close as possible to a central, target position (which is 0.0). The further away it drifts from this center, the less reward it gets. It also gets a small survival reward for each step it remains active.

---

## 2. Actor-Critic Architecture

We'll implement a simple Actor-Critic model using PyTorch. The Actor will output a probability distribution over actions, and the Critic will estimate the value function (expected return) of a given state. Both will share a common feature extraction network, which is a common practice to encourage stable learning.

Our architecture will consist of:

- **Shared Encoder:** A few fully connected layers to process the state input.
- **Actor Head:** Outputs logits for a categorical distribution over actions.
- **Critic Head:** Outputs a single scalar value representing the state-value.

---

### The Actor
This part of the model predicts the best action to take given the current state of the environment. More precisely, it outputs **action_logits** which represent the relative probabilities of taking each possible action. The agent then samples an action from this distribution to interact with the environment.

### The Critic
This part of the model predicts the value of being in a particular state. The **state_value** output estimates the expected cumulative future reward that the agent will receive starting from that state. This value is crucial for helping the Actor understand how good its chosen actions are and for stabilizing the training process.

In essence, the model is learning a policy (how to behave) and a value function (how good situations are) to maximize the total reward in the CustomEnv.

---

## 3. FP8-style Low-Precision Simulation & Basic Training Loop Components

To simulate FP8-style low-precision optimization, we'll leverage PyTorch's Automatic Mixed Precision (AMP) functionality (`torch.cuda.amp`). While AMP primarily uses float16 or bfloat16, it conceptually demonstrates the benefits of using lower precision data types for computation, including reduced memory usage and faster training, which are key advantages of true FP8.

---

In this section, we'll:

- Define core hyperparameters for our RL training.
- Initialize the CustomEnv and the ActorCritic model.
- Set up the optimizer and `torch.cuda.amp.GradScaler` for mixed-precision training.
- Outline the structure for our training loop, which will involve collecting experiences, computing advantages, and updating the network.

---

## Hyperparameters Explanation

### state_dim (State Dimension) - Set to 2
This just means how many numbers we use to describe the environment's current situation. For our simple CustomEnv, the agent's situation is fully described by its position and its velocity, so we need two numbers.

### action_dim (Action Dimension) - Set to 2
This tells us how many different actions the agent can choose from. In our environment, the agent can either push left (action 0) or push right (action 1), so there are two possible actions.

### hidden_size (Hidden Layer Size) - Set to 64
This refers to the 'thinking capacity' of the neural network inside our agent. A value of 64 means there are 64 processing units in the intermediate layers of the network. This number is a balance: too small, and the network can't learn complex patterns; too large, and it might overcomplicate things or take too long to train. 64 is a common, reasonable starting point for simple tasks.

### learning_rate - Set to 0.001
Think of this as how big a 'step' the agent takes when it learns from its mistakes. If the learning rate is too high, the agent might jump around wildly and never settle on a good strategy. If it's too low, it will learn very slowly. 0.001 is a commonly used small step size that allows for stable learning.

### gamma (Discount Factor) - Set to 0.99
This value determines how much the agent cares about rewards it gets in the immediate future versus rewards it might get much later. A gamma of 0.99 means the agent values future rewards almost as much as immediate ones, encouraging it to plan ahead.

### num_episodes (Number of Episodes) - Set to 2000
An 'episode' is one complete run of the agent in the environment, from start to finish (or until it fails). 2000 episodes means the agent gets 2000 chances to try, fail, and learn before we consider its training complete.

### max_steps_per_episode - Set to 200
This is a safety limit for how long a single episode can last. After 200 steps, we cut it short and start a new one.

### collect_steps (Collection Steps) - Set to 128
This is how many interactions the agent has with the environment before it pauses to 'think' and update its neural network.

### buffer_capacity (Replay Buffer Capacity) - Set to 10000
The 'replay buffer' is like the agent's memory bank. It can store up to 10,000 past experiences.

### batch_size - Set to 64
When the agent updates its network, it randomly picks 64 experiences from memory to learn from.

---

## 4. Training Loop and Experience Collection

With our environment and model defined, we can now implement the training loop. This loop will involve:

- **Experience Collection:** Interacting with the CustomEnv to gather state, action, reward, and next state transitions.
- **Replay Buffer:** Storing these experiences to break correlations and allow for efficient sampling during training.
- **Policy Update:** Using the collected experiences to update the Actor and Critic networks.
- **Mixed Precision:** Leveraging `torch.amp.autocast` within the training steps.

First, let's define a simple ReplayBuffer class.

---

## 5. Training Loop Implementation

Now, we'll put all the pieces together to create the main training loop. This loop will execute for a specified number of episodes.

### Key aspects:

- **Environment Interaction:** The agent interacts with the CustomEnv.
- **Experience Storage:** (state, action, reward, next_state, done) stored in ReplayBuffer.
- **Policy Update:** After collecting collect_steps experiences, a batch is sampled.
- **Mixed Precision:** `torch.amp.autocast` is used during forward pass.
- **Metrics:** Episode rewards are tracked.

---

### Training Loop Intuition

Imagine this loop as the agent repeatedly living through 'days' (episodes) and making 'choices' (steps) within those days.

---

### Preparing for Learning

- **Tracking Rewards:** We keep a list (`episode_rewards`) of recent rewards.
- **Total Interactions:** Counter (`total_steps`) tracks all actions.

---

### Main Training Cycle

The agent goes through **2000 episodes**:

#### Starting Fresh
- Environment reset
- State initialized
- Reward counter reset

---

### Taking Action (Each Episode)

The agent takes up to **200 steps**:

- Observes state
- Actor selects action
- Executes action in environment
- Receives next_state, reward, done
- Stores experience in replay buffer
- Updates counters

---

### End of Episode

Episode ends when:
- Done condition is met OR
- Step limit reached

---

### Learning Phase (Every collect_steps = 128)

- Sample batch from replay buffer
- Compute losses (Actor + Critic)
- Update model using GradScaler
- Improve policy and value estimates

---

### Progress Tracking

Every 100 episodes:
- Print average reward

---

## After Training

### Final Evaluation
Run test episodes without updates to evaluate learned policy.

---

## Key Learnings from FP8-Style RL Optimization Demo

### 1. Simplified Environment for Rapid Prototyping
A minimal environment enables fast RL experimentation.

### 2. Actor-Critic Architecture
- Actor = policy learning
- Critic = value estimation

---

### 3. FP8-style Low Precision Simulation

We used `torch.amp` to simulate FP8 benefits:

- Lower memory usage
- Faster computation
- Higher throughput

---

### 4. GradScaler Stability

Prevents gradient underflow in mixed precision training.

---

### 5. A2C-like Learning Loop

- Collect experience
- Compute advantage
- Update policy + value function

---

## Final Summary

This demo shows how:

- RL agents learn via interaction
- Actor-Critic stabilizes training
- Mixed precision improves efficiency
- Replay buffers improve sample efficiency

It forms a foundational pipeline for scalable RL systems inspired by high-throughput GPU training architectures.
