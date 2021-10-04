# Collaboration and Competetion Project Report

## 1. Introduction
To solve this environment, we used Reinforcement Learning algorithms to train an agent to move to a location where it hits the ball. In particular, we used the DDPG (Deep Deterministic Gradient Policy) algorithm. 

DDPG uses 2 neural networks that represet an actor and a critic. The actor is a policy-based method and the critic is a value-based method. Actor-critic methods are at the intersection of value-based methods such as DQN and policy-based methods such as reinforce. 

If a deep reinforcement learning agent uses a deep neural network to approximate a value function, the agent is said to be value-based. If an agent uses a deep neural network to approximate a policy, the agent is said to be policy-based. 

### Value-Based Methods
Deep Q-Network (DQN) is a value-based method that is used to represent the Q-table using a function approximator (neural network). It calculates the action-value using a state-action pair. The goal of value-based methods is to determine the optimal Q-table (or in this case Q-function) and use it to define the optimal policy.

DQN uses 2 identical neural networks: one called the local Q-Network with parameters _W_ and another called the target Q-Network with parameters _W<sup>-</sup>_.

In DQN, we needed the tuple _(S<sub>t</sub>,A<sub>t</sub>,R<sub>t</sub>,S<sub>t+1</sub>)_ at a given timestep _t_ to compute the gradients of the parameters of the neural network following the temporal difference (TD) algorithm. The gradient update rule was as follows:

<p align="center">
  <img src="https://user-images.githubusercontent.com/47497135/135057127-fe8da832-a2d3-487a-8df9-eba626e87073.png" alt="drawing" width="500"/>
</p>

Where Y<sub>t</sub><sup>DQN</sup> is the expected action-value and _Q(S<sub>t</sub>,A<sub>t</sub>;W<sub>t</sub>)_ is the predicted action-value.

The main disadvantage of value-based methods is the inability to handle continuous action spaces. It needs exclusively a discrete action space. Also, as a result of using temporal difference (TD) in learning, the agent estimates the action-value using an estimate. In other words, we guess the value using another guess. This produces bias in the agent and could lead to inaccuracy in the optimal policy.

### Policy-Based Methods
Policy-based methods don’t need an action-value. They, instead, tweak the policy continuously until the policy maximizes the total reward thus achieving the optimal policy. They have to use Monte-Carlo approach, meaning that they need an entire episode to finish to update the parameters using the rewards from those episodes. They can handle continuous action spaces very well.

The main disadvantage is of policy-based methods is the large variance due to using Monte Carlo approach and this leads to slow learning.

### Actor-Critic Methods
Actor-critic methods use 2 neural networks: one representing the actor and another representing the critic. The actor takes an action based on the state (this action can be continuous because it is policy-based) and the critic evaluates this action (it doesn’t need an entire episode because it is value-based). Thus, combining the advantages of both value-based and policy-based methods.

## 2. Learning Algorithm
### Neural Network Architecture
As the DQN algorithm has, the critic will be a set of 2 identical neural networks: one representing the target and another representing the local. The actor will also be a set of 2 identical neural networks: one representing the target and another representing the local. With a total of 4 neural networks, the update rule of the target networks will be soft-update with a parameter τ.

The hyperparameters are:
* Activation function: `relu`.
* Batch size: `128`.
* Learning rate of critic: `0.0001`.
* Learning rate of actor: `0.0001`.

#### Critic
The critic is a value-based neural network. It calculates the action-value using a state-action pair. 

The input layer of the network receives the state _S<sub>t</sub>_. The first hidden layer receives the output of the first layer and the action _A<sub>t</sub>_. You can add other hidden layers but the first 2 have to be like this. The output of the network is a single neuron representing the action-value _Q<sub>t</sub>(S<sub>t</sub>,A<sub>t</sub>;θ<sup>Q</sup>)_ where _θ<sup>Q</sup>_ are the parameters of the critic neural network.

The neural network architecture of the critic is as follows (shown in Fig.1):
- The input layer has the same number of states of the enivronment: `24`
- The first hidden layer has `200` neurons. The output of this layer is concatenated with the action input of `2`.
- The second hidden layer has `100` neurons.
- The output layer has `1` neuron.

<p align="center">
  <img src="https://user-images.githubusercontent.com/47497135/135766099-760e0d08-820e-4b3d-9121-ed2f3eea1f8c.png" alt="drawing" width="400"/>
</p>
<p align="center">
  <em>Fig.1: Critic Neural Network Architecture</em>
</p>

#### Actor
The actor is a policy-based neural network. It simply maps the state into action.

The input layer receives the state _S<sub>t</sub>_ and the output layer produces the probability distribution of each action _a∈A_ where _A_ is the action space.

The neural network architecture of the critic is as follows (shown in Fig.2):
- The input layer has the same number of states of the enivronment: `24` neurons.
- The first hidden layer has `200` neurons.
- The second hidden layer has `100` neurons.
- The output layer has the same number of actions of the enivronment: `2` neurons.

<p align="center">
  <img src="https://user-images.githubusercontent.com/47497135/135766160-359deded-5ada-4b63-bc33-4fd35d362d7a.png" alt="drawing" width="300"/>
</p>
<p align="center">
  <em>Fig.2: Actor Neural Network Architecture</em>
</p>

### Experience Replay
When the agent interacts with the environment, the sequence of experience tuples can be highly correlated. The naive Q-learning algorithm that learns from each of these experience tuples in sequential order runs the risk of getting swayed by the effects of this correlation. By instead keeping track of a __replay buffer__ and using __experience replay__ to sample from the buffer at random, we can prevent action values from oscillating or diverging catastrophically.

The __replay buffer__ contains a collection of experience tuples __*(S, A, R, S')*__. The tuples are gradually added to the buffer as we are interacting with the environment.

The act of sampling a small batch of tuples from the replay buffer in order to learn is known as __experience replay__. In addition to breaking harmful correlations, experience replay allows us to learn more from individual tuples multiple times, recall rare occurrences, and in general make better use of our experience.

The hyperparameters are:
* Buffer size: `1,000,000`.

### Fixed Q-Targets
In Q-Learning, we update a guess with a guess, and this can potentially lead to harmful correlations. To avoid this, we can update the parameters __*w*__ in the network $Q_hat$ to better approximate the action value corresponding to state __*S*__ and action __*A*__ with the following update rule:

<p align="center">
  <img src="https://user-images.githubusercontent.com/47497135/134077362-35bad85b-cb3e-41fc-9d35-35f2a66201ca.png" alt="drawing" width="600"/>
</p>

where __*w<sup>-</sup>*__ are the weights of a separate target network that are not changed during the learning step, and __*(S, A, R, S')*__ is an experience tuple.

The hyperparameters are:
* Discount factor &gamma;: `0.99`.

### Ornstein-Uhlenbeck Noise Process 
A major challenge of learning in continuous action spaces is exploration. An advantage of offpolicies algorithms such as DDPG is that we can treat the problem of exploration independently from the learning algorithm. We constructed an exploration policy _μ'_ by adding noise sampled from a noise process _N_ to our actor policy:

<p align="center">
  <img src="https://user-images.githubusercontent.com/47497135/135072151-3a25a5d1-599d-49a9-b129-1d92d63bdbf2.png" alt="drawing" width="220"/>
</p>

_N_ can be chosen to suit the environment. As detailed in the supplementary materials we used an Ornstein-Uhlenbeck process to generate temporally correlated exploration for exploration efficiency in physical control problems with inertia.

The hyperparameters are:
* &sigma;: `0.20`.
* &theta;: `0.15`.

### Update Rules
The update rule for the critic parameters will be minimizing the loss function:

<p align="center">
  <img src="https://user-images.githubusercontent.com/47497135/135066407-e9696d85-3d5c-4f10-a282-8ba022f7a90f.png" alt="drawing" width="250"/>
</p>

Where:
- _N_ is the number of minibatches.
- _i_ is a random minibatch number.
- _Q(S<sub>i</sub>,A<sub>i</sub>;θ<sup>Q</sup>)_ is the action-value produced by the local critic network when given the current state _S<sub>i</sub>_ and the current action _A<sub>i</sub>_. 
- _y<sub>i</sub>=R<sub>i</sub>+γQ<sup>-</sup> (S<sub>i+1</sub>,a<sup>-</sup>;θ<sup>Q<sup>-</sup></sup>)_ where _a<sup>-</sup>=μ<sup>-</sup>(S<sub>i+1</sub>;θ<sup>μ<sup>-</sup></sup>))_ is the action produced by the target actor network when given the next state _S<sub>i+1</sub>_.

Update actor parameters using the sampled policy gradient:

<p align="center">
  <img src="https://user-images.githubusercontent.com/47497135/135069066-83354a1e-30d5-4eca-9cb8-6f9f66f51c3a.png" alt="drawing" width="350"/>
</p>

where _a=μ(S<sub>i</sub>;θ<sup>μ</sup>)_. This is simply calculated by applying the actor local network on the current state S<sub>i</sub> producing the predicted action _a_. Then applying the critic local network on the current state S<sub>i</sub> and predicted action a producing the action-value which represents the expected reward _J_. The loss function of the actor network will be _loss=-J_ because we want to maximize J not minimize it. So we wil minimize _loss_.

### Soft Update of Target Network
Instead of updating the target network parameters every number of steps. The target network parameters are updated at every step decayed by a parameter &tau;:

<p align="center">
  <img src="https://user-images.githubusercontent.com/47497135/135068260-df723efb-d2f9-40d0-b6be-e435628f3c3a.png" alt="drawing" width="250"/>
</p>

The hyperparameters are:
* Soft update parameter &tau;: `0.001`.

### Learn Every n Time Steps
Instead of calculating the gradients and updating the parameters ever time step _t_, we calculate the gradients and update the parameters every _n_ time steps. This prevents the problem of exploding gradients and stablizes the learning process.

The hyperparameters are:
* n = `1`

## 3. Plot of Rewards
For the environment to be solved, the average reward over 100 episodes must reach at least 0.5. The implementation provided here needed just around `3250 episodes` to be completed! The average score reached `0.73` after `3900 episodes`. The plot of rewards per episode is shown in _fig.3_. The plot of the rewards running average over 100 episodes is shown in _fig.4_.

<p align="center">
  <img src="https://user-images.githubusercontent.com/47497135/135778790-3713bb52-66c8-4d23-b97e-06181db62554.png" alt="drawing" width="400"/>
</p>
<p align="center">
  <em>Fig.3: Rewards Plot in 5000 episodes</em>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/47497135/135779012-77dd64e1-96e5-4cb7-9790-bee7b11fd469.png" alt="drawing" width="400"/>
</p>
<p align="center">
  <em>Fig.4: Rewards Running Average Plot over 100 episodes for 5000 episodes</em>
</p>


## 4. Ideas for Future Work
Some additional features could be used to provide better performance:
* __Prioritized Experience Replay:__ Instead of randomly choosing the expriences, we choose them based on how much they affect our learning process. For more information read this paper [here](https://arxiv.org/abs/1511.05952).
* __Different Algorithms:__ According to [this paper](https://arxiv.org/pdf/1604.06778.pdf), which benchmarks the different algorithms based on the applications, using Trust Region Policy Optimization (TRPO) and Truncated Natural Policy Gradient (TNPG) should achieve better performance.
* __Using images:__ Instead of states, we use the image of the game itself as an input to the neural network. We then have to introduce some changes to the architecture of the netwok. We could use convolutional layers. This will be a more challenging problem.
