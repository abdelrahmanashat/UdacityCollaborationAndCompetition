# Udacity Reinforcement Learning Nano Degree: Collaboration and Competetion Project

In this project, we will implement a DDPG (Deep Deterministic Policy Gradient) agent to solve Unity's Tennis environment.

## 1. Environment Details: 
In this project, we have to train 2 agents to play tennis against each other.

<p align="center">
  <img src="https://user-images.githubusercontent.com/25223180/50571890-b3146280-0dda-11e9-8f77-3ffcea5df3f2.gif" alt="drawing" width="400"/>
</p>

NOTE:

1. This project was completed in the Udacity Workspace, but the project can also be completed on a local Machine. Instructions on how to download and setup Unity ML environments can be found in [Unity ML-Agents Github repo](https://github.com/Unity-Technologies/ml-agents).
1. The environment provided by Udacity is similar to, but not identical to the Reacher environment on the [Unity ML-Agents Github page](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md).

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of `+0.1`. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of `-0.01`. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of `+0.5` (over `100` consecutive episodes, after taking the maximum over both agents). Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the __maximum__ of these 2 scores.
* This yields a single __score__ for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least `+0.5`.

## 2. Requirements for running the code:

#### Step 1: Install Numpy
Follow the instructions [here](https://numpy.org/install/) to install the latest version of Numpy.

#### Step 2: Install Pytorch and ML-Agents
If you haven't already, please follow the instructions in the [DRLND (Deep Reinforcement Learning Nano Degree) GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment. These instructions can be found in `README.md` at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

_(For Windows users)_ The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

#### Step 3: Download the Reacher Environment
For this project, you will __not__ need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

- Linux: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Then, place the file in the `p2_continuous-control/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.

_(For Windows users)_ Check out this link if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

_(For AWS)_ If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use this link to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to enable a virtual screen, and then download the environment for the Linux operating system above.)

## 3. Explore the Environment
After you have followed the instructions above, open `Collaboration_and_Competition.ipynb` located in the `project_collaboration_and_competition/` folder and follow the instructions to learn how to use the Python API to control the agent. The saved weights files are 4 files named `checkpoint_actor_0.pth`, `checkpoint_critic_0.pth`, `checkpoint_actor_1.pth`, and `checkpoint_critic_1.pth` located in the `project_collaboration_and_competition/weights/` folder. 

## 4. Implementation Details

### Introduction
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

### Learning Algorithm
#### Experience Replay
When the agent interacts with the environment, the sequence of experience tuples can be highly correlated. The naive Q-learning algorithm that learns from each of these experience tuples in sequential order runs the risk of getting swayed by the effects of this correlation. By instead keeping track of a __replay buffer__ and using __experience replay__ to sample from the buffer at random, we can prevent action values from oscillating or diverging catastrophically.

The __replay buffer__ contains a collection of experience tuples __*(S, A, R, S')*__. The tuples are gradually added to the buffer as we are interacting with the environment.

The act of sampling a small batch of tuples from the replay buffer in order to learn is known as __experience replay__. In addition to breaking harmful correlations, experience replay allows us to learn more from individual tuples multiple times, recall rare occurrences, and in general make better use of our experience.

The hyperparameters are:
* Buffer size: `1,000,000`.

#### Fixed Q-Targets
In Q-Learning, we update a guess with a guess, and this can potentially lead to harmful correlations. To avoid this, we can update the parameters __*w*__ in the network $Q_hat$ to better approximate the action value corresponding to state __*S*__ and action __*A*__ with the following update rule:

<p align="center">
  <img src="https://user-images.githubusercontent.com/47497135/134077362-35bad85b-cb3e-41fc-9d35-35f2a66201ca.png" alt="drawing" width="600"/>
</p>

where __*w<sup>-</sup>*__ are the weights of a separate target network that are not changed during the learning step, and __*(S, A, R, S')*__ is an experience tuple.

The hyperparameters are:
* Discount factor &gamma;: `0.99`.

#### Ornstein-Uhlenbeck Noise Process 
A major challenge of learning in continuous action spaces is exploration. An advantage of offpolicies algorithms such as DDPG is that we can treat the problem of exploration independently from the learning algorithm. We constructed an exploration policy _μ'_ by adding noise sampled from a noise process _N_ to our actor policy:

<p align="center">
  <img src="https://user-images.githubusercontent.com/47497135/135072151-3a25a5d1-599d-49a9-b129-1d92d63bdbf2.png" alt="drawing" width="220"/>
</p>

_N_ can be chosen to suit the environment. As detailed in the supplementary materials we used an Ornstein-Uhlenbeck process to generate temporally correlated exploration for exploration efficiency in physical control problems with inertia.

The hyperparameters are:
* &sigma;: `0.20`.
* &theta;: `0.15`.

#### Update Rules
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

#### Soft Update of Target Network
Instead of updating the target network parameters every number of steps. The target network parameters are updated at every step decayed by a parameter &tau;:

<p align="center">
  <img src="https://user-images.githubusercontent.com/47497135/135068260-df723efb-d2f9-40d0-b6be-e435628f3c3a.png" alt="drawing" width="250"/>
</p>

The hyperparameters are:
* Soft update parameter &tau;: `0.001`.

#### Learn Every n Time Steps
Instead of calculating the gradients and updating the parameters ever time step _t_, we calculate the gradients and update the parameters every _n_ time steps. This prevents the problem of exploding gradients and stablizes the learning process.

The hyperparameters are:
* n = `1`

## 5. Plot of Rewards
For the environment to be solved, the average reward over the 20 agents over 100 episodes must reach at least 30. The implementation provided here needed just `155 episodes` to be completed! The average score reached `50` after `207 episodes`. The plot of rewards is shown in _fig.3_.

<p align="center">
  <img src="https://user-images.githubusercontent.com/47497135/135637162-c3cebfcf-e62b-4571-8e35-ca31b26cfe96.png" alt="drawing" width="400"/>
</p>
<p align="center">
  <em>Fig.3: Rewards Plot in 500 episodes for 20 agents</em>
</p>

## 6. Ideas for Future Work
Some additional features could be used to provide better performance:
* __Prioritized Experience Replay:__ Instead of randomly choosing the expriences, we choose them based on how much they affect our learning process. For more information read this paper [here](https://arxiv.org/abs/1511.05952).
* __Different Algorithms:__ According to [this paper](https://arxiv.org/pdf/1604.06778.pdf), which benchmarks the different algorithms based on the applications, using Trust Region Policy Optimization (TRPO) and Truncated Natural Policy Gradient (TNPG) should achieve better performance.
* __Using images:__ Instead of states, we use the image of the game itself as an input to the neural network. We then have to introduce some changes to the architecture of the netwok. We could use convolutional layers. This will be a more challenging problem.
