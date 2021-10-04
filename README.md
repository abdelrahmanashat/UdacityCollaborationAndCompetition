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
After you have followed the instructions above, open `Collaboration_and_Competition.ipynb` located in the `project_collaboration_and_competition/` folder and follow the instructions to learn how to use the Python API to control the agent. 

The saved weights files are 4 files named `checkpoint_actor_0.pth`, `checkpoint_critic_0.pth`, `checkpoint_actor_1.pth`, and `checkpoint_critic_1.pth` located in the `project_collaboration_and_competition/weights/` folder. 

## 4. Implementation Details
All the details of the implementation are included in the `Report.md` file located at the root of the repositary.
