# Multi Agent Deterministic Deep Policy Gradient

This project is an implementation of a the Multi Agent Deterministic Deep Policy Gradient algorithm for solving a unity environment.

### Required packages:
- numpy
- python (version 3.6)
- pytorch
- unityagents 

### Dependencies

The best way to run the code in this repository is to create a  conda environment by following the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
3. Clone this repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/jpruente92/RL_class_project_1
cd RL_class_project_1/python
pip install .
```
4. Use the dlrnd environment for starting the program.

### Required files:
The unity exe file has to be inside this folder; here is an environment for Windows (64-bit) called "Tennis.exe" included. If you do not have Windows (64-bit), you can download the environment with one of the following links:
- Linux (https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac (https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit) (https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)

### Tennis environment:
In the Tennis environment, two agents try to bounce a ball over a net. If an agent hits the ball over the net, 
it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. 
Thus, the goal of each agent is to keep the ball in play.
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. 
Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) 
the net, and jumping.
The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,
After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

### Starting the program:
- hyperparameters and settings for the algorithm can be changed in the file "hyperparameters.py".
    -> for viewing a trained agent set "LOAD" to True,"FILENAME_FOR_LOADING" to the name of the files of the model
    weights("tennis_trained") and "ENV_TRAIN" to False.
    -> for training a new agent set "LOAD" to False and "ENV_TRAIN" to True and if you want to
    save the model weights, set "Save" to True and "FILENAME_FOR_SAVING" to the name for the weight files
- after the hyperparameters are set, the file "main.py" has to be started
- changes in all other files are not recommended.


