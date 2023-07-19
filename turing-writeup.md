# Technical Description of the Algorithm

The source files, installation and packaging of the algorithm are described in the repository README file (see the class diagrams and sequence diagram). This document describes the algorithm itself.

## Introduction

The algorithm is based on the following assumptions.

There are 719 areas based on geographic divisions from Newcastle Upon Tyne, each with four variables of interest:

* Signature Type, int range 0-15 (`sig_type_interval`, `sig_type_step`)
* Land Use, real range -1..1, discretised by default into 21 points (`land_use_interval`, `land_use_n_points`)
* Green Space, real range 0..1, discretised by default into 11 points (`greenspace_interval`, `greenspace_n_points`)
* Job Type, real range 0..1, discretised by default into 11 points (`job_type_interval`, `job_type_n_points`)

**Note:** If we change those values, then we need to re-train the model.

The Turing Institute have provided a Black Box algorithm that can return four indicators of interest:

* Air Quality
* House Prices
* Job Accessibility
* Green Space Accessibility

Each indicator is assigned a maximum value for normalisation purposes (`max_air_quality`, `max_house_price`, `max_job_accessibility`, `max_greenspace_accessibility`).

We construct a Deep Neural Network with 3 layers (input, hidden and output). The input layer has `state_size = 719 * 4` neurons (number of areas × number of indicators per area). In the input layer we pass the difference between the current and target indicators after we flatten and normalise them. The output layer has `action_size = 719 * 4 * 2` neurons (number of areas × number of variables per area × number of actions per variable, which is either increase it or decrease it). In the output layer we get the Q value of each action.

Given initial and target indicators for each area, the algorithm will attempt to find a trajectory (a sequence of actions) that will minimise the difference between the initial and target indicators, using a Q Learning approach with a Deep Neural Network (DNN). The reason for using a DNN with Q Learning is so that the DNN can act as a function 'approximator' instead of storing a table of Q values for each state-action pair. This is necessary because the number of states is too large to store in memory and scales exponentially with the number of indicators.

## The DQN Algorithm

The DQN algorithm can be summarised in the following diagram.

<img src="./docs/DQN1.png" alt="The DQN algorithm" width="450" />

## The Classes Used in the Algorithm

The algorithm is implemented in Python. The following classes are used.

<img src="./docs/classes.png" alt="class diagram" width="450" />

## The Sequence of Events for the Algorithm

The class and method calls for the algorithm are shown in the following sequence diagram.

<img src="./docs/plantUML_Seq.png" alt="package diagram" width="450" /> 

The box labelled `until max episode` describes the sequence that is key to understanding the implementation of the algorithm. This box is refers to the loop that is defined with the following code:

```python

    # Start the episodes loop to train
    for i_episode in range(1, num_episodes + 1):
```

First, a target is selected from one of seven target indicator text files, provided by the Turing Institute. The train function is called using an Epsilon Greedy Policy for the action selection. The epsilon value decreases over time in order to reduce the amount of exploration and increase the amount of exploitation. This allows the selection of the best action for a given state, based on the current Q values.

The `step` function places `Experience` into the replay buffer. The size of the buffer is set to `replay_memory_size` and is designed to break the correlation between consecutive experiences. Every `update_rate` number of steps, the `learn` function samples a batch of `batch_size` experiences from the replay buffer and using mini-batch gradient descent updates the weights and biases of the DQN, using the Bellman equation. The inner loop (`for i_step in range(1, max_num_steps_per_episode + 1)`) ends when the maximum number of steps per episode has been reached, or we have converged to the target, if this happens earlier. The outer loop (`for i_episode in range(1, num_episodes + 1)`) ends when the maximum number of episodes has been reached.

### Pseudocode

```
For each episode:
	Randomly select a target for this episode.
	
	For each step in this episode:
		Using an epsilon-greedy policy select an action.
		Perform the action, reach a new state and get a reward.
		Add (state, action, new state, reward) to the replay buffer.
		Every update_rate steps:
			Sample a batch_size mini-batch from
			 the replay buffer.
			Compute the target Q values using a Double DQN approach
			 and the Bellman equation.
			Perform gradient descent.
			Soft-update the target network parameters.
```

## Modifying the Algorithm

The number of areas, output areas, and input areas are hard-coded in the algorithm, these can be edited to suit different sized data. The algorithm assumes 719 areas, so to change look for these values and alter to suit.

```python
state_size = 719 * 4
action_size = 719 * 4 * 2
```


### Number of Areas

This is set to 719 in the code, to match the data supplied by the Turing Institute. The code can be modified to use a different number of areas, but the data will need to be modified to match.

### Number or Output Areas

This is set to 4 in the code (x 2 to represent increase/decrease), to match the `actions` supplied by the Turing Institute. The code can be modified to use a different number of output areas, but the data will need to be modified to match.


### Number of Input Areas

This is set to 4 in the code, to match the `states` supplied by the Turing Institute. The code can be modified to use a different number of input areas, but the data will need to be modified to match.
