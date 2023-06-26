import logging


def train(agent, env, max_num_steps_per_episode, epsilon):
    # Get initial state of the environment
    state = env.episode['state'][-1]

    done = env.episode['done'][-1]
    if done:
        logging.info('Convergence criteria met. No steps needed for this episode.')
        return None

    for i_step in range(1, max_num_steps_per_episode + 1):
        valid_actions = env.episode['valid_actions'][-1]

        # Determine epsilon-greedy action from current sate
        action = agent.act(state, valid_actions, epsilon)

        # Send the action to the environment
        env.step(action)

        next_state = env.episode['state'][-1]    # get the next state
        reward = env.episode['reward'][-1]       # get the reward
        done = env.episode['done'][-1]           # see if episode has finished

        # Send (S, A, R, S') info to the DQN agent for a neural network update
        agent.step(state, action, reward, next_state, done)

        # Set new state to current state for determining next action
        state = next_state

        # If this episode converged then exit
        if done:
            return None


def evaluate(agent, env, max_num_steps_per_episode):
    # Get initial state of the environment
    state = env.episode['state'][-1]

    done = env.episode['done'][-1]
    if done:
        logging.info('Convergence criteria met')
        return None

    for i_step in range(1, max_num_steps_per_episode + 1):
        valid_actions = env.episode['valid_actions'][-1]

        # Determine epsilon-greedy action from current sate
        action = agent.act(state, valid_actions)

        # Send the action to the environment
        env.step(action)

        next_state = env.episode['state'][-1]    # get the next state
        reward = env.episode['reward'][-1]       # get the reward
        done = env.episode['done'][-1]           # see if episode has finished

        # Set new state to current state for determining next action
        state = next_state

        # If this episode converged then exit
        if done:
            return None
