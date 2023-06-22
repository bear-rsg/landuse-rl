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

        # If this episode is then exit
        if done:
            return None


# def evaluate(agent, max_num_steps_per_episode):
#     initial_variables = np.array([1.0, 0.0, 24.0, 13.0, 7.0, 23.0, 24.0, 2.0], dtype=np.float32)
#     target_indicators = np.array([2.5, 0.5, 10.5, 7.0, 3.5, 5.0, 11.0, 4.5], dtype=np.float32)
#     env = Environment(initial_variables, target_indicators)

#     logging.info(f"Target indicators: {target_indicators}")

#     # get initial state of the unity environment
#     state = env.episode['state'][-1]

#     done = env.episode['done'][-1]
#     if done:
#         logging.info('Convergence criteria met')

#     logging.info(f"Current indicators: {env.episode['indicators'][-1]}")

#     for i_step in range(1, max_num_steps_per_episode+1):
#         valid_actions = env.episode['valid_actions'][-1]

#         # determine epsilon-greedy action from current sate
#         action = agent.act(state, valid_actions)

#         # send the action to the environment
#         env.step(action)

#         logging.info(f"Current indicators: {env.episode['indicators'][-1]}")

#         next_state = env.episode['state'][-1]    # get the next state
#         reward = env.episode['reward'][-1]       # get the reward
#         done = env.episode['done'][-1]           # see if episode has finished

#         # set new state to current state for determining next action
#         state = next_state

#         if done:
#             break

#     # logging.info(f"Target indicators: {target_indicators}")
