import logging
import numpy as np
from src.toyenvironment import ToyEnvironment as Environment


def train(agent, num_episodes, max_num_steps_per_episode, epsilon, epsilon_min, epsilon_decay,
          results, scores_average_window):
    for i_episode in range(1, num_episodes + 1):
        # reset the environment
        initial_variables = np.array([1.0, 0.0, 24.0, 13.0, 7.0, 23.0, 24.0, 2.0], dtype=np.float32)
        target_indicators = np.array([2.5, 0.5, 10.5, 7.0, 3.5, 5.0, 11.0, 4.5], dtype=np.float32)
        env = Environment(initial_variables, target_indicators)

        # get initial state of the unity environment
        state = env.episode['state'][-1]

        done = env.episode['done'][-1]
        if done:
            logging.info('Convergence criteria met')
            continue

        # set the initial episode score to zero.
        score = 0

        for i_step in range(1, max_num_steps_per_episode + 1):
            valid_actions = env.episode['valid_actions'][-1]

            # determine epsilon-greedy action from current sate
            action = agent.act(state, valid_actions, epsilon)

            # send the action to the environment
            env.step(action)

            next_state = env.episode['state'][-1]    # get the next state
            reward = env.episode['reward'][-1]       # get the reward
            done = env.episode['done'][-1]           # see if episode has finished

            #Send (S, A, R, S') info to the DQN agent for a neural network update
            agent.step(state, action, reward, next_state, done)

            # set new state to current state for determining next action
            state = next_state

            # Update episode score
            score += reward

            # If this episode is done,
            # then exit episode loop, to begin new episode
            if done:
                break

        # Add episode score to Scores and...
        # Calculate mean score over last 'scores_average_window' episodes
        # Mean score is calculated over current episodes until i_episode > 'scores_average_window'
        results['scores'].append(score)
        average_score = np.mean(results['scores'][i_episode - min(i_episode, scores_average_window):i_episode + 1])

        # Decrease epsilon for epsilon-greedy policy by decay rate
        # Use max method to make sure epsilon doesn't decrease below epsilon_min
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        # (Over-) Print current average score
        # logging.info('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, average_score), end="")

        # Print average score every scores_average_window episodes
        if i_episode % scores_average_window == 0:
            logging.info('Episode {}\tAverage Score: {:.2f}'.format(i_episode, average_score))


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
