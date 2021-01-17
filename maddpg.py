import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from hyperparameters import *

from multi_agent import Multi_agent


def start_maddpg_tennis():
    from unityagents import UnityEnvironment
    env = UnityEnvironment(file_name='./Tennis_Windows_x86_64/Tennis.exe')
    brain_name = env.brain_names[0]
    multi_agent = Multi_agent(2, 24, 2)
    maddpg(multi_agent, env, brain_name, 2)


def maddpg(multi_agent, env, brain_name, nr_agents):
    all_scores = []
    scores_window = deque(maxlen=100)
    all_average_scores = []
    step = 0

    for i_episode in range(1, MAX_NR_EPISODES + 1):
        env_info = env.reset(train_mode=TRAINMODE)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(nr_agents)
        start_time = time.time()
        while True:
            actions = multi_agent.act(states)

            # collect data from environment
            env_info = env.step(actions)[brain_name]  # send all actions to the environment
            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished
            scores += rewards

            # save experience in shared replay buffer
            multi_agent.memory.add(states, actions, rewards, next_states, dones)

            # if enough samples are available in memory, every agent learns separately!
            step = (step + 1)
            if step % UPDATE_EVERY == 0 and len(multi_agent.memory) > BATCH_SIZE and TRAINMODE:
                multi_agent.learn()
            states = next_states
            if np.any(dones):
                break

        # statistics
        score_this_episode = np.max(scores)
        scores_window.append(score_this_episode)
        all_scores.append(score_this_episode)
        average_score = np.mean(scores_window)
        all_average_scores.append(average_score)
        time_for_episode = time.time() - start_time

        # save neural networks
        if (SAVE):
            multi_agent.save_networks()
        print(
            'Episode {}\tTime {}\tScore this episode (maximum over agents): {:.2f}\tAverage Score last 100 episodes (maximum over agents): {:.2f}'.format(
                i_episode, time_for_episode, score_this_episode, np.mean(scores_window)))

        if np.mean(scores_window) >= VAL_ENV_SOLVED:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            # plot the scores
            if PLOT:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.plot(np.arange(len(all_scores)), all_scores, label="score")
                plt.plot(np.arange(len(all_scores)), all_average_scores, label="average score over last 100 episodes")
                plt.ylabel('Score')
                plt.xlabel('Episode #')
                plt.legend()
                plt.savefig("Scores/"+PLOTNAME)
                plt.show()
            break

    env.close()
