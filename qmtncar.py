#! /usr/bin/env python
import datetime
import enum
from importlib.resources import path
import gym
import logging
import numpy as np
import pandas as pd
import pathlib
import os


POSITION_INDEX = 0
VELOCITY_INDEX = 1
MAX_STEPS_PER_EPISODE = 200

class Action(enum.Enum):
    # Constants for action space:
    ACCELERATE_LEFT = 0
    NO_ACCELERATION = 1
    ACCELERATE_RIGHT = 2


class State:
    def __init__(self, observation, binned):
        self.observation = observation
        self.binned      = binned
    
    @property
    def position(self):
        return self.observation[POSITION_INDEX]
    
    @property
    def velocity(self):
        return self.observation[VELOCITY_INDEX]

class StateSpace:


    LOGGER = logging.getLogger('StateSpace')

    def __init__(self, env: gym.Env, **kwargs):
        self.env        = env
        position_min    = env.observation_space.low[POSITION_INDEX]
        position_max    = env.observation_space.high[POSITION_INDEX]
        velocity_min    = env.observation_space.low[VELOCITY_INDEX]
        velocity_max    = env.observation_space.high[VELOCITY_INDEX]

        StateSpace.LOGGER.info(f'Position ranges from {position_min:.4f} to {position_max:.4f}')
        StateSpace.LOGGER.info(f'Position ranges from {velocity_min:.4f} to {velocity_max:.4f}')
        
        # Discretize each dimension to manageable bins.
        position_bins = kwargs.get('position_bins', 20)
        velocity_bins = kwargs.get('velocity_bins', 20)
        self.bins     = [position_bins, velocity_bins]
        self.bin_size = (env.observation_space.high - env.observation_space.low) / self.bins

        LOGGER.info(f'Position is divided into {self.bins[POSITION_INDEX]} bins of size {self.bin_size[POSITION_INDEX]:.3f}m each')
        LOGGER.info(f'Velocity is divided into {self.bins[VELOCITY_INDEX]} bins of size {self.bin_size[VELOCITY_INDEX]:.3f}m/s each')

    def observation_state(self, observation) -> State:
        binned_observation = (observation - self.env.observation_space.low) / self.bin_size
        binned_observation = tuple(binned_observation.astype(int))
        # QMtnCar.LOGGER.debug(f'Discrete observation ({observation[QMtnCar.POSITION_INDEX]:.4f}, {observation[QMtnCar.VELOCITY_INDEX]:.4f}) binned to {binned_observation}')

        return State(observation, binned_observation)

class QTable:
    LOGGER = logging.getLogger('QTable')

    def __init__(self, state_space: StateSpace, env, learning_rate, discount):
        self.state_space    = state_space
        self.env            = env
        self.learning_rate  = learning_rate
        self.discount       = discount

        # Uniform distribution over 20 x 20 x 3 (position "bins" x velocity "bins" x num action)
        #
        # np.random.uniform utilizes the half-open interval [low, high)
        self.q_table = np.random.uniform(low=-2, high=0, size=(state_space.bins + [env.action_space.n]))

    def select_action(self, state, epsilon=1) -> Action:
        rand    = np.random.random()
        explore = rand <= epsilon

        if explore:
            action = Action(self.env.action_space.sample())
        else:
            state_q = self.q_table[state.binned]
            action = Action(np.argmax(state_q))
        
        # QTable.LOGGER.debug(f'Action: {action.name} (explore={explore})')
        return action

    def update(self, current_state: State, action: Action, next_state: State, reward, goal_achieved) -> float:
        index   = (*current_state.binned, action.value)

        if goal_achieved:
            q_new = reward
        else:
            q_max       = np.max(self.q_table[next_state.binned])
            q_current   = self.q_table[index]
            q_delta     = self.learning_rate * (reward + (self.discount * q_max) - q_current)
            q_new       = q_current + q_delta

            # QTable.LOGGER.debug(f'QTable[{index} .. action: {action.name}, reward: {reward}, current: {q_current}, new: {q_new}')

        self.q_table[index] = q_new
        
        return q_new

class QMtnCar:

    LOGGER = logging.getLogger('QMtnCar')

    def __init__(self, learning_rate = 0.1, discount = 0.95, **kwargs):
        self.env            = gym.make("MountainCar-v0")
        self.state_space    = StateSpace(self.env, **kwargs)
        self.qtable         = QTable(self.state_space, self.env, learning_rate, discount)

        QMtnCar.LOGGER.info(f'Goal position: {self.env.goal_position}')

        self.metrics_df = pd.DataFrame()

    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        self.env.close()

    def run_episode(self, episode_num, epsilon = 1, render = False):
        observation_0   = self.env.reset()
        current_state   = self.state_space.observation_state(observation_0)

        step    = 0
        done    = False

        try:
            while not done:
                step += 1

                action = self.qtable.select_action(current_state, epsilon=epsilon)

                observation, reward, done, _ = self.env.step(action.value)

                next_state = self.state_space.observation_state(observation)

                if render:
                    self.env.render()

                goal_achieved = (next_state.position >= self.env.goal_position)

                if goal_achieved:
                    reward = MAX_STEPS_PER_EPISODE - step
                    QMtnCar.LOGGER.info(f'Episode {episode_num}, step {step}: GOAL ACHIEVED!!!!!  Reward: {reward}')

                self.metrics_df = pd.concat([self.metrics_df,
                                             pd.DataFrame([{
                                                'episode':  episode_num,
                                                'step':     step,
                                                'action':   action.value,
                                                'reward':   reward,
                                                'position': next_state.position,
                                                'velocity': next_state.velocity
                                             }])],
                                             ignore_index=True)

                self.qtable.update(current_state, action, next_state, reward, goal_achieved)

                current_state = next_state
        except KeyboardInterrupt:
            QMtnCar.LOGGER.warning('Episode interrupted.  Shutting down...')
            raise
        except:
            QMtnCar.LOGGER.exception('Unexpected error')
            raise



if __name__ == '__main__':
    import argparse

    DEFAULT_LEARNING_RATE: float    = 0.10
    DEFAULT_DISCOUNT: float         = 0.95
    DEFAULT_NUM_EPISODES: int       = 1000

    parser = argparse.ArgumentParser(description='Run the MountainCar Q-Learning program')
    parser.add_argument('-v',
                        '--verbose',
                        action='store_true')
    parser.add_argument('-r',
                        '--learning-rate',
                        type=float,
                        default=DEFAULT_LEARNING_RATE,
                        help=f'Learning rate (defaults to {DEFAULT_LEARNING_RATE})')
    parser.add_argument('-d',
                        '--discount',
                        type=float,
                        default=DEFAULT_DISCOUNT,
                        help=f'Discount (defaults to {DEFAULT_DISCOUNT})')
    parser.add_argument('-p',
                        '--progress',
                        default=1000,
                        type=int,
                        metavar='EPISODE_INTERVAL',
                        help=f'How often (in units of episodes) to report progress.  The first and last episode will always be reported')
    parser.add_argument('--render-every',
                        default=0,
                        type=int,
                        metavar='EPISODE_INTERVAL',
                        help=f'How often (in units of episodes) to graphically render the episode.  If non-zero, the first episode will always be rendered')
    parser.add_argument('--render-after',
                        default=0,
                        type=int,
                        metavar='EPISODE_NUM',
                        help=f'The episode after which render-every should take effect')
    parser.add_argument('-s',
                        '--save',
                        type=str,
                        help=f'Specifies a path to save the metrics DataFrame (in CSV format).  Filename will be <path>/mtn-car-<episodes>-episodes-<datetime>.csv')
    parser.add_argument('episodes',
                        nargs='?',
                        type=int,
                        default=DEFAULT_NUM_EPISODES,
                        help=f'Number of episodes to run (defaults to {DEFAULT_NUM_EPISODES})')

    args = parser.parse_args()

    os.environ['SDL_VIDEODRIVER']='x11'
    os.environ['SDL_AUDIODRIVER']='dsp'

    # Set up logging ...
    logging.basicConfig(format='%(asctime)s - %(levelname)-7s - %(name)10s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG if args.verbose else logging.INFO)

    LOGGER = logging.getLogger('qmtncar')

    # Exploration settings
    epsilon                 = 1  # not a constant, going to be decayed
    START_EPSILON_DECAYING  = 1
    END_EPSILON_DECAYING    = max(START_EPSILON_DECAYING + 1, args.episodes // 2)
    epsilon_decay_value     = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

    LOGGER.debug(f'Epsilon decay: {epsilon_decay_value} ({START_EPSILON_DECAYING}, {END_EPSILON_DECAYING})')

    with QMtnCar(learning_rate=args.learning_rate, discount=args.discount) as car:
        try:
            for episode_num in range(1, args.episodes + 1):
                report_progress = (episode_num == 1) or (episode_num == args.episodes) or (episode_num % args.progress == 0)
                render = (args.render_every > 0) and (episode_num > args.render_after) and ((episode_num == 1) or (episode_num % args.render_every == 0))

                if report_progress:
                    LOGGER.info(f'Starting episode {episode_num} (epsilon={epsilon:.4f})')

                car.run_episode(episode_num, epsilon=epsilon, render=render)
                    
                # Decaying is being done every episode if episode number is within decaying range
                if START_EPSILON_DECAYING < episode_num < END_EPSILON_DECAYING:
                    epsilon -= epsilon_decay_value
        except KeyboardInterrupt:
            pass

        if args.save:
            pathlib.Path(args.save).mkdir(parents=True, exist_ok=True)

            filename    = f'mtn-car-{episode_num}-episodes-{datetime.datetime.now().isoformat(timespec="seconds")}.csv'
            path        = os.path.join(args.save, filename)
            car.metrics_df.to_csv(path)
            LOGGER.info(f'Wrote {path}')
