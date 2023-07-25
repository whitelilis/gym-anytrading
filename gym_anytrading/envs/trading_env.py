import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import math
import pysnooper


class Actions(Enum):
    Sell = 0
    Buy = 1
    Hold = 2


class Positions(Enum):
    Short = 0
    Long = 1
    Empty = 2


class TradingEnv(gym.Env):

    metadata = {'render_modes': []}

    def __init__(self, df, window_size):
        assert df.ndim == 2

        self.seed()
        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = window_size * self.signal_features.shape[
            1] + 2  # add position, total_reward
        print(f"shape is {self.shape}")
        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(self.shape, ),
                                            dtype=np.float64)

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._done = False
        self._truncated = False
        self._current_tick = None
        self._last_trade_tick = None
        self._position = Positions.Empty
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self._last_step_fee = 0.0
        self._last_step_reward = 0.0
        self._last_penalty = 0.0
        self._last_position_reward = 0.0
        self.history = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, **kwarg):
        self.seed(seed)
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Empty
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}
        return self._get_observation()

    #@pysnooper.snoop(watch=('ac', 'self._position'))
    def _update_position(self, ac):
        if self._position == Positions.Empty:
            if ac == Actions.Buy.value:  # action maybe generate by model, which is a number, not Enum.
                self._position = Positions.Long
            elif ac == Actions.Sell.value:
                self._position = Positions.Short
            else:
                print(f"{ac} != {Actions.Sell.value} nor {Actions.Buy.value}")
        elif self._position == Positions.Long and ac == Actions.Sell.value:
            self._position = Positions.Empty
        elif self._position == Positions.Short and ac == Actions.Buy.value:
            self._position = Positions.Empty

    def step(self, action):
        ac = action.item(0)
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        step_reward = self._calculate_reward(ac)
        self._total_reward += step_reward

        self._update_profit(ac)
        self._last_trade_tick = self._current_tick

        # update position
        self._update_position(ac)
        self._position_history.append(self._position)
        observation, info = self._get_observation(ac)
        self._update_history(info)
        return observation, step_reward, self._done, self._truncated, info

    def _get_observation(self, ac=None):
        info = {
            "ac": ac,
            "last_step_fee": self._last_step_fee,
            "last_penalty": self._last_penalty,
            "last_position_reward": self._last_position_reward,
            "last_step_reward": self._last_step_reward,
            "total_reward": self._total_reward,
            "total_profit": self._total_profit,
            "position": self._position.value
        }
        tick_feature = self.signal_features[(self._current_tick -
                                             self.window_size +
                                             1):self._current_tick +
                                            1].flatten()
        obs = np.append(tick_feature, self._position.value)  # only support int
        obs = np.append(obs, int(self._total_reward * 100))  # only support int
        return obs, info

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def render(self, mode='human'):

        def _plot_position(position, tick):
            color = None
            if position == Positions.Short:
                color = 'green'
            elif position == Positions.Long:
                color = 'red'
            else:  # no position
                color = "black"
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.subtitle("Total Reward: %.6f" % self._total_reward + ' ~ ' +
                     "Total Profit: %.6f" % self._total_profit)

        plt.pause(0.01)

    def render_all(self, mode='human'):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        empty_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.Short:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.Long:
                long_ticks.append(tick)
            else:
                empty_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], 'go')
        plt.plot(long_ticks, self.prices[long_ticks], 'ro')
        plt.plot(empty_ticks, self.prices[empty_ticks], 'bo')

        plt.suptitle("Total Reward: %.6f" % self._total_reward + ' ~ ' +
                     "Total Profit: %.6f" % self._total_profit)

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _process_data(self):
        raise NotImplementedError

    def _calculate_reward(self, ac):
        raise NotImplementedError

    def _update_profit(self, ac):
        raise NotImplementedError

    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError
