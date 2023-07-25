import numpy as np
import math
from .trading_env import TradingEnv, Actions, Positions


class FutureEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(df, window_size)

        self._trade_fee = 3  # unit
        self._no_position_penalty = 0.0
        self._invalidate_action_penalty = 10

    def _process_data(self):
        prices = self.df.loc[:, 'lastPrice'].to_numpy()

        prices[self.frame_bound[0] -
               self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0] -
                        self.window_size:self.frame_bound[1]]

        self.df.loc[:, 'diff'] = self.df['lastPrice'].diff()
        self.df['av1'] = self.df.apply(
            lambda row: math.log(row['askVolume1'], 10), axis=1)

        self.df['av2'] = self.df.apply(
            lambda row: math.log(row['askVolume2'], 10), axis=1)
        self.df['av3'] = self.df.apply(
            lambda row: math.log(row['askVolume3'], 10), axis=1)
        self.df['bv1'] = self.df.apply(
            lambda row: math.log(row['bidVolume1'], 10), axis=1)
        self.df['bv2'] = self.df.apply(
            lambda row: math.log(row['bidVolume2'], 10), axis=1)
        self.df['bv3'] = self.df.apply(
            lambda row: math.log(row['bidVolume2'], 10), axis=1)
        self.df['pec'] = self.df.apply(lambda row:
                                       (row['lastPrice'] - row['open']) * 2 /
                                       (row['upperLimit'] - row['lowerLimit']),
                                       axis=1)
        signal_features = self.df.loc[:, [
            'diff', 'pec', 'av1', 'av2', 'av3', 'bv1', 'bv2', 'bv3',
            'lastPrice'
        ]].to_numpy()

        return prices, signal_features

    def _calculate_reward(self, ac):
        factor = 0
        if self._position == Positions.Long:
            factor = 1
        elif self._position == Positions.Short:
            factor = -1
        else:
            factor = 0

        diff = self.prices[self._current_tick] - self.prices[self._current_tick
                                                             - 1]
        self._last_position_reward = 10 * factor * diff
        print(
            f"p-1: {self.prices[self._current_tick - 1]}, p: {self.prices[self._current_tick]},  po: {self._position},  po_reward: {self._last_position_reward}, ac: {Actions(ac)}"
        )
        self._last_penalty = 0
        if ac == Actions.Hold.value or (self._position == Positions.Long
                                        and ac == Actions.Buy.value) or (
                                            self._position == Positions.Short
                                            and ac == Actions.Sell.value):
            self._last_fee = 0
        else:
            self._last_fee = self._trade_fee

        if (self._position == Positions.Long and ac
                == Actions.Buy.value) or (self._position == Positions.Short
                                          and ac == Actions.Sell.value):
            self._last_penalty = self._last_penalty + self._invalidate_action_penalty
        elif self._position == Positions.Empty:
            self._last_penalty = self._last_penalty + self._no_position_penalty
        else:
            self._last_penalty = 0

        if self._total_reward > 50:
            self._done = True
        self._last_step_reward = self._last_position_reward - self._last_step_fee - self._last_penalty
        return self._last_step_reward

    def _update_profit(self, ac):
        pass

    def max_possible_profit(self):
        pass