import numpy as np

from .trading_env import TradingEnv, Actions, Positions


class FutureEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(df, window_size)

        self._trade_fee = 0.1  # unit
        self._no_position_penalty = 0.01


    def _process_data(self):
        prices = self.df.loc[:, 'lastPrice'].to_numpy()

        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        self.df.loc[:, 'diff'] = self.df['lastPrice'].diff()
        signal_features = self.df.loc[:, ['diff', 'lastPrice', 'askVolume1', 'askVolume2', 'askVolume3', 'bidVolume1', 'bidVolume2', 'bidVolume3']].to_numpy()

        return prices, signal_features


    def _calculate_reward(self, ac):
        factor = 0
        if self._position == Positions.Long:
            factor = 1
        elif self._position == Positions.Short:
            factor = -1
        else:
            factor = 0

        diff = self.prices[self._current_tick] - self.prices[self._current_tick - 1]
        self._last_position_reward = factor * diff
        if ac == Actions.Hold.value or (self._position == Positions.Long and ac == Actions.Buy.value) or (self._position == Positions.Short and ac == Actions.Sell.value):
            self._last_fee = 0
        else:
            self._last_fee = self._trade_fee


        if (self._position == Positions.Long and ac == Actions.Buy.value) or (self._position == Positions.Short and ac == Actions.Sell.value) or (self._position == Positions.Empty):
            self._last_penalty = self._no_position_penalty
        else:
            self._last_penalty = 0

        hold_r = 1 if self._total_reward > 0 else -1

        self._last_step_reward = self._last_position_reward - self._last_step_fee - self._last_penalty
        return self._last_step_reward


    def _update_profit(self, ac):
        pass

    def max_possible_profit(self):
        pass