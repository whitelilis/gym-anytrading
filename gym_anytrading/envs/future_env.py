import numpy as np

from .trading_env import TradingEnv, Actions, Positions


class FutureEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(df, window_size)

        self.trade_fee = 4  # unit


    def _process_data(self):
        prices = self.df.loc[:, 'lastPrice'].to_numpy()

        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = self.df.loc[:, ['lastPrice', 'askVolume1', 'askVolume2', 'askVolume3', 'bidVolume1', 'bidVolume2', 'bidVolume3']].to_numpy()

        return prices, signal_features


    def _calculate_reward(self, action):

        factor = 0
        if self._position == Positions.Long:
            factor = 1
        elif self._position == Positions.Short:
            factor = -1
        else:
            factor = 0

        diff = self.prices[self._current_tick] - self.prices[self._current_tick - 1]
        position_reward = 10 * factor * diff
        self._last_step_reward = position_reward
        commission = 0 
        if action == Actions.Hold.value or (self._position == Positions.Long and action == Actions.Buy.value) or (self._position == Positions.Short and action == Actions.Sell.value):
            commission = 0
        else:
            commission = -self.trade_fee
        self._last_fee = commission
        #print(f"diff: {diff}, action: {action}, position reward: {position_reward}, fee: {commission}")
        return position_reward + commission


    def _update_profit(self, action):
        pass

    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            if position == Positions.Long:
                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                shares = profit / last_trade_price
                profit = shares * current_price
            last_trade_tick = current_tick - 1

        return profit
