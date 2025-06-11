from __future__ import annotations

from typing import List

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path # Added for directory creation

matplotlib.use("Agg")

# from stable_baselines3.common.logger import Logger, KVWriter, CSVOutputFormat


class StockTradingEnv(gym.Env):
    """
    A stock trading environment for OpenAI gym

    Parameters:
        df (pandas.DataFrame): Dataframe containing data
        hmax (int): Maximum cash to be traded in each trade per asset.
        initial_amount (int): Amount of cash initially available
        buy_cost_pct (float, array): Cost for buying shares, each index corresponds to each asset
        sell_cost_pct (float, array): Cost for selling shares, each index corresponds to each asset
        turbulence_threshold (float): Maximum turbulence allowed in market for purchases to occur. If exceeded, positions are liquidated
        print_verbosity(int): When iterating (step), how often to print stats about state of env
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: list[str],
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots: bool = False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount  # get the initial cash
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        # Ensure index is DatetimeIndex if it wasn't already set by the caller
        if not isinstance(self.df.index, pd.DatetimeIndex):
            try:
                 self.df.index = pd.to_datetime(self.df.index)
                 print("StockTradingEnv Info: Converted DataFrame index to DatetimeIndex.")
            except Exception as e:
                 print(f"StockTradingEnv Warning: Failed to convert index to DatetimeIndex: {e}")

        # Access data using iloc based on day and stock_dim
        # Assumes df has DatetimeIndex and is sorted by date, then tic
        self.data = self.df.iloc[self.day * self.stock_dim : (self.day + 1) * self.stock_dim]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        # initalize state
        self.state = self._initiate_state()

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1 : 1 + self.stock_dim])
            )
        ]  # the initial total asset is calculated by cash + sum (num_share_stock_i * price_stock_i)
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = (
            []
        )  # we need sometimes to preserve the state in the middle of trading process
        self.date_memory = [self._get_date()]
        #         self.logger = Logger('results',[CSVOutputFormat])
        # self.reset()
        self._seed()

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            # Check if price is positive (data available)
            if self.state[index + 1] > 0:
                if self.state[index + self.stock_dim + 1] > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 1]
                    )
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct[index])
                    )
                    # update balance
                    self.state[0] += sell_amount

                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (
                        self.state[index + 1]
                        * sell_num_shares
                        * self.sell_cost_pct[index]
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0: # Check price > 0
                    # if turbulence goes over threshold, just clear out all positions
                    if self.state[index + self.stock_dim + 1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                            self.state[index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct[index])
                        )
                        # update balance
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                            self.state[index + 1]
                            * sell_num_shares
                            * self.sell_cost_pct[index]
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
             # Check if price is positive
            if self.state[index + 1] > 0:
                available_amount = self.state[0] // (
                    self.state[index + 1] * (1 + self.buy_cost_pct[index])
                )
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.state[index + 1]
                    * buy_num_shares
                    * (1 + self.buy_cost_pct[index])
                )
                self.state[0] -= buy_amount
                self.state[index + self.stock_dim + 1] += buy_num_shares
                self.cost += (
                    self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                )
                self.trades += 1
            else:
                buy_num_shares = 0
            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass
        return buy_num_shares

    def _make_plot(self):
        # Ensure results directory exists
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        plt.plot(self.asset_memory, "r")
        plt.savefig(results_dir / f"account_value_trade_{self.episode}.png")
        plt.close()

    def step(self, actions):
        # Use unique dates in the index to determine terminal condition
        # Handle potential MultiIndex or DatetimeIndex
        if isinstance(self.df.index, pd.MultiIndex):
             num_unique_days = len(self.df.index.get_level_values(0).unique())
        else:
             num_unique_days = len(self.df.index.unique())

        self.terminal = self.day >= num_unique_days - 1

        if self.terminal:
            if self.make_plots:
                self._make_plot()
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = end_total_asset - self.asset_memory[0]

            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)

            sharpe = 0.0 # Default value
            # Calculate Sharpe Ratio only if standard deviation is non-zero and not NaN
            if df_total_value["daily_return"].std() != 0 and not np.isnan(df_total_value["daily_return"].std()):
                sharpe = (
                    (252**0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )

            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            if len(self.date_memory) > 1: # Check if there's more than the initial date
                 # Ensure the length matches rewards_memory
                 df_rewards["date"] = self.date_memory[1:] # Rewards correspond to steps taken

            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                print(f"Sharpe: {sharpe:0.3f}") # Print Sharpe even if 0
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                # Ensure results directory exists
                results_dir = Path("results")
                results_dir.mkdir(parents=True, exist_ok=True)

                df_actions = self.save_action_memory()
                df_actions.to_csv(results_dir / f"actions_{self.mode}_{self.model_name}_{self.iteration}.csv")
                df_total_value.to_csv(results_dir / f"account_value_{self.mode}_{self.model_name}_{self.iteration}.csv", index=False)
                df_rewards.to_csv(results_dir / f"account_rewards_{self.mode}_{self.model_name}_{self.iteration}.csv", index=False)
                plt.plot(self.asset_memory, "r")
                plt.savefig(results_dir / f"account_value_{self.mode}_{self.model_name}_{self.iteration}.png")
                plt.close()

            # Gymnasium expects 5 return values: obs, reward, terminated, truncated, info
            return self.state, self.reward, self.terminal, False, {} # Return truncated as False when terminal

        else:
            actions = actions * self.hmax  # actions initially is scaled between 0 to 1
            actions = actions.astype(int)

            # Update turbulence before potentially overriding actions
            # Note: self.data already points to the correct slice for the *upcoming* state update (day d+1)
            # If turbulence check needs data from day d, access self.df directly using iloc for day d
            current_day_data_slice = self.df.iloc[self.day * self.stock_dim : (self.day + 1) * self.stock_dim]
            if self.turbulence_threshold is not None:
                if self.risk_indicator_col in current_day_data_slice.columns:
                    try:
                        self.turbulence = current_day_data_slice[self.risk_indicator_col].iloc[0]
                    except IndexError:
                        print(f"Warning: Could not access turbulence data at day {self.day}, index 0. Slice shape: {current_day_data_slice.shape}. Setting turbulence to 0.")
                        self.turbulence = 0
                else:
                    print(f"Warning: Turbulence column '{self.risk_indicator_col}' not found. Setting turbulence to 0.")
                    self.turbulence = 0

                # Apply turbulence action override
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)

            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )

            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                actions[index] = self._sell_stock(index, actions[index]) * (-1)

            for index in buy_index:
                actions[index] = self._buy_stock(index, actions[index])
            
            self.actions_memory.append(actions)

            # state: s -> s+1
            self.day += 1
            # Access data using iloc based on day and stock_dim for the *next* day
            self.data = self.df.iloc[self.day * self.stock_dim : (self.day + 1) * self.stock_dim]

            # Update state using the new self.data slice
            self.state = self._update_state()

            # Calculate end total asset *after* state update
            # state[1:stock_dim+1] = prices, state[stock_dim+1:2*stock_dim+1] = holdings
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling
            self.state_memory.append(self.state)

            # Gymnasium expects 5 return values: obs, reward, terminated, truncated, info
            return self.state, self.reward, self.terminal, False, {} # Return truncated as False

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        # Gymnasium reset signature requires handling seed and options
        super().reset(seed=seed) # Important for seeding the action space sampler etc.

        # Reset internal state
        self.day = 0
        self.data = self.df.iloc[self.day * self.stock_dim : (self.day + 1) * self.stock_dim]
        self.state = self._initiate_state()

        # Reset memory
        if self.initial:
            self.asset_memory = [
                self.initial_amount
                + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state[1 : 1 + self.stock_dim])
                )
            ]
        else:
            # This logic might be less relevant now if each episode starts fresh
            # Keep it for now, but ensure previous_state handling is correct if used.
            if self.previous_state and len(self.previous_state) > 0:
                 previous_total_asset = self.previous_state[0] + sum(
                     np.array(self.state[1 : (self.stock_dim + 1)])
                     * np.array(
                         self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                     )
                 )
                 self.asset_memory = [previous_total_asset]
            else: # Fallback if previous_state is invalid
                 self.asset_memory = [self.initial_amount]

        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.episode += 1

        # Gymnasium reset returns obs, info
        return self.state, {}

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        # Ensures data slice is valid before accessing
        if self.data.empty:
             print(f"Warning: Data slice empty during _initiate_state at day {self.day}. Returning zeros.")
             # Return a state of appropriate shape filled with zeros or handle error
             # Shape is 1 (balance) + stock_dim (prices) + stock_dim (shares) + N indicators * stock_dim
             state_len = 1 + 2 * self.stock_dim + len(self.tech_indicator_list) * self.stock_dim
             # Return as numpy array of correct shape/dtype even if zeros
             return np.zeros(state_len, dtype=np.float32)

        # Check if self.df has 'tic' column for multi-stock check
        # Use self.data which is the relevant slice for the current day
        is_multi_stock = 'tic' in self.data.columns and len(self.data['tic'].unique()) > 1

        if self.initial:
            # For Initial State
            if is_multi_stock:
                state = (
                    [self.initial_amount] # Cash
                    + self.data.close.values.tolist() # Prices
                    + (self.num_stock_shares if isinstance(self.num_stock_shares, list) else []) # Shares (ensure list, default empty if invalid)
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list if tech in self.data
                        ),
                        [],
                    ) # Indicators
                )
            else: # Single stock case (or fallback if 'tic' column missing)
                state = (
                    [self.initial_amount] # Cash
                    + [self.data.close.iloc[0] if not self.data.close.empty else 0] # Price
                    + (self.num_stock_shares if isinstance(self.num_stock_shares, list) else []) # Shares (ensure list, default empty if invalid)
                    + sum(([self.data[tech].iloc[0] if tech in self.data and not self.data[tech].empty else 0] for tech in self.tech_indicator_list), []) # Indicators
                )
        else:
            # Using Previous State - Ensure previous_state is valid
            if not self.previous_state or len(self.previous_state) < 1 + 2 * self.stock_dim:
                 print("Warning: Invalid previous_state provided to _initiate_state. Re-initializing.")
                 # Fallback to initial state logic
                 self.initial = True
                 return self._initiate_state() # This will call _initiate_state again with initial=True

            if is_multi_stock:
                state = (
                    [self.previous_state[0]] # Cash
                    + self.data.close.values.tolist() # Current prices
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1) # Previous shares
                    ]
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list if tech in self.data
                        ),
                        [],
                    ) # Current indicators
                )
            else: # Single stock case
                state = (
                    [self.previous_state[0]] # Cash
                    + [self.data.close.iloc[0] if not self.data.close.empty else 0] # Current price
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1) # Previous shares (list of 1 value)
                    ]
                    + sum(([self.data[tech].iloc[0] if tech in self.data and not self.data[tech].empty else 0] for tech in self.tech_indicator_list), []) # Current indicators
                )

        # Removed state length validation/padding logic
        # Let potential length errors surface during array conversion or later checks
        try:
            np_state = np.array(state, dtype=np.float32)
            expected_len = 1 + 2 * self.stock_dim + len(self.tech_indicator_list) * self.stock_dim
            if np_state.shape[0] != expected_len:
                print(f"ERROR: State length mismatch in _initiate_state AFTER construction. Expected {expected_len}, Got {np_state.shape[0]}. This indicates indicators might be missing from self.data or tech_indicator_list.")
                # Optional: Raise an error here to stop execution immediately
                # raise ValueError(f"State length mismatch: Expected {expected_len}, Got {np_state.shape[0]}")
            return np_state
        except ValueError as e:
            print(f"ERROR creating numpy state array in _initiate_state: {e}")
            print(f"Original state list length: {len(state)}")
            # Fallback: Return zeros if array creation fails
            state_len = 1 + 2 * self.stock_dim + len(self.tech_indicator_list) * self.stock_dim
            return np.zeros(state_len, dtype=np.float32)


    def _update_state(self):
        # Ensures data slice is valid before accessing
        if self.data.empty:
             print(f"Warning: Data slice empty during _update_state at day {self.day}. Returning previous state.")
             # Ensure previous state is returned as numpy array
             return np.array(self.state, dtype=np.float32) if not isinstance(self.state, np.ndarray) else self.state

        is_multi_stock = 'tic' in self.data.columns and len(self.data['tic'].unique()) > 1

        if is_multi_stock:
            state = (
                [self.state[0]] # Cash
                + self.data.close.values.tolist() # Current prices
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]) # Current shares
                + sum(
                    (
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list if tech in self.data
                    ),
                    [],
                ) # Current indicators
            )
        else: # Single stock case
            state = (
                [self.state[0]] # Cash
                + [self.data.close.iloc[0] if not self.data.close.empty else 0] # Current price
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]) # Current shares (list of 1 value)
                + sum(([self.data[tech].iloc[0] if tech in self.data and not self.data[tech].empty else 0] for tech in self.tech_indicator_list), []) # Current indicators
            )

        # Removed state length validation/padding logic
        try:
            np_state = np.array(state, dtype=np.float32)
            expected_len = 1 + 2 * self.stock_dim + len(self.tech_indicator_list) * self.stock_dim
            if np_state.shape[0] != expected_len:
                print(f"ERROR: State length mismatch in _update_state AFTER construction. Expected {expected_len}, Got {np_state.shape[0]}.")
                # Optional: Raise error
                # raise ValueError(f"State length mismatch: Expected {expected_len}, Got {np_state.shape[0]}")
            return np_state
        except ValueError as e:
            print(f"ERROR creating numpy state array in _update_state: {e}")
            print(f"Original state list length: {len(state)}")
            # Fallback: Return previous state if array creation fails
            return np.array(self.state, dtype=np.float32) if not isinstance(self.state, np.ndarray) else self.state


    def _get_date(self):
        # Access the index (which should be DatetimeIndex now) of the current data slice
        # Since all rows in the slice share the same date, get the first one
        if not self.data.empty and isinstance(self.data.index, pd.DatetimeIndex):
             # Handle MultiIndex case where 'date' might be the first level
             if isinstance(self.data.index, pd.MultiIndex):
                 date = self.data.index.get_level_values(0)[0]
             else:
                 date = self.data.index[0]
        else:
             # Handle case where data slice might be empty or index is not datetime
             date = None # Or handle appropriately
        return date

    # add save_state_memory to preserve state in the trading process
    def save_state_memory(self):
        # Requires 'tic' column for differentiation if multi-stock
        is_multi_stock = 'tic' in self.df.columns and len(self.df['tic'].unique()) > 1

        date_list = self.date_memory[:-1] # Dates corresponding to states recorded
        state_list = self.state_memory

        if not state_list:
             return pd.DataFrame() # Return empty if no states were recorded

        if is_multi_stock:
            # Define columns dynamically based on stock_dim and indicators
            price_cols = [f'price_{i}' for i in range(self.stock_dim)]
            share_cols = [f'shares_{i}' for i in range(self.stock_dim)]
            indicator_cols = [f'{tech}_{i}' for i in range(self.stock_dim) for tech in self.tech_indicator_list]
            all_cols = ['cash'] + price_cols + share_cols + indicator_cols

            # Ensure state_list elements have the correct length, pad/truncate if necessary
            expected_len = 1 + 2 * self.stock_dim + len(self.tech_indicator_list) * self.stock_dim
            processed_state_list = []
            for state in state_list:
                 # Ensure state is a list or numpy array before checking len
                 current_state = state.tolist() if isinstance(state, np.ndarray) else list(state)
                 if len(current_state) != expected_len:
                      print(f"Warning: State length mismatch in save_state_memory. Expected {expected_len}, Got {len(current_state)}. Adjusting...")
                      current_state = (current_state + [0.0] * expected_len)[:expected_len]
                 processed_state_list.append(current_state)

            df_states = pd.DataFrame(processed_state_list, columns=all_cols)
            if len(date_list) == len(df_states):
                 df_states['date'] = date_list
                 # Ensure date is the index name
                 df_states = df_states.set_index(pd.Index(date_list, name='date'))
            else:
                 print("Warning: Mismatch between date_list and state_list lengths in save_state_memory.")

        else: # Single stock case
            all_cols = ['cash', 'price', 'shares'] + self.tech_indicator_list
            expected_len = 1 + 1 + 1 + len(self.tech_indicator_list) # Adjust based on actual single stock state structure
            processed_state_list = []
            for state in state_list:
                 current_state = state.tolist() if isinstance(state, np.ndarray) else list(state)
                 if len(current_state) != expected_len:
                      print(f"Warning: State length mismatch (single stock) in save_state_memory. Expected {expected_len}, Got {len(current_state)}. Adjusting...")
                      current_state = (current_state + [0.0] * expected_len)[:expected_len]
                 processed_state_list.append(current_state)

            df_states = pd.DataFrame(processed_state_list, columns=all_cols)
            if len(date_list) == len(df_states):
                 df_states['date'] = date_list
                 df_states = df_states.set_index(pd.Index(date_list, name='date'))
            else:
                 print("Warning: Mismatch between date_list and state_list lengths (single stock) in save_state_memory.")

        return df_states

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        if len(date_list) != len(asset_list):
            print(f"Warning: Mismatch in length of date_memory ({len(date_list)}) and asset_memory ({len(asset_list)})")
            # Adjust to minimum length
            min_len = min(len(date_list), len(asset_list))
            date_list = date_list[:min_len]
            asset_list = asset_list[:min_len]

        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        date_list = self.date_memory[:-1] # Actions correspond to steps taken
        action_list = self.actions_memory

        if len(date_list) != len(action_list):
            print(f"Warning: Mismatch in length of date_memory[:-1] ({len(date_list)}) and actions_memory ({len(action_list)})")
            min_len = min(len(date_list), len(action_list))
            date_list = date_list[:min_len]
            action_list = action_list[:min_len]

        df_actions = pd.DataFrame(action_list)

        # Try to get ticker names for columns if possible
        try:
             # Use tickers from the initial data slice if available
             initial_data = self.df.iloc[0 * self.stock_dim : (0 + 1) * self.stock_dim]
             if 'tic' in initial_data.columns:
                  # Ensure number of actions matches stock_dim
                  if df_actions.shape[1] == self.stock_dim:
                       df_actions.columns = initial_data['tic'].values
                  else:
                       df_actions.columns = [f'action_{i}' for i in range(df_actions.shape[1])]
             else: # Fallback column names
                  df_actions.columns = [f'action_{i}' for i in range(df_actions.shape[1])]
        except Exception:
             # Fallback if error accessing initial data or 'tic'
             df_actions.columns = [f'action_{i}' for i in range(df_actions.shape[1])]

        df_actions['date'] = date_list
        df_actions = df_actions.set_index(pd.Index(date_list, name='date'))
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
