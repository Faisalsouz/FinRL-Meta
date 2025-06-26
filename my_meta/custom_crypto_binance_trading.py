import time
import threading
import datetime as dt
import numpy as np
import pandas as pd
import torch
import gym
import stockstats
import talib
import logging
from binance.client import Client
from my_meta.single_asset_training_ppo import AgentPPO  # Adjust the import as needed

def convert_time_interval_to_timedelta(time_interval: str) -> dt.timedelta:
    """Convert a time_interval string (e.g., '1Day', '5Min', '15Min', '30s') into a timedelta."""
    time_interval = time_interval.lower()
    if time_interval.endswith('day'):
        value = int(time_interval.replace('day', ''))
        return dt.timedelta(days=value)
    elif time_interval.endswith('hour'):
        value = int(time_interval.replace('hour', ''))
        return dt.timedelta(hours=value)
    elif time_interval.endswith('min'):
        value = int(time_interval.replace('min', ''))
        return dt.timedelta(minutes=value)
    elif time_interval.endswith('s'):
        value = int(time_interval.replace('s', ''))
        return dt.timedelta(seconds=value)
    else:
        raise ValueError(f"Unsupported time_interval: {time_interval}")

def _calculate_indicators_generic(
    df: pd.DataFrame,
    data_source: str,
    tech_indicator_list: list,
    select_stockstats_talib=0
) -> pd.DataFrame:
    """
    Helper function to compute technical indicators using either stockstats or TA‑Lib.
    Expects columns: ['time','tic','open','high','low','close','volume'].
    """
    df.sort_values(by=['time','tic'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    if select_stockstats_talib == 0:
        # Stockstats approach
        stock = stockstats.StockDataFrame.retype(df.copy(deep=True))
        unique_tic = stock['tic'].unique()
        for indicator in tech_indicator_list:
            indicator_df = pd.DataFrame()
            for tic in unique_tic:
                sub_df = stock[stock['tic'] == tic]
                if indicator not in sub_df.columns:
                    continue
                temp_vals = sub_df[indicator]
                tmp2 = pd.DataFrame({
                    'time': sub_df['time'],
                    'tic': tic,
                    indicator: temp_vals
                })
                indicator_df = pd.concat([indicator_df, tmp2], ignore_index=True)
            if not indicator_df.empty:
                df = df.merge(indicator_df, on=['time','tic'], how='left')
    else:
        # TA‑Lib approach
        final_df = pd.DataFrame()
        for tic_val in df['tic'].unique():
            tic_df = df[df['tic'] == tic_val].copy()
            if 'macd' in tech_indicator_list or 'macd_signal' in tech_indicator_list or 'macd_hist' in tech_indicator_list:
                macd, macd_signal, macd_hist = talib.MACD(
                    tic_df['close'], fastperiod=12, slowperiod=26, signalperiod=9
                )
                tic_df['macd'] = macd
                tic_df['macd_signal'] = macd_signal
                tic_df['macd_hist'] = macd_hist
            if 'rsi' in tech_indicator_list:
                tic_df['rsi'] = talib.RSI(tic_df["close"], timeperiod=14)
            if 'cci' in tech_indicator_list:
                tic_df['cci'] = talib.CCI(
                    tic_df["high"], tic_df["low"], tic_df["close"], timeperiod=14
                )
            if 'dx' in tech_indicator_list:
                tic_df['dx'] = talib.DX(
                    tic_df["high"], tic_df["low"], tic_df["close"], timeperiod=14
                )
            final_df = pd.concat([final_df, tic_df], ignore_index=True)
        df = final_df

    df.sort_values(by=['time','tic'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.dropna(axis=0, how='any').reset_index(drop=True)
    return df
def submitOrder(self, qty, symbol, side, resp):
    """Place a MARKET order via Binance testnet."""
    if qty > 0:
        try:
            order = self.binance.order_market(
                symbol=symbol,
                side=side.upper(),
                quantity=qty
            )
            self.logger.info(f"{side.upper()} {qty} {symbol} completed.")
            resp.append(True)
        except Exception as e:
            self.logger.error(f"{side.upper()} {qty} {symbol} failed. Error: {e}")
            resp.append(False)
    else:
        self.logger.info(f"Quantity=0, skipping order for {symbol} ({side}).")
        resp.append(True)
