"""
binance_paper_trading_budget_logging.py
Self-contained Binance *test-net* paper-trading bot driven by a PPO policy.
Supports fixed-budget BUYs, ATR-based TP/SL, and deep diagnostic logging.

Version 2.0  –  2025-06-29
──────────────────────────────────────────────────────────────────────────────
Usage
-----
from my_meta.binance_paper_trading_budget_logging import (
    BinancePaperTradingCryptoLive,
)

bot = BinancePaperTradingCryptoLive(
    ticker_list        = ["SOLUSDT"],
    time_interval      = "1min",
    drl_lib            = "elegantrl",
    agent              = "ppo",
    cwd                = "api/papertrading_crypto",
    net_dim            = [256,128,64,32],
    state_dim          = 1 + 4 + 1 + 1,      # 4 tech indicators
    action_dim         = 1,
    API_KEY            = YOUR_TESTNET_KEY,
    API_SECRET         = YOUR_TESTNET_SECRET,
    tech_indicator_list= ["macd","rsi","cci","dx"],
    trade_budget       = 5_000,              # ← spend max 5 000 USDT per BUY
    debug              = True,               # ← show DEBUG lines
)
bot.run()
"""

from __future__ import annotations

# ────────────────────────────── stdlib ───────────────────────────────────────
import json
import os
import time
import datetime as dt
import logging
import warnings
from typing import Optional, List, Dict, Tuple

warnings.filterwarnings(
    "ignore",
    message="There is no current event loop",
    category=DeprecationWarning,
    module=r"binance\.helpers",
)

# ──────────────────────── third-party dependencies ───────────────────────────
import numpy as np
import pandas as pd
import torch
import stockstats
import talib
from binance.client import Client
# flexible error import (binance-connector vs python-binance)
try:
    from binance.error import ClientError          # binance-connector ≥1.5
except ImportError:
    try:
        from binance.exceptions import BinanceAPIException as ClientError  # python-binance
    except ImportError:
        ClientError = Exception                                            # fallback

from my_meta.single_asset_training_ppo import AgentPPO

# ════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ════════════════════════════════════════════════════════════════════════════
def convert_time_interval_to_timedelta(time_interval: str) -> dt.timedelta:
    """Convert '1Day', '5Min', '30s' → `datetime.timedelta`."""
    ti = time_interval.lower()
    if ti.endswith("day"):
        return dt.timedelta(days=int(ti[:-3]))
    if ti.endswith("hour"):
        return dt.timedelta(hours=int(ti[:-4]))
    if ti.endswith("min"):
        return dt.timedelta(minutes=int(ti[:-3]))
    if ti.endswith("s"):
        return dt.timedelta(seconds=int(ti[:-1]))
    raise ValueError(f"Unsupported time_interval: {time_interval}")


def _calculate_indicators_generic(
    df: pd.DataFrame,
    tech_indicator_list: List[str],
    select_stockstats_talib: int = 0,
) -> pd.DataFrame:
    """Compute technical indicators with StockStats (0) or TA-Lib (1)."""
    df.sort_values(["time", "tic"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    if select_stockstats_talib == 0:
        sdf = stockstats.StockDataFrame.retype(df.copy())
        for ind in tech_indicator_list:
            tmp = pd.DataFrame()
            for tic in sdf["tic"].unique():
                sub = sdf[sdf["tic"] == tic]
                if ind not in sub.columns:
                    continue
                tmp = pd.concat(
                    [tmp,
                     pd.DataFrame({"time": sub["time"], "tic": tic, ind: sub[ind]})]
                )
            if not tmp.empty:
                df = df.merge(tmp, on=["time", "tic"], how="left")
    else:
        out = pd.DataFrame()
        for tic in df["tic"].unique():
            tdf = df[df["tic"] == tic].copy()
            if any(i in tech_indicator_list for i in ["macd", "macd_signal", "macd_hist"]):
                macd, macds, macdh = talib.MACD(tdf["close"], 12, 26, 9)
                tdf["macd"], tdf["macd_signal"], tdf["macd_hist"] = macd, macds, macdh
            if "rsi" in tech_indicator_list:
                tdf["rsi"] = talib.RSI(tdf["close"], 14)
            if "cci" in tech_indicator_list:
                tdf["cci"] = talib.CCI(tdf["high"], tdf["low"], tdf["close"], 14)
            if "dx" in tech_indicator_list:
                tdf["dx"] = talib.DX(tdf["high"], tdf["low"], tdf["close"], 14)
            out = pd.concat([out, tdf])
        df = out

    df.dropna(inplace=True)
    df.sort_values(["time", "tic"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def fetch_latest_data_binance(
    API_KEY: str,
    API_SECRET: str,
    ticker_list: List[str],
    time_interval: str,
    tech_indicator_list: List[str],
    start_date: Optional[dt.datetime] = None,
    end_date: Optional[dt.datetime] = None,
    select_stockstats_talib: int = 0,
    atr_window: int = 14,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Pull klines + indicators from Binance *test-net* spot."""
    client = Client(API_KEY, API_SECRET, testnet=True)
    client.API_URL = "https://testnet.binance.vision/api"

    ivl_map = {
        "1min": Client.KLINE_INTERVAL_1MINUTE,
        "3min": Client.KLINE_INTERVAL_3MINUTE,
        "5min": Client.KLINE_INTERVAL_5MINUTE,
        "15min": Client.KLINE_INTERVAL_15MINUTE,
        "30min": Client.KLINE_INTERVAL_30MINUTE,
        "1hour": Client.KLINE_INTERVAL_1HOUR,
        "4hour": Client.KLINE_INTERVAL_4HOUR,
        "1day": Client.KLINE_INTERVAL_1DAY,
    }
    ivl_key = time_interval.lower()
    if ivl_key not in ivl_map:
        raise ValueError(f"Unsupported interval {time_interval}")

    bar_delta = convert_time_interval_to_timedelta(time_interval)
    bars = 60 + atr_window
    end_date = end_date or dt.datetime.utcnow()
    start_date = start_date or end_date - bars * bar_delta
    start_ms, end_ms = int(start_date.timestamp()*1000), int(end_date.timestamp()*1000)

    frames: List[pd.DataFrame] = []
    for sym in ticker_list:
        raw = client.get_klines(symbol=sym,
                        interval=ivl_map[ivl_key],
                        startTime=start_ms, endTime=end_ms)

        if not raw:
            continue
        df = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote", "ignore",
        ])
        df["time"] = pd.to_datetime(df["close_time"], unit="ms")
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        df["tic"] = sym
        frames.append(df[["time", "tic", "open", "high", "low", "close", "volume"]])

    if not frames:
        return tuple([np.array([])]*5 + [pd.DataFrame()])

    all_df = pd.concat(frames)
    all_df = _calculate_indicators_generic(all_df, tech_indicator_list, select_stockstats_talib)

    latest = all_df.groupby("tic", as_index=False).tail(1).copy()
    latest["sort"] = latest["tic"].apply(lambda x: ticker_list.index(x))
    latest.sort_values("sort", inplace=True)

    latest_price = latest["close"].astype(np.float32).values
    latest_high  = latest["high"].astype(np.float32).values
    latest_low   = latest["low"].astype(np.float32).values
    latest_close = latest["close"].astype(np.float32).values

    tech_arr = np.stack([
        latest[ind].astype(np.float32).values for ind in tech_indicator_list
    ], axis=1) if tech_indicator_list else np.empty((len(latest), 0), dtype=np.float32)

    return latest_price, tech_arr, latest_high, latest_low, latest_close, all_df

# ════════════════════════════════════════════════════════════════════════════
# Trading class
# ════════════════════════════════════════════════════════════════════════════
class BinancePaperTradingCryptoLive:
    """Binance test-net paper-trading bot with PPO, fixed-budget orders, ATR TP/SL."""

    # ─────────────────────────── initialiser ────────────────────────────────
    def __init__(
        self,
        ticker_list: List[str],
        time_interval: str,
        drl_lib: str,
        agent: str,
        cwd: str,
        net_dim: List[int],
        state_dim: int,
        action_dim: int,
        API_KEY: str,
        API_SECRET: str,
        tech_indicator_list: List[str],
        *,
        trade_budget: Optional[float] = None,
        tp_multiplier: float = 2.0,
        sl_multiplier: float = 1.0,
        atr_window: int = 14,
        max_trade_duration: int = 20,
        initial_capital: float = 10_000,
        log_file_path: Optional[str] = None,
        actor_filename: str = "best_actor.pth",
        debug: bool = False,
    ) -> None:
        # ── logging ─────────────────────────────────────────────────────────
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        h = logging.FileHandler(log_file_path, mode="w") if log_file_path else logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(h)

        if not API_KEY or not API_SECRET:
            self.logger.warning("API_KEY / API_SECRET not provided – private endpoints will fail!")

        # ── DRL policy ──────────────────────────────────────────────────────
        if agent.lower() != "ppo" or drl_lib.lower() != "elegantrl":
            raise ValueError("Only PPO from ElegantRL is supported currently.")
        tmp_agent = AgentPPO(net_dim, state_dim, action_dim)
        tmp_agent.act.load_state_dict(torch.load(f"{cwd}/{actor_filename}", map_location="cpu"))
        self.act = tmp_agent.act
        self.device = tmp_agent.device

        # ── Binance client ──────────────────────────────────────────────────
        self.binance = Client(API_KEY, API_SECRET, testnet=True)
        self.binance.API_URL = "https://testnet.binance.vision/api"
        self.logger.info("Connected to Binance testnet ✔︎")

        # ── params & portfolio ─────────────────────────────────────────────
        self.ticker_list = ticker_list
        self.time_interval = time_interval
        self.tech_indicator_list = tech_indicator_list
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier
        self.atr_window = atr_window
        self.max_trade_duration = max_trade_duration
        self.trade_budget = trade_budget

        self.cash = float(initial_capital)
        self.stocks = np.zeros(len(ticker_list), dtype=float)
        self.price = np.zeros(len(ticker_list), dtype=float)

        # current trade state
        self.in_position = False
        self.entry_price: Optional[float] = None
        self.target_price: Optional[float] = None
        self.stop_loss_price: Optional[float] = None
        self.trade_entry_step: Optional[int] = None
        self.current_step = 0
        self.stop_trading = False
        self.current_atr = np.nan
        self.last_order_error: Optional[str] = None

        # logs
        self.trade_logs: List[Dict] = []
        self.step_logs: List[Dict] = []
        self.log_file_path = log_file_path or "binance_paper_trading.log"

    # ─────────────────────────── filter helper ─────────────────────────────
    @staticmethod
    def _get_filter(filters: list[dict], *types: str) -> dict | None:
        """Return the first filter whose type matches one of *types*."""
        return next((f for f in filters if f["filterType"] in types), None)

    # ─────────────────────── balance / position helpers ─────────────────────
    def _update_balances(self) -> None:
        try:
            acct = self.binance.get_account()
            usdt = next(b for b in acct["balances"] if b["asset"] == "USDT")
            self.cash = float(usdt["free"])
        except Exception as e:
            self.logger.error(f"Balance fetch error: {e}")

    def _fetch_positions(self) -> np.ndarray:
        pos_arr = np.zeros(len(self.ticker_list), dtype=float)
        try:
            bal_map = {
                b["asset"]: float(b["free"]) + float(b["locked"])
                for b in self.binance.get_account()["balances"]
            }
            for idx, sym in enumerate(self.ticker_list):
                base = sym[:-4]  # e.g. BTC from BTCUSDT
                pos_arr[idx] = bal_map.get(base, 0.0)
        except Exception as e:
            self.logger.error(f"Position fetch error: {e}")
        return pos_arr

    # ───────────────────────────── main loop ───────────────────────────────
    def run(self) -> None:
        self.logger.info("Starting live loop …")
        while not self.stop_trading:
            self.trade()
            # sleep according to bar size
            if self.time_interval.endswith("min"):
                time.sleep(int(self.time_interval.replace("min", "")) * 60)
            elif self.time_interval.endswith("s"):
                time.sleep(int(self.time_interval.replace("s", "")))
            else:
                time.sleep(60)

    def stop(self) -> None:
        self.stop_trading = True
        self.save_logs_to_csv()

    # ─────────────────────────── trading logic ─────────────────────────────
    def trade(self) -> None:
        # snapshot balances & positions
        self._update_balances()
        base_asset = self.ticker_list[0][:-4]
        base_qty   = self._fetch_positions()[0]
        self.logger.debug("BEGIN step=%d cash=%.2f %s=%.6f", self.current_step, self.cash, base_asset, base_qty)

        # build RL state & get action
        state = self.get_state()
        with torch.no_grad():
            raw_action = self.act(torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0))
        act_val = float(raw_action.cpu().numpy()[0][0])

        executed_qty = 0.0
        order_ok = False

        # ── ENTRY ──────────────────────────────────────────────────────────
        if not self.in_position and act_val > 0:
            price, _, _, _, _, hist_df = fetch_latest_data_binance(
                API_KEY="", API_SECRET="", ticker_list=self.ticker_list,
                time_interval=self.time_interval, tech_indicator_list=self.tech_indicator_list,
                atr_window=self.atr_window, select_stockstats_talib=1,
            )
            if price.size > 0 and len(hist_df) >= self.atr_window:
                self.entry_price = float(price[0])
                atr_series = talib.ATR(hist_df["high"], hist_df["low"], hist_df["close"], timeperiod=self.atr_window)
                self.current_atr = float(atr_series.iloc[-1])
                self.target_price = self.entry_price + self.tp_multiplier * self.current_atr
                self.stop_loss_price = max(self.entry_price - self.sl_multiplier * self.current_atr, 0.01)

                budget = min(self.trade_budget or self.cash, self.cash)
                executed_qty = budget / self.entry_price
                resp: List[bool] = []
                self.submitOrder(budget, self.ticker_list[0], "buy", resp)
                order_ok = resp and resp[0]
                if order_ok:
                    self.stocks[0] = executed_qty
                    self.in_position = True
                    self.trade_entry_step = self.current_step
                    self.logger.info(
                        "BUY %.6f %s @ %.3f | TP=%.3f SL=%.3f (ATR=%.4f)",
                        executed_qty, self.ticker_list[0], self.entry_price,
                        self.target_price, self.stop_loss_price, self.current_atr,
                    )
        # ── EXIT / MANAGE ──────────────────────────────────────────────────
        elif self.in_position:
            current_price = self.price[0]
            if (current_price >= self.target_price or
                current_price <= self.stop_loss_price or
                (self.current_step - self.trade_entry_step) >= self.max_trade_duration):
                resp: List[bool] = []
                self.submitOrder(self.stocks[0], self.ticker_list[0], "sell", resp)
                order_ok = resp and resp[0]
                if order_ok:
                    executed_qty = self.stocks[0]
                    result = ("TP" if current_price >= self.target_price else
                              ("SL" if current_price <= self.stop_loss_price else "Timeout"))
                    self.in_position = False
                    self.stocks[0] = 0
                    self.logger.info(
                        "SELL %.6f %s @ %.3f – %s",
                        executed_qty, self.ticker_list[0], current_price, result,
                    )

        # update latest price & equity
        self.price = fetch_latest_data_binance(
            API_KEY="", API_SECRET="", ticker_list=self.ticker_list,
            time_interval=self.time_interval, tech_indicator_list=self.tech_indicator_list,
            atr_window=self.atr_window, select_stockstats_talib=1,
        )[0]
        equity = self.cash + self.price[0] * self.stocks[0]

        # step summary
        self.logger.info(
            "STEP %04d | a=%+.4f | P=%.3f | cash=%.2f | eq=%.2f | budget=%.2f | ok=%s | err=%s",
            self.current_step, act_val, self.price[0], self.cash, equity,
            self.trade_budget or self.cash, order_ok, self.last_order_error,
        )

        self.current_step += 1

    # ───────────────────── state builder for PPO ───────────────────────────
    def get_state(self) -> np.ndarray:
        price, tech, *_ = fetch_latest_data_binance(
            API_KEY="", API_SECRET="", ticker_list=self.ticker_list,
            time_interval=self.time_interval, tech_indicator_list=self.tech_indicator_list,
            atr_window=self.atr_window, select_stockstats_talib=1,
        )
        if price.size == 0:
            dim = 1 + tech.size + 2
            return np.zeros(dim, dtype=np.float32)

        self.price = price
        self.stocks = self._fetch_positions()
        self._update_balances()

        scaled_price = 1.0                           # already 1 (placeholder)
        last_pct_change = 0.0                        # could compute if needed
        in_trade_flag = 1.0 if self.in_position else 0.0

        return np.hstack([
            np.array([scaled_price], dtype=np.float32),
            tech.flatten(),
            np.array([last_pct_change, in_trade_flag], dtype=np.float32),
        ]).astype(np.float32)

    # ────────────────────────── order wrapper ──────────────────────────────
    def _get_filter(filters: list[dict], *types: str) -> dict | None:
        """Return the first filter whose type matches one of *types*."""
        return next((f for f in filters if f["filterType"] in types), None)

    def submitOrder(self, qty_or_quote: float, symbol: str, side: str, resp: list[bool]):
        """Market order that honours LOT_SIZE & (MIN_)NOTIONAL, with full logging."""
        self.last_order_error = None
        try:
            # ---------- live balances ----------
            self._update_balances()                  # refresh self.cash
            self.logger.debug("CASH free USDT = %.2f", self.cash)

            info       = self.binance.get_symbol_info(symbol)
            lot        = BinancePaperTradingCryptoLive._get_filter(info["filters"], "LOT_SIZE")
            notional_f = BinancePaperTradingCryptoLive._get_filter(info["filters"], "MIN_NOTIONAL", "NOTIONAL")

            if lot is None or notional_f is None:
                raise RuntimeError(
                    f"Symbol {symbol} missing required filters "
                    f"(LOT_SIZE={bool(lot)}, NOTIONAL={bool(notional_f)})"
                )

            step        = float(lot["stepSize"])
            min_qty     = float(lot["minQty"])
            min_notional= float(notional_f["minNotional"])

            # ---------- decide order size ----------
            price = self.price[0]                    # last trade loop update
            if side.lower() == "buy":
                quote = self.trade_budget or self.cash    # spend *all* cash if no budget
                quote = min(quote, self.cash)             # can’t spend more than you have
                if quote < min_notional:
                    self.logger.warning(
                        "BUY aborted – quote %.2f < minNotional %.2f", quote, min_notional
                    )
                    resp.append(False); return
                self.logger.info(
                    "BUY %.2f USDT @ %.3f  (minNotional %.2f)",
                    quote, price, min_notional
                )
                order = self.binance.create_order(
                    symbol=symbol,
                    side="BUY",
                    type="MARKET",
                    quoteOrderQty=round(quote, 2)
                )
            else:  # SELL
                qty = max(qty_or_quote, min_qty)
                # round *down* to the nearest stepSize
                qty = (qty // step) * step
                if qty * price < min_notional:
                    self.logger.warning(
                        "SELL aborted – notional %.2f < minNotional %.2f",
                        qty * price, min_notional
                    )
                    resp.append(False); return
                self.logger.info(
                    "SELL %.6f %s @ %.3f  (minNotional %.2f)",
                    qty, symbol, price, min_notional
                )
                order = self.binance.create_order(
                    symbol=symbol,
                    side="SELL",
                    type="MARKET",
                    quantity=round(qty, 8)
                )

            # ---------- success ----------
            self.logger.debug("ORDER RESPONSE:\n%s", json.dumps(order, indent=2))
            resp.append(True)

        except ClientError as ce:
            self.last_order_error = f"{getattr(ce, 'error_code', '?')} {getattr(ce, 'error_message', str(ce))}"
            self.logger.exception("Order failed (Binance)")
            resp.append(False)
        except Exception:
            import traceback
            self.last_order_error = traceback.format_exc(limit=1)
            self.logger.exception("Order failed (unexpected)")
            resp.append(False)

    # ───────────────────────────── CSV dump ────────────────────────────────
    def save_logs_to_csv(self):
        import csv
        base = os.path.splitext(self.log_file_path)[0]
        trade_path = f"{base}_trade_logs.csv"
        step_path  = f"{base}_step_logs.csv"
        if self.trade_logs:
            with open(trade_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.trade_logs[0].keys())
                w.writeheader(); w.writerows(self.trade_logs)
        if self.step_logs:
            with open(step_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.step_logs[0].keys())
                w.writeheader(); w.writerows(self.step_logs)
        self.logger.info("Logs dumped → %s & %s", trade_path, step_path)
