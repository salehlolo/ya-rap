#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simplified OKX demo trading bot implementing a grid-like strategy."""

import argparse
import json
import os
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

try:  # Optional environment loader
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - best-effort optional dependency
    pass

import ccxt  # type: ignore
import requests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(val)
    except Exception:
        return default


def _to_ns(d: Any) -> Any:
    return SimpleNamespace(**d) if isinstance(d, dict) else d


def _ensure_symbols(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [value]
    return ["BTC/USDT:USDT"]


# ---------------------------------------------------------------------------
# Telegram notifier
# ---------------------------------------------------------------------------


class Notifier:
    def __init__(self, cfg: SimpleNamespace):
        self.enabled = bool(
            getattr(cfg, "telegram_enabled", False)
            and getattr(cfg, "telegram_token", None)
            and getattr(cfg, "telegram_chat_id", None)
        )
        self.base = (
            f"https://api.telegram.org/bot{cfg.telegram_token}" if self.enabled else None
        )
        self.chat_id = getattr(cfg, "telegram_chat_id", None)

    def send(self, text: str) -> None:
        if not self.enabled:
            print(text)
            return
        try:
            resp = requests.post(
                f"{self.base}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
            if resp.status_code != 200:
                print("[WARN] Telegram send failed:", resp.text)
        except Exception as exc:
            print("[WARN] Telegram exception:", exc)


# ---------------------------------------------------------------------------
# OKX exchange wrapper
# ---------------------------------------------------------------------------


class FuturesExchange:
    def __init__(self, cfg: SimpleNamespace):
        key = os.getenv("OKX_API_KEY")
        secret = os.getenv("OKX_SECRET_KEY")
        password = os.getenv("OKX_PASSPHRASE")

        self.x = ccxt.okx(
            {
                "apiKey": key,
                "secret": secret,
                "password": password,
                "enableRateLimit": True,
                "timeout": 15000,
                "options": {"fetchCurrencies": False},
            }
        )

        try:
            self.x.set_sandbox_mode(True)
        except Exception:
            pass

        self.x.headers = self.x.headers or {}
        self.x.headers["x-simulated-trading"] = "1"

        try:
            self.x.load_markets()
        except Exception:
            try:
                if hasattr(self.x, "has") and isinstance(self.x.has, dict):
                    self.x.has["fetchCurrencies"] = False
            except Exception:
                pass
            self.x.load_markets()

        self.is_hedged = False
        try:
            if hasattr(self.x, "set_position_mode"):
                self.x.set_position_mode(False)
        except Exception:
            pass

        self.cfg = cfg

    def ensure_cross_leverage(self, symbol: str, lev: int = 10) -> None:
        try:
            self.x.set_leverage(lev, symbol, {"mgnMode": "cross"})
        except Exception:
            pass

    def get_top_symbols(self, n: int = 10) -> List[str]:
        markets = getattr(self.x, "markets", {}) or {}
        try:
            tickers = self.x.fetch_tickers()
        except Exception:
            tickers = {}

        rows: List[tuple[float, str]] = []
        for sym, market in markets.items():
            try:
                if not market.get("swap"):
                    continue
                if market.get("quote") != "USDT":
                    continue
                if market.get("active") is False:
                    continue

                ticker = tickers.get(sym, {}) or {}
                qv = ticker.get("quoteVolume")
                if qv is None:
                    info = ticker.get("info", {}) if isinstance(ticker, dict) else {}
                    qv = (
                        info.get("volCcy24h")
                        or info.get("volUsd24h")
                        or info.get("vol24h")
                    )
                qv_f = safe_float(qv, 0.0) or 0.0
                rows.append((qv_f, sym))
            except Exception:
                continue

        rows.sort(key=lambda item: item[0], reverse=True)
        limit = max(1, int(n))
        top = [sym for _, sym in rows[:limit]]
        if not top:
            top = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        return top

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 2) -> List[List[float]]:
        return self.x.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def create_market_order(
        self, symbol: str, side: str, amount: float
    ) -> Optional[Dict[str, Any]]:
        params = {"tdMode": "cross"}
        try:
            return self.x.create_order(symbol, "market", side, amount, None, params)
        except Exception as exc:
            print(f"[ERROR] create_order failed for {symbol}: {exc}")
            return None

    def place_exit_algo(
        self,
        symbol: str,
        side: str,
        amount: float,
        tp: float,
        sl: float,
        tp_type: str = "last",
        sl_type: str = "last",
    ) -> Optional[Dict[str, Any]]:
        market = self.x.market(symbol)
        inst_id = market.get("id") if isinstance(market, dict) else None
        method = getattr(self.x, "private_post_trade_order_algo", None) or getattr(
            self.x, "privatePostTradeOrderAlgo", None
        )
        if not inst_id or not method:
            return None

        opposite = "sell" if side == "buy" else "buy"
        try:
            sz = self.x.amount_to_precision(symbol, amount)
        except Exception:
            sz = amount

        payload = {
            "instId": inst_id,
            "tdMode": "cross",
            "side": opposite,
            "ordType": "conditional",
            "sz": str(sz),
            "tpTriggerPx": str(tp),
            "tpOrdPx": "-1",
            "slTriggerPx": str(sl),
            "slOrdPx": "-1",
            "tpTriggerPxType": tp_type,
            "slTriggerPxType": sl_type,
        }

        try:
            resp = method(payload)
            return resp if isinstance(resp, dict) else {"raw": resp}
        except Exception as exc:
            print(f"[WARN] Failed to place TP/SL algo order for {symbol}: {exc}")
            return None

    def has_any_open_position(self) -> bool:
        """Return True when any swap position or open order exists."""
        try:
            positions = self.x.fetch_positions(params={"instType": "SWAP"})
            if isinstance(positions, list):
                for pos in positions:
                    if not isinstance(pos, dict):
                        continue
                    for key in ("contracts", "positionAmt", "size", "amount"):
                        val = pos.get(key)
                        if val in (None, "0", 0):
                            continue
                        try:
                            if abs(float(val)) > 0.0:
                                return True
                        except Exception:
                            continue
                    info = pos.get("info") if isinstance(pos, dict) else {}
                    if isinstance(info, dict):
                        for key in ("pos", "posCcy", "availPos"):
                            val = info.get(key)
                            if val in (None, "0", 0):
                                continue
                            try:
                                if abs(float(val)) > 0.0:
                                    return True
                            except Exception:
                                continue
        except Exception:
            pass

        try:
            opens = self.x.fetch_open_orders()
            return bool(opens)
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Grid-like strategy (translated from Pine Script)
# ---------------------------------------------------------------------------


class GridLikeStrategy:
    def __init__(self, point: float, os_: float, mf: float, anti: bool):
        self.point = float(point)
        self.os = float(os_)
        self.mf = float(mf)
        self.anti = bool(anti)

        self.baseline: Optional[float] = None
        self.size: Optional[float] = None
        self._last_closed_win = False
        self._last_closed_loss = False

    def on_close_result(
        self, *, was_win: bool = False, was_loss: bool = False
    ) -> None:
        self._last_closed_win = bool(was_win)
        self._last_closed_loss = bool(was_loss)

    def update(self, close: float) -> Dict[str, Optional[float]]:
        baseline_old = self.baseline
        if baseline_old is None:
            self.baseline = close
        else:
            if close > baseline_old + self.point or close < baseline_old - self.point:
                self.baseline = close
            else:
                self.baseline = baseline_old

        upper = self.baseline + self.point
        lower = self.baseline - self.point

        prev_size = self.size if self.size is not None else self.os

        if self.anti:
            if self._last_closed_win:
                new_size = prev_size * self.mf
            elif self._last_closed_loss:
                new_size = self.os
            else:
                new_size = prev_size if self.size is not None else self.os
        else:
            if self._last_closed_loss:
                new_size = prev_size * self.mf
            elif self._last_closed_win:
                new_size = self.os
            else:
                new_size = prev_size if self.size is not None else self.os

        self.size = float(new_size)

        signal: Optional[str] = None
        tp: Optional[float] = None
        sl: Optional[float] = None
        if baseline_old is not None:
            if self.baseline > baseline_old:
                signal = "buy"
                tp = upper
                sl = lower
            elif self.baseline < baseline_old:
                signal = "sell"
                tp = lower
                sl = upper

        self._last_closed_win = False
        self._last_closed_loss = False

        return {
            "signal": signal,
            "tp": tp,
            "sl": sl,
            "size": float(self.size),
            "baseline": float(self.baseline),
        }


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------


class Bot:
    def __init__(self, cfg: SimpleNamespace):
        self.cfg = cfg
        self.notifier = Notifier(cfg)
        self.ex = FuturesExchange(cfg)

        self.timeframe = getattr(cfg, "timeframe", "5m")
        self.leverage = int(getattr(cfg, "leverage", 10))
        self.poll_interval = float(getattr(cfg, "poll_interval", 20.0))
        self.single_global = bool(getattr(cfg, "single_global_position", True))

        self.contexts: Dict[str, Dict[str, Any]] = {}
        top_n = getattr(cfg, "top_n", None)
        if top_n is None:
            top_n = getattr(cfg, "top_n_symbols", 10)
        try:
            symbols = self.ex.get_top_symbols(int(top_n))
        except Exception:
            symbols = list(getattr(cfg, "symbols", ["BTC/USDT:USDT"]))
        if not symbols:
            symbols = ["BTC/USDT:USDT"]

        self.symbols = list(symbols)
        cfg.symbols = list(symbols)

        os_map = getattr(cfg, "grid_order_size_map", {}) or {}
        if not isinstance(os_map, dict):
            os_map = {}
        for symbol in self.symbols:
            base_os = os_map.get(symbol, cfg.grid_order_size)
            try:
                base_os = float(base_os)
            except Exception:
                base_os = float(cfg.grid_order_size)

            if base_os <= 0:
                base_os = float(cfg.grid_order_size)

            try:
                market = self.ex.x.market(symbol)
                if isinstance(market, dict):
                    limits = market.get("limits") or {}
                    amount_limits = limits.get("amount") or {}
                    min_amt = safe_float(amount_limits.get("min"))
                    if min_amt is not None and min_amt > 0 and base_os < min_amt:
                        base_os = min_amt
            except Exception:
                pass

            strategy = GridLikeStrategy(
                cfg.grid_point, base_os, cfg.grid_mf, cfg.grid_anti
            )
            self.contexts[symbol] = {
                "strategy": strategy,
                "last_ts": None,
                "active_trade": None,
            }

    def _price_tick(self, symbol: str) -> float:
        try:
            market = self.ex.x.market(symbol)
        except Exception:
            market = {}

        tick = None
        if isinstance(market, dict):
            info = market.get("info") or {}
            try:
                tick_val = info.get("tickSz")
                if tick_val is not None:
                    tick = safe_float(tick_val, None)
            except Exception:
                tick = None

        if tick is not None and tick > 0:
            return tick

        precision = None
        if isinstance(market, dict):
            precision = (market.get("precision") or {}).get("price")

        try:
            if precision is not None:
                prec_int = int(precision)
                if prec_int >= 0:
                    return 10 ** (-prec_int)
        except Exception:
            pass

        return 0.0001

    def _guard_tp_sl(
        self, symbol: str, side_open: str, tp: float, sl: float, last_price: Optional[float]
    ) -> tuple[float, float, str, str]:
        if last_price is None or last_price <= 0:
            return float(tp), float(sl), "last", "last"

        tick = max(self._price_tick(symbol), 1e-9)
        tp_f = float(tp)
        sl_f = float(sl)

        if side_open == "buy":
            tp_f = max(tp_f, last_price + tick)
            sl_f = min(sl_f, last_price - tick)
        else:
            tp_f = min(tp_f, last_price - tick)
            sl_f = max(sl_f, last_price + tick)

        return tp_f, sl_f, "last", "last"

    def _has_any_open_trade(self) -> bool:
        for ctx in self.contexts.values():
            if ctx.get("active_trade"):
                return True
        if hasattr(self.ex, "has_any_open_position"):
            try:
                if self.ex.has_any_open_position():
                    return True
            except Exception:
                pass
        return False

    def run(self) -> None:
        symbols = ", ".join(self.symbols)
        self.notifier.send(
            f"[START] Grid Like Strategy | OKX Demo | TF {self.timeframe} | Symbols: {symbols}"
        )
        while True:
            try:
                for symbol, ctx in self.contexts.items():
                    self._process_symbol(symbol, ctx)
                time.sleep(self.poll_interval)
            except KeyboardInterrupt:
                self.notifier.send("[EXIT] Stopping bot")
                break
            except Exception as exc:
                self.notifier.send(f"[ERROR] loop: {exc}")
                time.sleep(self.poll_interval)

    def _process_symbol(self, symbol: str, ctx: Dict[str, Any]) -> None:
        candles = self.ex.fetch_ohlcv(symbol, self.timeframe, limit=2)
        if len(candles) < 2:
            return

        ts, _open, high, low, close, _vol = candles[-2]
        if ctx["last_ts"] == ts:
            # still same candle: only check exit conditions using latest extremes
            self._evaluate_trade_exit(symbol, ctx, safe_float(high), safe_float(low))
            return

        ctx["last_ts"] = ts

        high_f = safe_float(high)
        low_f = safe_float(low)
        close_f = safe_float(close)
        if high_f is None or low_f is None or close_f is None:
            return

        self._evaluate_trade_exit(symbol, ctx, high_f, low_f)

        result = ctx["strategy"].update(close_f)
        signal = result.get("signal")
        if signal and ctx["active_trade"] is None:
            if self.single_global and self._has_any_open_trade():
                self.notifier.send(
                    "⛔ Signal ignored: an open position already exists (single-position mode)"
                )
                return
            self._execute_trade(symbol, ctx, close_f, result)

    def _evaluate_trade_exit(
        self, symbol: str, ctx: Dict[str, Any], high: Optional[float], low: Optional[float]
    ) -> None:
        trade = ctx.get("active_trade")
        if not trade or high is None or low is None:
            return

        side = trade["side"]
        tp = trade["tp"]
        sl = trade["sl"]
        hit_tp = False
        hit_sl = False

        if side == "buy":
            hit_tp = high >= tp
            hit_sl = low <= sl
        else:
            hit_tp = low <= tp
            hit_sl = high >= sl

        if not (hit_tp or hit_sl):
            return

        if hit_tp and hit_sl:
            outcome = "sl"
            exit_price = sl
        elif hit_tp:
            outcome = "tp"
            exit_price = tp
        else:
            outcome = "sl"
            exit_price = sl

        ctx["active_trade"] = None
        ctx["strategy"].on_close_result(
            was_win=(outcome == "tp"), was_loss=(outcome == "sl")
        )

        entry_price = trade.get("entry_price")
        emoji = "✅" if outcome == "tp" else "❌"
        message = (
            f"📤 Trade Closed {emoji}\n"
            f"• Pair: {symbol}\n"
            f"• Side: {side.upper()}\n"
            f"• Entry: {entry_price:.4f}\n"
            f"• Exit: {exit_price:.4f} (TP={tp:.4f} | SL={sl:.4f})"
        )
        self.notifier.send(message)

    def _execute_trade(
        self, symbol: str, ctx: Dict[str, Any], close_price: float, result: Dict[str, Any]
    ) -> None:
        tp = safe_float(result.get("tp"))
        sl = safe_float(result.get("sl"))
        if tp is None or sl is None:
            return

        side_val = result.get("signal")
        if side_val not in ("buy", "sell"):
            return
        side = str(side_val)

        try:
            ticker = self.ex.x.fetch_ticker(symbol)
        except Exception:
            ticker = {}

        last_price = safe_float((ticker or {}).get("last"))
        price_ref = safe_float(close_price)
        if last_price is None or last_price <= 0:
            last_price = price_ref
        if last_price is None or last_price <= 0:
            self.notifier.send(
                f"⚠️ Skipping {symbol}: unable to determine valid market price"
            )
            return

        raw_amount = safe_float(result.get("size"))
        if raw_amount is None:
            return

        try:
            amount = float(self.ex.x.amount_to_precision(symbol, raw_amount))
        except Exception:
            amount = float(raw_amount)

        try:
            market = self.ex.x.market(symbol)
        except Exception:
            market = {}

        limits = (market.get("limits") or {}).get("amount") if isinstance(market, dict) else None
        min_amt = safe_float((limits or {}).get("min"), None) if isinstance(limits, dict) else None
        max_amt = safe_float((limits or {}).get("max"), None) if isinstance(limits, dict) else None

        if min_amt is not None and amount < min_amt:
            amount = float(min_amt)
        if max_amt is not None and amount > max_amt:
            amount = float(max_amt)

        try:
            amount = float(self.ex.x.amount_to_precision(symbol, amount))
        except Exception:
            amount = float(amount)

        if amount <= 0:
            self.notifier.send(f"⚠️ Skipping {symbol}: amount<=0 after limits.")
            return

        self.ex.ensure_cross_leverage(symbol, self.leverage)
        order = self.ex.create_market_order(symbol, side, amount)
        if not order:
            self.notifier.send(f"❌ Order failed for {symbol}")
            return

        tp_adj, sl_adj, tp_type, sl_type = self._guard_tp_sl(
            symbol, side, tp, sl, last_price
        )

        try:
            tp_adj = float(self.ex.x.price_to_precision(symbol, tp_adj))
        except Exception:
            tp_adj = float(tp_adj)

        try:
            sl_adj = float(self.ex.x.price_to_precision(symbol, sl_adj))
        except Exception:
            sl_adj = float(sl_adj)

        tick = max(self._price_tick(symbol), 1e-9)
        if last_price is not None:
            if side == "buy":
                attempts = 0
                while not (tp_adj > last_price) and attempts < 5:
                    tp_adj = float(last_price + (attempts + 1) * tick)
                    try:
                        tp_adj = float(self.ex.x.price_to_precision(symbol, tp_adj))
                    except Exception:
                        tp_adj = float(tp_adj)
                    attempts += 1

                attempts = 0
                while not (sl_adj < last_price) and attempts < 5:
                    candidate = last_price - (attempts + 1) * tick
                    if candidate <= 0:
                        break
                    sl_adj = float(candidate)
                    try:
                        sl_adj = float(self.ex.x.price_to_precision(symbol, sl_adj))
                    except Exception:
                        sl_adj = float(sl_adj)
                    attempts += 1

                if not (tp_adj > last_price) or not (sl_adj < last_price) or sl_adj <= 0:
                    self.notifier.send(
                        f"⚠️ Skipping {symbol}: unable to guard TP/SL relative to last price"
                    )
                    return
            else:
                attempts = 0
                while not (tp_adj < last_price) and attempts < 5:
                    candidate = last_price - (attempts + 1) * tick
                    if candidate <= 0:
                        break
                    tp_adj = float(candidate)
                    try:
                        tp_adj = float(self.ex.x.price_to_precision(symbol, tp_adj))
                    except Exception:
                        tp_adj = float(tp_adj)
                    attempts += 1

                attempts = 0
                while not (sl_adj > last_price) and attempts < 5:
                    sl_adj = float(last_price + (attempts + 1) * tick)
                    try:
                        sl_adj = float(self.ex.x.price_to_precision(symbol, sl_adj))
                    except Exception:
                        sl_adj = float(sl_adj)
                    attempts += 1

                if not (tp_adj < last_price) or not (sl_adj > last_price):
                    self.notifier.send(
                        f"⚠️ Skipping {symbol}: unable to guard TP/SL relative to last price"
                    )
                    return

        exit_resp = self.ex.place_exit_algo(
            symbol, side, amount, tp_adj, sl_adj, tp_type, sl_type
        )
        if exit_resp is None:
            self.notifier.send(
                f"⚠️ Unable to confirm TP/SL algo order for {symbol}. Manage exits manually."
            )
        else:
            self.notifier.send(f"ℹ️ TP/SL algo order placed for {symbol}")

        order_id: Optional[str] = None
        if isinstance(order, dict):
            order_id = order.get("id")
            if not order_id:
                info = order.get("info", {})
                if isinstance(info, dict):
                    order_id = info.get("ordId") or info.get("orderId")

        ctx["active_trade"] = {
            "side": side,
            "size": amount,
            "tp": tp_adj,
            "sl": sl_adj,
            "entry_price": last_price,
            "entry_time": time.time(),
            "order_id": order_id,
        }

        message = (
            f"✅ EXECUTED {side.upper()} {symbol} @ MARKET\n"
            f"• Qty: {amount:.6f}\n"
            f"• Entry: {last_price:.4f}\n"
            f"• TP: {tp_adj:.4f} | SL: {sl_adj:.4f}\n"
            f"• OrderID: {order_id or 'n/a'}"
        )
        self.notifier.send(message)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid Like Strategy OKX Demo Bot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", help="Path to JSON config file", default=None)
    parser.add_argument("--timeframe", help="OHLC timeframe", default=None)
    parser.add_argument("--point", type=float, help="Grid point distance", default=None)
    parser.add_argument(
        "--order-size", dest="order_size", type=float, help="Base order size", default=None
    )
    parser.add_argument("--mf", type=float, help="Martingale multiplier", default=None)
    parser.add_argument(
        "--anti", dest="anti", action="store_true", help="Enable anti-martingale"
    )
    parser.add_argument(
        "--no-anti", dest="anti", action="store_false", help="Disable anti-martingale"
    )
    parser.add_argument(
        "--symbols", nargs="+", help="Symbols to trade (e.g. BTC/USDT:USDT)", default=None
    )
    parser.add_argument(
        "--poll", type=float, help="Polling interval in seconds", default=None
    )
    parser.add_argument("--leverage", type=int, help="Requested leverage", default=None)
    parser.set_defaults(anti=None)
    return parser.parse_args()


def load_config(path: Optional[str]) -> Any:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_config(args: argparse.Namespace) -> SimpleNamespace:
    raw_cfg = load_config(args.config)
    cfg = _to_ns(raw_cfg)

    if not hasattr(cfg, "timeframe"):
        cfg.timeframe = "5m"
    if not hasattr(cfg, "poll_interval"):
        cfg.poll_interval = 20.0
    if not hasattr(cfg, "telegram_token"):
        cfg.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not hasattr(cfg, "telegram_chat_id"):
        cfg.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not hasattr(cfg, "telegram_enabled"):
        cfg.telegram_enabled = bool(cfg.telegram_token and cfg.telegram_chat_id)
    if not hasattr(cfg, "grid_point"):
        cfg.grid_point = 2.0
    if not hasattr(cfg, "grid_order_size"):
        cfg.grid_order_size = 0.01
    if not hasattr(cfg, "grid_order_size_map"):
        cfg.grid_order_size_map = {}
    else:
        map_val = cfg.grid_order_size_map
        if isinstance(map_val, str):
            try:
                map_val = json.loads(map_val)
            except Exception:
                map_val = {}
        if isinstance(map_val, dict):
            cfg.grid_order_size_map = {
                str(key): float(val)
                for key, val in map_val.items()
                if val is not None
            }
        else:
            cfg.grid_order_size_map = {}
    if not hasattr(cfg, "grid_mf"):
        cfg.grid_mf = 2.0
    if not hasattr(cfg, "grid_anti"):
        cfg.grid_anti = False
    if not hasattr(cfg, "symbols"):
        cfg.symbols = ["BTC/USDT:USDT"]
    else:
        cfg.symbols = _ensure_symbols(cfg.symbols)
    if not hasattr(cfg, "leverage"):
        cfg.leverage = 10
    if not hasattr(cfg, "top_n"):
        cfg.top_n = 10
    if not hasattr(cfg, "single_global_position"):
        cfg.single_global_position = False

    if args.timeframe:
        cfg.timeframe = args.timeframe
    if args.order_size is not None:
        cfg.grid_order_size = float(args.order_size)
    if args.point is not None:
        cfg.grid_point = float(args.point)
    if args.mf is not None:
        cfg.grid_mf = float(args.mf)
    if args.anti is not None:
        cfg.grid_anti = bool(args.anti)
    if args.symbols:
        cfg.symbols = _ensure_symbols(args.symbols)
    if args.poll is not None:
        cfg.poll_interval = float(args.poll)
    if args.leverage is not None:
        cfg.leverage = int(args.leverage)

    cfg.symbols = list(dict.fromkeys(cfg.symbols))

    return cfg


def sanitize_cfg(cfg: SimpleNamespace) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for key, value in vars(cfg).items():
        if any(token in key.lower() for token in ("token", "secret", "pass")) and value:
            sanitized[key] = "***"
        else:
            sanitized[key] = value
    return sanitized


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    safe_view = sanitize_cfg(cfg)
    print("Config:\n", json.dumps(safe_view, indent=2, default=str))
    bot = Bot(cfg)
    bot.run()


if __name__ == "__main__":
    main()
