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

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 2) -> List[List[float]]:
        return self.x.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def normalize_amount(self, symbol: str, amount: float) -> float:
        try:
            prec = self.x.amount_to_precision(symbol, amount)
            return float(prec)
        except Exception:
            return float(amount)

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
        self, symbol: str, side: str, amount: float, tp: float, sl: float
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
            "tpTriggerPxType": "last",
            "slTriggerPxType": "last",
        }

        try:
            resp = method(payload)
            return resp if isinstance(resp, dict) else {"raw": resp}
        except Exception as exc:
            print(f"[WARN] Failed to place TP/SL algo order for {symbol}: {exc}")
            return None


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

        self.timeframe = getattr(cfg, "timeframe", "1m")
        self.leverage = int(getattr(cfg, "leverage", 10))
        self.poll_interval = float(getattr(cfg, "poll_interval", 20.0))

        self.contexts: Dict[str, Dict[str, Any]] = {}
        for symbol in cfg.symbols:
            strategy = GridLikeStrategy(
                cfg.grid_point, cfg.grid_order_size, cfg.grid_mf, cfg.grid_anti
            )
            self.contexts[symbol] = {
                "strategy": strategy,
                "last_ts": None,
                "active_trade": None,
            }

    def run(self) -> None:
        symbols = ", ".join(self.contexts.keys())
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
        size_val = safe_float(result.get("size"), 0.0)
        tp = safe_float(result.get("tp"))
        sl = safe_float(result.get("sl"))
        if size_val is None or size_val <= 0 or tp is None or sl is None:
            return

        amount = self.ex.normalize_amount(symbol, size_val)
        if amount <= 0:
            self.notifier.send(
                f"⚠️ Skipping {symbol}: normalized amount is non-positive ({amount})"
            )
            return

        self.ex.ensure_cross_leverage(symbol, self.leverage)
        order = self.ex.create_market_order(symbol, str(result.get("signal")), amount)
        if not order:
            self.notifier.send(f"❌ Order failed for {symbol}")
            return

        exit_resp = self.ex.place_exit_algo(symbol, str(result.get("signal")), amount, tp, sl)
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
            "side": str(result.get("signal")),
            "size": amount,
            "tp": tp,
            "sl": sl,
            "entry_price": close_price,
            "entry_time": time.time(),
            "order_id": order_id,
        }

        message = (
            f"✅ EXECUTED {str(result.get('signal')).upper()} {symbol} @ MARKET\n"
            f"• Qty: {amount:.6f}\n"
            f"• Entry: {close_price:.4f}\n"
            f"• TP: {tp:.4f} | SL: {sl:.4f}\n"
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
        cfg.timeframe = "1m"
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
        cfg.grid_order_size = 1.0
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
