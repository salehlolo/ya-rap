#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bot.py — Triple+3 Strategies (Self-Evolving) Scalper — OKX Demo Trading
(نسخة بدون أي تكامل مع OpenAI — تداول/إشعارات فقط)

تعليمي فقط — يعمل على حساب OKX التجريبي مع تنفيذ أوامر حقيقية على الديمو + Paper Engine.
Env:
  OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  (اختياري) CRYPTOPANIC_TOKEN, NEWSAPI_KEY  ← تقدر تسيبهم فاضيين
"""

import os, time, json, argparse, datetime as dt, random
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Callable

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import ccxt
import ta
import requests

# =========================
# Helpers
# =========================

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)

def fmt_ts(ts: Optional[dt.datetime] = None) -> str:
    t = (ts or now_utc())
    s = t.isoformat()
    return s.replace("+00:00", "Z")

def ensure_dir(path: str):
    d = path if os.path.splitext(path)[1] == "" else os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def clamp(v, lo, hi): return max(lo, min(hi, v))
def safe_float(x, default=np.nan):
    try: return float(x)
    except Exception: return default

# =========================
# Config
# =========================

@dataclass
class Config:
    timeframe: str = "5m"
    lookback: int = 800

    # Indicators / windows
    ema_fast: int = 9
    ema_slow: int = 21
    atr_window: int = 14
    rsi_len: int = 14
    bb_len: int = 20
    bb_std: float = 2.0
    vol_ma_len: int = 30
    box_len: int = 20
    regime_lookback: int = 120
    low_vol_pct: float = 0.35
    high_vol_pct: float = 0.70

    # Fixed TP/SL (يستخدم كخيار احتياطي)
    fixed_tp_pct: float = 0.01
    fixed_sl_pct: float = 0.005

    # ==== تعديل #2: إضافة إعدادات TP/SL الديناميكية ====
    use_atr_tp_sl: bool = True    # لتفعيل/تعطيل الميزة بسهولة
    atr_tp_mult: float = 2.0      # TP = السعر + (2.0 * ATR)
    atr_sl_mult: float = 1.2      # SL = السعر - (1.2 * ATR)
    
    # TREND
    trend_min_slope: float = 0.0003
    trend_vol_mult: float = 1.3

    # BO
    bo_vol_mult: float = 1.2
    bo_range_share: float = 0.5

    # MR
    mr_rsi_buy: float = 25.0
    mr_rsi_sell: float = 75.0
    sr_lookback: int = 50

    # PB (Pullback)
    pb_pullback_pct: float = 0.0035
    pb_wick_ratio: float = 0.35

    # VWAP-R
    vwap_dev_mult: float = 1.5

    # KSQ (Keltner Squeeze)
    keltner_len: int = 20
    keltner_mult: float = 1.5
    squeeze_bb_mult: float = 1.6

    # Sizing (display only)
    risk_k: float = 0.01
    max_position_value_usd: float = 250.0

    # Filters
    funding_filter: bool = True
    max_abs_funding: float = 0.003

    # Quiet windows (UTC HH:MM)
    event_quiet_minutes: int = 10
    quiet_windows_utc: Tuple[str, ...] = ()

    # News Guard — افتراضيًا مقفول
    news_enabled: bool = False
    news_lookback_minutes: int = 60
    news_keywords: Tuple[str, ...] = ("ETF","hack","exploit","ban","SEC","lawsuit","fork","upgrade","halving")

    # Universe
    # ==== Universe Size: Top 20 على OKX ====
    top_n_symbols: int = 20
    refresh_universe_minutes: int = 360
    health_refresh_minutes: int = 90
    health_test_limit: int = 50

    # Telegram
    telegram_enabled: bool = True
    telegram_token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")

    # Throttles
    min_minutes_between_same_signal: int = 3
    min_seconds_between_alerts_global: int = 50

    # Committee & Bandit
    exploration_eps: float = 0.08
    dyn_quorum_base: float = 0.55
    quorum_boost_high_vol: float = -0.07
    quorum_boost_good_hit: float = -0.05
    quorum_penalty_bad_hit: float = 0.05

    # Self-Evolving (محلي — مش بيعتمد على OpenAI)
    evolve_enabled: bool = True
    evolve_mutations_per_round: int = 2
    evolve_trial_weight: float = 0.25
    evolve_decay: float = 0.98

    # ==== الإضافات الجديدة (إدارة المخاطر المتقدمة) ====
    # Confidence Filter
    min_confidence_accept: float = 0.75

    # Committee Override: لازم أقل حاجة X نماذج تتفق على نفس الاتجاه
    committee_min_agree: int = 2

    # Daily Stop Loss: يوقف لباقي اليوم لو نزل -2% من رأس المال المرجعي
    daily_stop_enabled: bool = True
    daily_stop_pct: float = 0.02  # 2%

    # Files
    logs_dir: str = "./logs"
    signals_csv: str = "./logs/signals_log.csv"
    trades_csv: str  = "./logs/trades_log.csv"
    models_csv: str  = "./logs/models_log.csv"
    ml_csv: str      = "./logs/ml_dataset.csv"
    state_json: str  = "./logs/state.json"

# =========================
# Telegram
# =========================

class Notifier:
    def __init__(self, cfg: Config):
        self.enabled = bool(cfg.telegram_enabled and cfg.telegram_token and cfg.telegram_chat_id)
        self.base = f"https://api.telegram.org/bot{cfg.telegram_token}" if self.enabled else None
        self.chat_id = cfg.telegram_chat_id
    def send(self, text: str):
        if not self.enabled:
            print(text)
            return
        try:
            r = requests.post(f"{self.base}/sendMessage",
                              json={"chat_id": self.chat_id, "text": text, "disable_web_page_preview": True},
                              timeout=10)
            if r.status_code != 200:
                print("[WARN] Telegram send failed:", r.text)
        except Exception as e:
            print("[WARN] Telegram exception:", e)

# =========================
# Exchange
# =========================

class FuturesExchange:
    def __init__(self, cfg: Config):
        key = os.getenv("OKX_API_KEY")
        secret = os.getenv("OKX_SECRET_KEY")
        password = os.getenv("OKX_PASSPHRASE")
        self.x = ccxt.okx({
            "apiKey": key,
            "secret": secret,
            "password": password,
            "enableRateLimit": True,
            "timeout": 15000,
            "options": {
                "fetchCurrencies": False,
            },
        })
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
        # === Detect/force position mode (NET vs Hedged) ===
        self.is_hedged = False
        forced_mode = False
        try:
            if hasattr(self.x, "set_position_mode"):
                self.x.set_position_mode(False)
                forced_mode = True
                self.is_hedged = False
        except Exception:
            forced_mode = False
            try:
                get_cfg = (getattr(self.x, "private_get_account_config", None)
                           or getattr(self.x, "privateGetAccountConfig", None))
                if get_cfg:
                    cfg = get_cfg()
                    data = (cfg.get("data") or [{}]) if isinstance(cfg, dict) else [{}]
                    first = data[0] if data else {}
                    mode = first.get("posMode") if isinstance(first, dict) else None
                    self.is_hedged = (mode == "long_short_mode")
            except Exception:
                pass
        if not forced_mode and not self.is_hedged:
            try:
                get_cfg = (getattr(self.x, "private_get_account_config", None)
                           or getattr(self.x, "privateGetAccountConfig", None))
                if get_cfg:
                    cfg = get_cfg()
                    data = (cfg.get("data") or [{}]) if isinstance(cfg, dict) else [{}]
                    first = data[0] if data else {}
                    mode = first.get("posMode") if isinstance(first, dict) else None
                    self.is_hedged = (mode == "long_short_mode")
            except Exception:
                pass
        self.cfg = cfg
        self._universe_cache: Dict[str, any] = {"ts": 0.0, "symbols": []}
        self._health_cache: Dict[str, float] = {}
        self._bad_cache: Dict[str, float] = {}

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        ohlcv = self.x.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("datetime").drop(columns=["timestamp"])
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna()

    def fetch_funding_rate(self, symbol: str) -> Optional[float]:
        try:
            fr = self.x.fetch_funding_rate(symbol)
            if isinstance(fr, dict):
                val = fr.get("fundingRate")
                if val is None and hasattr(fr, "fundingRate"):
                    val = getattr(fr, "fundingRate", None)
                return safe_float(val, default=None)
            return safe_float(fr, default=None)
        except Exception:
            try:
                market = self.x.market(symbol)
                method = getattr(self.x, "public_get_public_funding_rate", None)
                if not method:
                    method = getattr(self.x, "publicGetPublicFundingRate", None)
                if not method:
                    return None
                data = method({"instId": market["id"]})
                items = (data or {}).get("data") if isinstance(data, dict) else None
                if items:
                    rate = items[0].get("fundingRate")
                    return safe_float(rate, default=None)
            except Exception:
                return None
        return None

    def get_balance_usdt(self) -> float:
        try:
            bal = self.x.fetch_balance(params={"type": "swap"})
            total = bal.get("total", {}) if isinstance(bal, dict) else {}
            if isinstance(total, dict) and total.get("USDT") is not None:
                return float(total.get("USDT", 0.0))
            usdt = bal.get("USDT") if isinstance(bal, dict) else None
            if isinstance(usdt, dict) and usdt.get("total") is not None:
                return float(usdt.get("total", 0.0))
            info = bal.get("info", {}) if isinstance(bal, dict) else {}
            data = (info.get("data") or [{}]) if isinstance(info, dict) else [{}]
            details = data[0].get("details") if isinstance(data[0], dict) else None
            if isinstance(details, list):
                for d in details:
                    if d.get("ccy") == "USDT":
                        eq = d.get("eq") or d.get("cashBal") or d.get("availEq")
                        if eq is not None:
                            return float(eq)
        except Exception:
            return 0.0
        return 0.0

    def get_top_symbols(self, n: int = 50) -> List[str]:
        nowt = time.time()
        if (nowt - self._universe_cache["ts"]) < self.cfg.refresh_universe_minutes*60 and self._universe_cache["symbols"]:
            return self._universe_cache["symbols"]
        top: List[Tuple[str,float]] = []
        try:
            tickers = self.x.fetch_tickers()
            for m in self.x.markets.values():
                if not m.get("swap") or not m.get("contract"): continue
                if m.get("quote") != "USDT": continue
                if not m.get("active", True): continue
                sym = m["symbol"]
                t = tickers.get(sym, {})
                qv = t.get("quoteVolume")
                if qv is None:
                    info = t.get("info", {}) if isinstance(t, dict) else {}
                    qv = info.get("quoteVolume24h") or info.get("volCcy24h") or info.get("volUsd24h")
                try:
                    qv_float = float(qv)
                except Exception:
                    qv_float = 0.0
                top.append((sym, qv_float))
            top.sort(key=lambda x: x[1], reverse=True)
            syms = [s for s,_ in top[:n]] or ["BTC/USDT:USDT","ETH/USDT:USDT"]
        except Exception:
            syms = ["BTC/USDT:USDT","ETH/USDT:USDT"]
        self._universe_cache = {"ts": nowt, "symbols": syms}
        return syms

    def filter_healthy(self, symbols: List[str]) -> List[str]:
        ok = []
        nowt = time.time()
        for s in symbols:
            last_bad = self._bad_cache.get(s, 0)
            if nowt - last_bad < self.cfg.health_refresh_minutes*60:
                continue
            last_ok = self._health_cache.get(s, 0)
            if nowt - last_ok < self.cfg.health_refresh_minutes*60:
                ok.append(s); continue
            try:
                _ = self.fetch_ohlcv(s, self.cfg.timeframe, limit=self.cfg.health_test_limit)
                if _.shape[0] > 10:
                    ok.append(s); self._health_cache[s] = nowt
                else:
                    self._bad_cache[s] = nowt
            except Exception:
                self._bad_cache[s] = nowt
            time.sleep(0.05)
        if not ok:
            ok = ["BTC/USDT","ETH/USDT"]
        return ok

# =========================
# Indicators
# =========================

def compute_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = df.copy()

    d["ema9"]  = ta.trend.EMAIndicator(d["close"], window=cfg.ema_fast).ema_indicator()
    d["ema21"] = ta.trend.EMAIndicator(d["close"], window=cfg.ema_slow).ema_indicator()

    d["rsi"] = ta.momentum.RSIIndicator(d["close"], window=cfg.rsi_len).rsi()

    bb = ta.volatility.BollingerBands(d["close"], window=cfg.bb_len, window_dev=cfg.bb_std)
    d["bb_mid"], d["bb_up"], d["bb_dn"] = bb.bollinger_mavg(), bb.bollinger_hband(), bb.bollinger_lband()

    atr = ta.volatility.AverageTrueRange(d["high"], d["low"], d["close"], window=cfg.atr_window)
    d["atr"] = atr.average_true_range()
    d["atr_pct"] = d["atr"] / d["close"]

    session = d.index.tz_convert("UTC").normalize()
    d["vwap_num"] = (d["close"] * d["volume"]).groupby(session).cumsum()
    d["vwap_den"] = d["volume"].groupby(session).cumsum().replace(0, np.nan)
    d["vwap"] = d["vwap_num"] / d["vwap_den"]

    d["vol_ma"] = d["volume"].rolling(cfg.vol_ma_len).mean()
    d["vol_spike"] = d["volume"] > (d["vol_ma"] * cfg.trend_vol_mult)
    d["bo_vol_spike"] = d["volume"] > (d["vol_ma"] * cfg.bo_vol_mult)

    d["recent_high"] = d["high"].rolling(cfg.box_len).max()
    d["recent_low"]  = d["low"].rolling(cfg.box_len).min()

    d["sr_high"] = d["high"].rolling(cfg.sr_lookback).max()
    d["sr_low"]  = d["low"].rolling(cfg.sr_lookback).min()

    d["ema9_slope"] = (d["ema9"] - d["ema9"].shift(3)) / d["close"]
    d["ema21_slope"] = (d["ema21"] - d["ema21"].shift(3)) / d["close"]
    d["bb_width"] = (d["bb_up"] - d["bb_dn"]) / d["bb_mid"]

    adx = ta.trend.ADXIndicator(d["high"], d["low"], d["close"], window=cfg.atr_window)
    d["adx"] = adx.adx()
    d["di_pos"] = adx.adx_pos()
    d["di_neg"] = adx.adx_neg()

    ema = ta.trend.EMAIndicator(d["close"], window=cfg.keltner_len).ema_indicator()
    rng = ta.volatility.AverageTrueRange(d["high"], d["low"], d["close"], window=cfg.keltner_len).average_true_range()
    d["kel_mid"] = ema
    d["kel_up"] = ema + cfg.keltner_mult * rng
    d["kel_dn"] = ema - cfg.keltner_mult * rng

    def rolling_pctl(x: pd.Series):
        last = x.iloc[-1]
        return float((x <= last).mean())
    d["atr_pct_pctl"] = d["atr_pct"].rolling(cfg.regime_lookback).apply(rolling_pctl, raw=False)

    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    return d

# =========================
# Regime
# =========================

@dataclass
class Regime:
    trend: str
    vol_bucket: str

def classify_regime(row: pd.Series, cfg: Config) -> Regime:
    if row["ema9"] > row["ema21"] and row["close"] > row["vwap"]:
        t = "up"
    elif row["ema9"] < row["ema21"] and row["close"] < row["vwap"]:
        t = "down"
    else:
        t = "neutral"
    p = row.get("atr_pct_pctl", np.nan)
    if np.isnan(p): v = "medium"
    elif p < cfg.low_vol_pct: v = "low"
    elif p > cfg.high_vol_pct: v = "high"
    else: v = "medium"
    return Regime(t, v)

# =========================
# Signal object
# =========================

@dataclass
class Signal:
    side: Optional[str]
    sl: float
    tp: float
    model: str
    reason: str
    confidence: float = 0.5

# ==== تعديل #2: إضافة دالة لحساب TP/SL بناءً على ATR ====
def make_tp_sl_atr(entry: float, side: str, atr: float, cfg: Config) -> Tuple[float, float]:
    """يحسب TP/SL بناءً على مضاعفات الـ ATR."""
    if side == "buy":
        tp = entry + (cfg.atr_tp_mult * atr)
        sl = entry - (cfg.atr_sl_mult * atr)
    else: # sell
        tp = entry - (cfg.atr_tp_mult * atr)
        sl = entry + (cfg.atr_sl_mult * atr)
    return tp, sl

def make_tp_sl(entry: float, side: str, cfg: Config) -> Tuple[float, float]:
    if side == "buy":
        tp = entry * (1 + cfg.fixed_tp_pct)
        sl = entry * (1 - cfg.fixed_sl_pct)
    else:
        tp = entry * (1 - cfg.fixed_tp_pct)
        sl = entry * (1 + cfg.fixed_sl_pct)
    return tp, sl

def get_tp_sl(entry: float, side: str, row: pd.Series, cfg: Config) -> Tuple[float, float]:
    """دالة موحدة لاختيار طريقة حساب TP/SL."""
    atr_val = safe_float(row.get("atr"))
    if cfg.use_atr_tp_sl and not np.isnan(atr_val) and atr_val > 0:
        return make_tp_sl_atr(entry, side, atr_val, cfg)
    else:
        return make_tp_sl(entry, side, cfg)

# =========================
# Base strategies
# =========================

def sig_trend(row: pd.Series, cfg: Config) -> Optional[Signal]:
    # ==== تعديل #3: فلترة حسب حالة السوق (Regime Filtering) ====
    regime = classify_regime(row, cfg)
    if regime.trend == "neutral":
        return None
        
    up = (row["ema9"] > row["ema21"]) and (row["close"] > row["ema9"]) and (row["ema9_slope"] > cfg.trend_min_slope)
    dn = (row["ema9"] < row["ema21"]) and (row["close"] < row["ema9"]) and (row["ema9_slope"] < -cfg.trend_min_slope)
    
    if up or dn:
        side = "buy" if up else "sell"
        entry = float(row["close"])
        tp, sl = get_tp_sl(entry, side, row, cfg)
        if up:
            conf = clamp(0.55 + (0.2 if row.get("vol_spike", False) else 0.0) + (0.15 if row["ema21_slope"]>0 else 0.0),0,1)
            return Signal("buy", sl, tp, "TREND", "EMA9>EMA21 + slope + vol", conf)
        if dn:
            conf = clamp(0.55 + (0.2 if row.get("vol_spike", False) else 0.0) + (0.15 if row["ema21_slope"]<0 else 0.0),0,1)
            return Signal("sell", sl, tp, "TREND", "EMA9<EMA21 + slope + vol", conf)
    return None

def sig_bo(row: pd.Series, cfg: Config) -> Optional[Signal]:
    # ==== تعديل #3: فلترة حسب حالة السوق (Regime Filtering) ====
    regime = classify_regime(row, cfg)
    if regime.trend == "neutral":
        return None
        
    atr = safe_float(row.get("atr", np.nan))
    if np.isnan(atr) or atr <= 0: return None
    rng = (row["high"] - row["low"])
    if not (rng > cfg.bo_range_share * atr): return None
    
    above = row["close"] >= row["recent_high"] * 0.999
    below = row["close"] <= row["recent_low"] * 1.001
    
    if above or below:
        side = "buy" if above else "sell"
        entry = float(row["close"])
        tp, sl = get_tp_sl(entry, side, row, cfg)
        if above:
            conf = clamp(0.6 + (0.2 if row.get("bo_vol_spike", False) else 0.0) + (0.2 if row["ema9"]>row["ema21"] else 0.0),0,1)
            return Signal("buy", sl, tp, "BO", "High breakout + range + vol", conf)
        if below:
            conf = clamp(0.6 + (0.2 if row.get("bo_vol_spike", False) else 0.0) + (0.2 if row["ema9"]<row["ema21"] else 0.0),0,1)
            return Signal("sell", sl, tp, "BO", "Low breakout + range + vol", conf)
    return None

def sig_mr(row: pd.Series, cfg: Config) -> Optional[Signal]:
    # ==== تعديل #3: فلترة حسب حالة السوق (Regime Filtering) ====
    regime = classify_regime(row, cfg)
    if regime.trend != "neutral":
        return None
        
    if np.isnan(row.get("rsi", np.nan)): return None
    price = float(row["close"])
    
    is_buy = (row["rsi"] <= cfg.mr_rsi_buy) and (price <= row["bb_dn"])
    is_sell = (row["rsi"] >= cfg.mr_rsi_sell) and (price >= row["bb_up"])
    
    if is_buy or is_sell:
        side = "buy" if is_buy else "sell"
        tp, sl = get_tp_sl(price, side, row, cfg)
        if is_buy:
            conf = clamp(0.55 + (0.2 if price <= row["sr_low"]*1.003 else 0.0) + (0.1 if row["ema9"]>row["ema21"] else 0.0),0,1)
            return Signal("buy", sl, tp, "MR", "RSI low + lower band + S/R", conf)
        if is_sell:
            conf = clamp(0.55 + (0.2 if price >= row["sr_high"]*0.997 else 0.0) + (0.1 if row["ema9"]<row["ema21"] else 0.0),0,1)
            return Signal("sell", sl, tp, "MR", "RSI high + upper band + S/R", conf)
    return None

# =========================
# New strategies (PB / VWAP-R / KSQ)
# =========================

def sig_pb(row: pd.Series, cfg: Config) -> Optional[Signal]:
    # ==== تعديل #3: فلترة حسب حالة السوق (Regime Filtering) ====
    regime = classify_regime(row, cfg)
    if regime.trend == "neutral":
        return None

    wick_up = row["high"] - max(row["close"], row["open"])
    wick_dn = min(row["close"], row["open"]) - row["low"]
    wick_ratio_up = wick_up / max((row["high"] - row["low"]), 1e-9)
    wick_ratio_dn = wick_dn / max((row["high"] - row["low"]), 1e-9)

    if row["ema9"] > row["ema21"]:
        near = (row["close"] >= row["ema21"]*(1 - cfg.pb_pullback_pct)) and (row["close"] <= row["ema21"]*(1 + cfg.pb_pullback_pct))
        if near and wick_ratio_dn >= cfg.pb_wick_ratio and row["ema21_slope"] > 0:
            price = float(row["close"]); tp, sl = get_tp_sl(price,"buy", row, cfg)
            conf = clamp(0.58 + (0.12 if row.get("vol_spike", False) else 0.0),0,1)
            return Signal("buy", sl, tp, "PB", "Pullback to EMA21 + bullish wick", conf)

    if row["ema9"] < row["ema21"]:
        near = (row["close"] >= row["ema21"]*(1 - cfg.pb_pullback_pct)) and (row["close"] <= row["ema21"]*(1 + cfg.pb_pullback_pct))
        if near and wick_ratio_up >= cfg.pb_wick_ratio and row["ema21_slope"] < 0:
            price = float(row["close"]); tp, sl = get_tp_sl(price,"sell", row, cfg)
            conf = clamp(0.58 + (0.12 if row.get("vol_spike", False) else 0.0),0,1)
            return Signal("sell", sl, tp, "PB", "Pullback to EMA21 + bearish wick", conf)
    return None

def sig_vwap_r(row: pd.Series, cfg: Config) -> Optional[Signal]:
    # ==== تعديل #3: فلترة حسب حالة السوق (Regime Filtering) ====
    regime = classify_regime(row, cfg)
    if regime.trend != "neutral":
        return None
        
    dev = abs(row["close"] - row["vwap"]) / row["close"]
    threshold = max(cfg.vwap_dev_mult * (safe_float(row.get("atr_pct"), 0.0) or 0.0), 0.0025)
    if dev < threshold: return None
    price = float(row["close"])
    
    if row["close"] < row["vwap"] and row["rsi"] <= cfg.mr_rsi_buy:
        tp, sl = get_tp_sl(price, "buy", row, cfg)
        return Signal("buy", sl, tp, "VWAP-R", "Deep below VWAP + RSI low", 0.56)
    if row["close"] > row["vwap"] and row["rsi"] >= cfg.mr_rsi_sell:
        tp, sl = get_tp_sl(price, "sell", row, cfg)
        return Signal("sell", sl, tp, "VWAP-R", "Deep above VWAP + RSI high", 0.56)
    return None

def sig_ksq(row: pd.Series, cfg: Config) -> Optional[Signal]:
    # ==== تعديل #3: فلترة حسب حالة السوق (Regime Filtering) ====
    regime = classify_regime(row, cfg)
    if regime.trend == "neutral":
        return None
        
    squeeze = row.get("bb_width", np.nan) < (safe_float(row.get("atr_pct"),0)*cfg.squeeze_bb_mult)
    if not squeeze: return None
    price = float(row["close"])
    
    if (price >= row["kel_up"]) and (row["di_pos"] > row["di_neg"]) and (row["ema21_slope"] > 0):
        tp, sl = get_tp_sl(price, "buy", row, cfg)
        return Signal("buy", sl, tp, "KSQ", "Squeeze + break above Keltner + DI+", 0.6)
    if (price <= row["kel_dn"]) and (row["di_neg"] > row["di_pos"]) and (row["ema21_slope"] < 0):
        tp, sl = get_tp_sl(price, "sell", row, cfg)
        return Signal("sell", sl, tp, "KSQ", "Squeeze + break below Keltner + DI-", 0.6)
    return None

# =========================
# Self-Evolving (محلي)
# =========================

def mutate_params(cfg: Config) -> Dict[str,float]:
    g = {}
    def jitter(val, rel=0.2, lo=None, hi=None):
        v = val * (1 + random.uniform(-rel, rel))
        if lo is not None: v = max(v, lo)
        if hi is not None: v = min(v, hi)
        return v
    g["trend_min_slope"] = jitter(cfg.trend_min_slope, 0.5, 0.00005, 0.0015)
    g["bo_range_share"]  = clamp(jitter(cfg.bo_range_share, 0.3, 0.2, 0.8), 0.2, 0.8)
    g["mr_rsi_buy"]      = clamp(jitter(cfg.mr_rsi_buy, 0.2), 10, 40)
    g["mr_rsi_sell"]     = clamp(jitter(cfg.mr_rsi_sell, 0.2), 60, 90)
    g["pb_pullback_pct"] = clamp(jitter(cfg.pb_pullback_pct, 0.5, 0.001, 0.008), 0.001, 0.01)
    g["keltner_mult"]    = clamp(jitter(cfg.keltner_mult, 0.3, 1.0, 2.5), 1.0, 3.0)
    return g

def apply_mutated_signal(row: pd.Series, base_name: str, params: Dict[str,float], cfg: Config) -> Optional[Signal]:
    c2 = Config(**asdict(cfg))
    for k,v in params.items():
        setattr(c2, k, v)
    if base_name=="TREND": return sig_trend(row, c2)
    if base_name=="BO":    return sig_bo(row, c2)
    if base_name=="MR":    return sig_mr(row, c2)
    if base_name=="PB":    return sig_pb(row, c2)
    if base_name=="VWAP-R":return sig_vwap_r(row, c2)
    if base_name=="KSQ":   return sig_ksq(row, c2)
    return None

# =========================
# Committee & Bandit memory
# =========================

def ctx_key(regime: Regime) -> str:
    return f"{regime.trend}|{regime.vol_bucket}"

class Bandit:
    def __init__(self, state: dict):
        self.s = state.setdefault("bandit", {})

    def _slot(self, key: str) -> Dict[str,float]:
        d = self.s.setdefault(key, {"a":1.0,"b":1.0,"w":1.0})
        if "w" not in d: d["w"]=1.0
        return d

    def weight(self, key: str) -> float:
        d = self._slot(key)
        mean = d["a"]/(d["a"]+d["b"])
        return 0.3 + 0.7*mean

    def update(self, key: str, result: str):
        d = self._slot(key)
        if result == "tp": d["a"] += 1.0
        else: d["b"] += 1.0

    def decay_weights(self, factor: float = 0.995):
        for k in list(self.s.keys()):
            self.s[k]["w"] = max(0.1, self.s[k].get("w",1.0) * factor)

# =========================
# Paper Engine
# =========================

@dataclass
class PaperTrade:
    id: str
    timestamp: str
    symbol: str
    timeframe: str
    side: str
    entry: float
    sl: float
    tp: float
    model: str
    status: str = "open"
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    result: Optional[str] = None
    pnl_usd: Optional[float] = None

class Paper:
    def __init__(self, cfg: Config, ref_equity: float):
        self.cfg = cfg
        self.ref_equity = ref_equity
        self.open: Dict[str, PaperTrade] = {}
        ensure_dir(cfg.signals_csv); ensure_dir(cfg.trades_csv); ensure_dir(cfg.ml_csv); ensure_dir(cfg.models_csv); ensure_dir(cfg.state_json)
        if not os.path.exists(cfg.signals_csv):
            pd.DataFrame(columns=[
                "time","symbol","tf","price","side","model","tp","sl",
                "ref_qty","ref_notional","rr","reason","conf",
                "trend","vol_bucket","ctx_key"
            ]).to_csv(cfg.signals_csv, index=False)
        if not os.path.exists(cfg.trades_csv):
            pd.DataFrame(columns=[
                "id","open_time","close_time","symbol","tf","side","entry","exit","result","model","pnl_usd","hold_sec","ctx_key"
            ]).to_csv(cfg.trades_csv, index=False)
        if not os.path.exists(cfg.ml_csv):
            pd.DataFrame(columns=[
                "trade_id","symbol","tf","side","model","open_time","close_time","result","pnl_usd",
                "price","ema9","ema21","ema9_slope","ema21_slope","rsi","bb_mid","bb_up","bb_dn","bb_width",
                "atr","atr_pct","atr_pct_pctl","vwap","vol","vol_ma","vol_spike","bo_vol_spike","recent_high",
                "recent_low","sr_high","sr_low","regime_trend","regime_vol","ctx_key","adx","di_pos","di_neg",
                "kel_mid","kel_up","kel_dn"
            ]).to_csv(cfg.ml_csv, index=False)
        if not os.path.exists(cfg.models_csv):
            pd.DataFrame(columns=["time","symbol","tf","model","ctx_key","decision_score","accepted","weight","conf","notes"]).to_csv(cfg.models_csv, index=False)

    def _gen_id(self) -> str: return f"T{int(time.time()*1000)}"

    def open_virtual(self, symbol: str, price: float, sig: Signal, cfg: Config) -> PaperTrade:
        t = PaperTrade(
            id=self._gen_id(), timestamp=fmt_ts(), symbol=symbol, timeframe=cfg.timeframe,
            side=sig.side, entry=price, sl=float(sig.sl), tp=float(sig.tp), model=sig.model
        )
        self.open[t.id] = t
        return t

    def _hit(self, side: str, high: float, low: float, level: float, is_tp: bool) -> bool:
        if side == "buy":
            return high >= level if is_tp else low <= level
        else:
            return low <= level if is_tp else high >= level

    def update_with_candle(self, symbol: str, high: float, low: float):
        to_close = []
        for tid, t in list(self.open.items()):
            if t.status != "open" or t.symbol != symbol: continue
            hit_tp = self._hit(t.side, high, low, t.tp, True)
            hit_sl = self._hit(t.side, high, low, t.sl, False)
            if hit_tp and hit_sl:
                res, px = "sl", t.sl
            elif hit_tp:
                res, px = "tp", t.tp
            elif hit_sl:
                res, px = "sl", t.sl
            else:
                continue
            t.status = "closed"; t.result = res; t.exit_price = float(px); t.exit_time = fmt_ts()
            notional_ref = 100.0
            qty = notional_ref / t.entry
            pnl = (t.exit_price - t.entry) * qty * (1 if t.side=="buy" else -1)
            t.pnl_usd = round(pnl,4)
            to_close.append(tid)
        closed = [self.open[k] for k in to_close]
        for k in to_close: self.open.pop(k, None)
        return closed

    def persist_closed(self, closed: List[PaperTrade], cfg: Config, ctx: str):
        if not closed: return
        rows=[]
        for t in closed:
            hold = int((pd.to_datetime(t.exit_time) - pd.to_datetime(t.timestamp)).total_seconds())
            rows.append({
                "id": t.id, "open_time": t.timestamp, "close_time": t.exit_time, "symbol": t.symbol, "tf": t.timeframe,
                "side": t.side, "entry": round(t.entry,6), "exit": round(t.exit_price,6),
                "result": t.result, "model": t.model, "pnl_usd": t.pnl_usd, "hold_sec": hold, "ctx_key": ctx
            })
        pd.DataFrame(rows).to_csv(cfg.trades_csv, mode="a", header=False, index=False)

    def ml_snapshot(self, trade_id:str, symbol:str, row:pd.Series, regime:Regime):
        feat = {
            "trade_id": trade_id, "symbol": symbol, "tf": self.cfg.timeframe, "side": "", "model": "",
            "open_time": fmt_ts(), "close_time":"", "result":"", "pnl_usd":"",
            "price": float(row["close"]),
            "ema9": float(row.get("ema9", np.nan) or np.nan),
            "ema21": float(row.get("ema21", np.nan) or np.nan),
            "ema9_slope": float(row.get("ema9_slope", np.nan) or np.nan),
            "ema21_slope": float(row.get("ema21_slope", np.nan) or np.nan),
            "rsi": float(row.get("rsi", np.nan) or np.nan),
            "bb_mid": float(row.get("bb_mid", np.nan) or np.nan),
            "bb_up": float(row.get("bb_up", np.nan) or np.nan),
            "bb_dn": float(row.get("bb_dn", np.nan) or np.nan),
            "bb_width": float(row.get("bb_width", np.nan) or np.nan),
            "atr": float(row.get("atr", np.nan) or np.nan),
            "atr_pct": float(row.get("atr_pct", np.nan) or np.nan),
            "atr_pct_pctl": float(row.get("atr_pct_pctl", np.nan) or np.nan),
            "vwap": float(row.get("vwap", np.nan) or np.nan),
            "vol": float(row.get("volume", np.nan) or np.nan),
            "vol_ma": float(row.get("vol_ma", np.nan) or np.nan),
            "vol_spike": bool(row.get("vol_spike", False)),
            "bo_vol_spike": bool(row.get("bo_vol_spike", False)),
            "recent_high": float(row.get("recent_high", np.nan) or np.nan),
            "recent_low": float(row.get("recent_low", np.nan) or np.nan),
            "sr_high": float(row.get("sr_high", np.nan) or np.nan),
            "sr_low": float(row.get("sr_low", np.nan) or np.nan),
            "regime_trend": regime.trend, "regime_vol": regime.vol_bucket, "ctx_key": ctx_key(regime),
            "adx": float(row.get("adx", np.nan) or np.nan),
            "di_pos": float(row.get("di_pos", np.nan) or np.nan),
            "di_neg": float(row.get("di_neg", np.nan) or np.nan),
            "kel_mid": float(row.get("kel_mid", np.nan) or np.nan),
            "kel_up": float(row.get("kel_up", np.nan) or np.nan),
            "kel_dn": float(row.get("kel_dn", np.nan) or np.nan),
        }
        pd.DataFrame([feat]).to_csv(self.cfg.ml_csv, mode="a", header=False, index=False)

    def log_signal(self, symbol: str, row: pd.Series, sig: Signal, qty_ref: float,
                   notional_ref: float, rr: Optional[float], cfg: Config, regime: Regime):
        pd.DataFrame([{
            "time": fmt_ts(), "symbol": symbol, "tf": cfg.timeframe, "price": float(row["close"]),
            "side": sig.side, "model": sig.model, "tp": round(sig.tp,6), "sl": round(sig.sl,6),
            "ref_qty": round(qty_ref,8), "ref_notional": round(notional_ref,2),
            "rr": rr if rr is not None else "", "reason": sig.reason, "conf": round(sig.confidence,2),
            "trend": regime.trend, "vol_bucket": regime.vol_bucket, "ctx_key": ctx_key(regime)
        }]).to_csv(cfg.signals_csv, mode="a", header=False, index=False)

    def log_model_vote(self, cfg: Config, symbol: str, regime: Regime, model: str, score: float, accepted: bool, weight: float, conf: float, notes: str = ""):
        pd.DataFrame([{
            "time": fmt_ts(), "symbol": symbol, "tf": cfg.timeframe, "model": model, "ctx_key": ctx_key(regime),
            "decision_score": round(score,4), "accepted": int(accepted), "weight": round(weight,4), "conf": round(conf,4),
            "notes": notes[:120]
        }]).to_csv(cfg.models_csv, mode="a", header=False, index=False)

# =========================
# News (اختياري)
# =========================

class NewsGuard:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cp_token = os.getenv("CRYPTOPANIC_TOKEN")
        self.newsapi_key = os.getenv("NEWSAPI_KEY")
        self.cache: Dict[str, float] = {}
    def too_hot(self, asset: str) -> bool:
        if not self.cfg.news_enabled: return False
        hot = False
        try:
            if self._cryptopanic(asset): hot = True
        except Exception: pass
        try:
            if self._newsapi(asset): hot = True
        except Exception: pass
        if hot: self.cache[asset] = time.time()
        else:
            if asset in self.cache and (time.time() - self.cache[asset]) < 600:
                hot = True
        return hot
    def _cryptopanic(self, asset: str) -> bool:
        if not self.cp_token: return False
        url = "https://cryptopanic.com/api/v1/posts/"
        params = {"auth_token": self.cp_token, "currencies": asset.lower(), "kind": "news", "public": "true"}
        r = requests.get(url, params=params, timeout=8)
        if r.status_code != 200: return False
        data = r.json().get("results", [])
        since = int((now_utc() - dt.timedelta(minutes=self.cfg.news_lookback_minutes)).timestamp())
        for it in data:
            pub = it.get("published_at") or it.get("created_at") or ""
            try: ts = dt.datetime.fromisoformat(pub.replace("Z","+00:00")).timestamp()
            except Exception: ts = 0
            if ts >= since:
                title = (it.get("title") or "").lower()
                important = it.get("importance") in ("high","very_high")
                kw = any(k.lower() in title for k in self.cfg.news_keywords)
                if important or kw: return True
        return False
    def _newsapi(self, asset: str) -> bool:
        if not self.newsapi_key: return False
        q = "bitcoin" if asset.upper()=="BTC" else ("ethereum" if asset.upper()=="ETH" else asset)
        url = "https://newsapi.org/v2/everything"
        since = (now_utc() - dt.timedelta(minutes=self.cfg.news_lookback_minutes)).isoformat()
        params = {"q": q, "from": since, "language": "en", "sortBy": "publishedAt",
                  "apiKey": self.newsapi_key, "pageSize": 20}
        r = requests.get(url, params=params, timeout=8)
        if r.status_code != 200: return False
        arts = r.json().get("articles", [])
        for a in arts:
            title = (a.get("title") or "").lower()
            desc  = (a.get("description") or "").lower()
            if any(k.lower() in title or k.lower() in desc for k in self.cfg.news_keywords):
                return True
        return False

# =========================
# Quiet & sizing
# =========================

def in_quiet_window(cfg: Config) -> bool:
    if not cfg.quiet_windows_utc: return False
    nowt = now_utc().time().replace(second=0, microsecond=0)
    for hhmm in cfg.quiet_windows_utc:
        try:
            t = dt.datetime.strptime(hhmm, "%H:%M").time()
        except Exception:
            continue
        start = (dt.datetime.combine(dt.date.today(), t, tzinfo=dt.timezone.utc)
                 - dt.timedelta(minutes=cfg.event_quiet_minutes)).time()
        end   = (dt.datetime.combine(dt.date.today(), t, tzinfo=dt.timezone.utc)
                 + dt.timedelta(minutes=cfg.event_quiet_minutes)).time()
        if start <= nowt <= end:
            return True
    return False

def calc_order_amount_okx(ex: ccxt.Exchange, symbol: str, price: float, equity_usdt: float,
                          leverage: float = 10.0) -> float:
    """حساب حجم الصفقة على OKX بناءً على 85% من رأس المال مع رافعة معينة."""
    if price <= 0 or equity_usdt <= 0:
        return 0.0
    target_notional = equity_usdt * 0.85 * leverage
    if target_notional <= 0:
        return 0.0
    raw_amount = target_notional / price
    try:
        market = ex.market(symbol)
    except Exception:
        market = {}
    limits = market.get("limits", {}) if isinstance(market, dict) else {}
    amt_limits = limits.get("amount", {}) if isinstance(limits, dict) else {}
    min_amt = safe_float(amt_limits.get("min"), default=0.0)
    max_amt = safe_float(amt_limits.get("max"), default=0.0)
    if min_amt and raw_amount < min_amt:
        raw_amount = min_amt
    if max_amt and raw_amount > max_amt:
        raw_amount = max_amt
    try:
        precise = ex.amount_to_precision(symbol, raw_amount)
        amount = float(precise)
    except Exception:
        amount = raw_amount
    return max(amount, 0.0)

# =========================
# Bot
# =========================

class Bot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ex = FuturesExchange(cfg)
        self.notifier = Notifier(cfg)
        self.ref_equity = self.ex.get_balance_usdt() or 10000.0
        self.paper = Paper(cfg, self.ref_equity)
        base_universe = self.ex.get_top_symbols(cfg.top_n_symbols)
        self.symbols: List[str] = self.ex.filter_healthy(base_universe)
        self.news = NewsGuard(cfg)
        self.state = self._load_state()
        self.bandit = Bandit(self.state)
        self.last_key: Dict[str, Optional[str]] = {}
        self.last_time: Dict[str, Optional[dt.datetime]] = {}
        self.last_alert_ts: float = 0.0

        # ==== إدارة المخاطر الإضافية (state) ====
        s = self.state.setdefault("risk", {})
        s.setdefault("daily_date", now_utc().date().isoformat())
        s.setdefault("daily_pnl", 0.0)           # صافي اليوم
        s.setdefault("daily_stopped", False)     # تم تفعيل الوقف اليومي؟
        self._save_state()

    def _load_state(self) -> dict:
        if os.path.exists(self.cfg.state_json):
            try:
                with open(self.cfg.state_json,"r",encoding="utf-8") as f: return json.load(f)
            except Exception: return {}
        return {}

    def _save_state(self):
        ensure_dir(self.cfg.state_json)
        with open(self.cfg.state_json,"w",encoding="utf-8") as f: json.dump(self.state, f, ensure_ascii=False, indent=2)

    # ===== أدوات إدارة المخاطر الإضافية =====

    def _daily_rollover_if_needed(self):
        r = self.state.setdefault("risk", {})
        today = now_utc().date().isoformat()
        if r.get("daily_date") != today:
            r["daily_date"] = today
            r["daily_pnl"] = 0.0
            r["daily_stopped"] = False
            self._save_state()

    def _daily_stop_active(self) -> bool:
        r = self.state.setdefault("risk", {})
        return bool(r.get("daily_stopped", False))

    def _check_and_apply_daily_stop(self):
        if not self.cfg.daily_stop_enabled:
            return
        r = self.state.setdefault("risk", {})
        # حد الوقف اليومي بالدولار بناءً على راس المال المرجعي
        limit = -abs(self.cfg.daily_stop_pct) * self.ref_equity
        if float(r.get("daily_pnl", 0.0)) <= limit and not r.get("daily_stopped", False):
            r["daily_stopped"] = True
            self._save_state()
            self.notifier.send(f"🛑 Daily Stop Triggered — صافي اليوم {r['daily_pnl']:.2f} USDT ≤ {limit:.2f}. إيقاف باقي اليوم.")

    # ==========================================

    def can_alert_now(self) -> bool:
        return (time.time() - self.last_alert_ts) >= self.cfg.min_seconds_between_alerts_global

    def _all_generators(self) -> List[Tuple[str, Callable]]:
        return [
            ("TREND", sig_trend),
            ("BO", sig_bo),
            ("MR", sig_mr),
            ("PB", sig_pb),
            ("VWAP-R", sig_vwap_r),
            ("KSQ", sig_ksq),
        ]

    def _ensure_leverage_cross(self, symbol: str, leverage: int = 10):
        try:
            hedged = bool(getattr(self.ex, "is_hedged", False))
        except Exception:
            hedged = False
        setter = getattr(self.ex.x, "set_leverage", None) or getattr(self.ex.x, "setLeverage", None)
        if setter is None:
            self.notifier.send(f"⚠️ Unable to set leverage/cross for {symbol}: method not available")
            return
        try:
            if hedged:
                setter(leverage, symbol, {"mgnMode": "cross", "posSide": "long"})
                setter(leverage, symbol, {"mgnMode": "cross", "posSide": "short"})
            else:
                setter(leverage, symbol, {"mgnMode": "cross"})
        except Exception as e:
            self.notifier.send(f"⚠️ Unable to set leverage/cross for {symbol}: {e}")

    def _place_exit_algo_orders(self, symbol: str, sig: Signal, amount: float) -> bool:
        market = self.ex.x.market(symbol)
        inst_id = market.get("id") if isinstance(market, dict) else None
        if not inst_id:
            return False
        method = getattr(self.ex.x, "private_post_trade_order_algo", None) or getattr(
            self.ex.x, "privatePostTradeOrderAlgo", None
        )
        if method is None:
            return False
        side_close = "sell" if sig.side == "buy" else "buy"
        pos_side = "long" if sig.side == "buy" else "short"
        try:
            hedged = bool(getattr(self.ex, "is_hedged", False))
        except Exception:
            hedged = False
        try:
            sz = self.ex.x.amount_to_precision(symbol, amount)
        except Exception:
            sz = amount
        payload = {
            "instId": inst_id,
            "tdMode": "cross",
            "side": side_close,
            "ordType": "conditional",
            "sz": str(sz),
            "tpTriggerPx": str(sig.tp),
            "tpOrdPx": "-1",
            "slTriggerPx": str(sig.sl),
            "slOrdPx": "-1",
            "tpTriggerPxType": "last",
            "slTriggerPxType": "last",
        }
        if hedged:
            payload["posSide"] = pos_side
        try:
            resp = method(payload)
            if isinstance(resp, dict):
                return bool(resp.get("code") == "0" or resp.get("data"))
            return bool(resp)
        except Exception as e:
            self.notifier.send(f"⚠️ Failed to submit separate TP/SL orders for {symbol}: {e}")
            return False

    def _execute_market_order(self, symbol: str, sig: Signal, amount: float) -> Optional[dict]:
        leverage = 10
        self._ensure_leverage_cross(symbol, leverage)
        params = {"tdMode": "cross"}
        try:
            if bool(getattr(self.ex, "is_hedged", False)):
                params["posSide"] = "long" if sig.side == "buy" else "short"
        except Exception:
            pass
        try:
            order = self.ex.x.create_order(symbol, "market", sig.side, amount, None, params)
        except Exception as e:
            self.notifier.send(f"❌ Order failed for {symbol}: {e}")
            return None
        if self._place_exit_algo_orders(symbol, sig, amount):
            self.notifier.send(f"ℹ️ Placed separate TP/SL orders for {symbol} (algo)")
        else:
            self.notifier.send(f"⚠️ Separate TP/SL orders not confirmed for {symbol}. Manage exits manually.")
        return order

    def _committee(self, symbol: str, row: pd.Series, regime: Regime) -> Optional[Signal]:
        """
        تمت إضافة:
        - Committee Override: لازم حد أدنى من النماذج تتفق على نفس الاتجاه (cfg.committee_min_agree)
        - Confidence Filter: فلترة بالثقة (cfg.min_confidence_accept)
        """
        base = self._all_generators()
        candidates: List[Signal] = []

        # مرشحين أساسيين
        for name, fn in base:
            try:
                s = fn(row, self.cfg)
                if s: candidates.append(s)
            except Exception:
                continue

        # مرشحين مُتحورين (self-evolve)
        mutated_meta: List[Tuple[str, Signal, float]] = []
        if self.cfg.evolve_enabled:
            for _ in range(self.cfg.evolve_mutations_per_round):
                base_name, fn = random.choice(base)
                params = mutate_params(self.cfg)
                s = apply_mutated_signal(row, base_name, params, self.cfg)
                if s:
                    tag = f"{base_name}-MUT"
                    mutated_meta.append((tag, s, self.cfg.evolve_trial_weight))

        if not candidates and not mutated_meta:
            return None

        totals = {"buy":0.0, "sell":0.0}
        details = []
        agree_count = {"buy":0, "sell":0}

        # نحسب أوزان/درجات التصويت + نعد المتفقين من الأساسيين فقط للـ Override
        for s in candidates:
            key = f"{symbol}|{ctx_key(regime)}|{s.model}"
            w = self.bandit.weight(key)
            sc = w * (0.5 + 0.5*s.confidence)
            totals[s.side] += sc
            details.append((s.model, s.side, sc, w, s.confidence))
            agree_count[s.side] += 1

        # المتحورات: تؤثر على السكور فقط (وليس عدّ الاتفاق الأدنى)
        for tag, s, trial_w in mutated_meta:
            key = f"{symbol}|{ctx_key(regime)}|{tag}"
            w = self.bandit.weight(key) * trial_w
            sc = w * (0.5 + 0.5*s.confidence)
            totals[s.side] += sc
            details.append((tag, s.side, sc, w, s.confidence))

        side_pick = "buy" if totals["buy"] > totals["sell"] else "sell"
        same_side = [s for s in candidates if s.side==side_pick] + [s for _,s,_ in mutated_meta if s.side==side_pick]
        if not same_side:
            return None
        best_sig = max(same_side, key=lambda x: x.confidence)

        # Dynamic quorum القديم + تعديل خفيف
        quorum = self.cfg.dyn_quorum_base
        if regime.vol_bucket == "high": quorum += self.cfg.quorum_boost_high_vol
        ctx = ctx_key(regime)
        ctx_means = []
        for d in self.bandit.s.keys():
            if ctx in d:
                bd = self.bandit.s[d]
                ctx_means.append(bd["a"]/(bd["a"]+bd["b"]))
        if ctx_means:
            hit = sum(ctx_means)/len(ctx_means)
            if hit >= 0.55: quorum += self.cfg.quorum_boost_good_hit
            elif hit < 0.45: quorum += self.cfg.quorum_penalty_bad_hit
        quorum = clamp(quorum, 0.45, 0.65)

        denom = totals["buy"] + totals["sell"] + 1e-9
        decision_score = totals[side_pick] / denom
        # لجنة القبول النهائية
        accept_score = (decision_score >= quorum) or (random.random() < self.cfg.exploration_eps)
        accept_conf  = (best_sig.confidence >= self.cfg.min_confidence_accept)
        accept_agree = (agree_count[side_pick] >= self.cfg.committee_min_agree)

        accept = accept_score and accept_conf and accept_agree

        for name, side, sc, w, conf in details:
            self.paper.log_model_vote(self.cfg, symbol, regime, name,
                                      sc, accept and (side==side_pick), w, conf,
                                      notes=f"Q={quorum:.2f}; DS={decision_score:.2f}; Agree[{side}]={agree_count[side]}")

        return best_sig if accept else None

    def run(self):
        self.notifier.send(
            f"[START] Evolving Scalper | OKX Demo | TOP {self.cfg.top_n_symbols}"
            f" | TF {self.cfg.timeframe} | RefEq={self.ref_equity:.2f} USDT | Lev x10"
        )
        while True:
            try:
                self.loop_once()
                time.sleep(2)
            except KeyboardInterrupt:
                self.notifier.send("[EXIT] Stopping…"); break
            except Exception as e:
                self.notifier.send(f"[ERROR main] {e}"); time.sleep(3)

    def loop_once(self):
        # رولات اليوم
        self._daily_rollover_if_needed()

        base_universe = self.ex.get_top_symbols(self.cfg.top_n_symbols)
        self.symbols = self.ex.filter_healthy(base_universe)

        # تحديث إغلاقات الصفقات (حتى أثناء الوقف/التهدئة)
        for symbol in self.symbols:
            try:
                df = self.ex.fetch_ohlcv(symbol, self.cfg.timeframe, limit=self.cfg.lookback)
                d  = compute_indicators(df, self.cfg)
                if len(d) < 2: continue

                last = d.iloc[-1]
                closed = self.paper.update_with_candle(symbol, float(last["high"]), float(last["low"]))
                if closed:
                    reg_row = d.iloc[-2] if len(d)>1 else last
                    regime = classify_regime(reg_row, self.cfg)
                    ctx = ctx_key(regime)
                    self.paper.persist_closed(closed, self.cfg, ctx)

                    # تحديث المخاطر (streak + daily pnl)
                    pnl_sum = 0.0
                    for t in closed:
                        key = f"{t.symbol}|{ctx}|{t.model}"
                        self.bandit.update(key, t.result)
                        pnl_sum += float(t.pnl_usd or 0.0)
                        emoji = "✅" if t.result=="tp" else "❌"
                        hold_s = int((pd.to_datetime(t.exit_time)-pd.to_datetime(t.timestamp)).total_seconds())
                        self.notifier.send(
                            f"📤 Trade Closed {emoji}\n"
                            f"• Pair: {t.symbol} | TF: {t.timeframe}\n"
                            f"• Side: {t.side.upper()} | Model: {t.model}\n"
                            f"• Entry: {t.entry:.4f} → Exit: {t.exit_price:.4f}\n"
                            f"• PnL: {t.pnl_usd:+.2f} USDT | Hold: {hold_s}s"
                        )
                    # حدث صافي اليوم
                    r = self.state.setdefault("risk", {})
                    r["daily_pnl"] = float(r.get("daily_pnl", 0.0)) + pnl_sum
                    self._save_state()
                    # وقّف يومي لو تعدى الحد
                    self._check_and_apply_daily_stop()
                    self.bandit.decay_weights(0.998)
                    self._save_state()
            except Exception:
                continue

        # لو في صفقات مفتوحة — نكتفي بتتبع الإغلاق فقط
        if len(self.paper.open) > 0:
            return

        # لا تدخل صفقات جديدة لو في وقف يومي
        if self._daily_stop_active():
            return

        # هدوء أحداث أو ثروتل
        if in_quiet_window(self.cfg): return
        if not self.can_alert_now(): return

        # البحث عن إشارة جديدة
        for symbol in self.symbols:
            try:
                asset = symbol.split("/")[0]
                if self.cfg.news_enabled and self.news.too_hot(asset):
                    continue
                if self.cfg.funding_filter:
                    fr = self.ex.fetch_funding_rate(symbol)
                    if fr is not None and abs(fr) > self.cfg.max_abs_funding:
                        continue

                df = self.ex.fetch_ohlcv(symbol, self.cfg.timeframe, limit=self.cfg.lookback)
                d  = compute_indicators(df, self.cfg)
                if len(d) < 3: continue

                row = d.iloc[-2]
                price = float(row["close"])
                regime = classify_regime(row, self.cfg)

                sig = self._committee(symbol, row, regime)
                if not sig: continue

                # مانع تكرار نفس الإشارة
                key = f"{symbol}:{self.cfg.timeframe}:{sig.model}:{sig.side}"
                if self.last_key.get(symbol) == key and self.last_time.get(symbol):
                    if (now_utc() - self.last_time[symbol]).total_seconds()/60.0 < self.cfg.min_minutes_between_same_signal:
                        continue

                risk = abs(price - sig.sl); reward = abs(sig.tp - price)
                rr = round(reward / risk, 2) if risk > 0 else None

                equity_now = self.ex.get_balance_usdt() or self.ref_equity
                if equity_now and equity_now > 0:
                    self.ref_equity = equity_now
                sizing_equity = equity_now if equity_now and equity_now > 0 else self.ref_equity
                order_amount = calc_order_amount_okx(self.ex.x, symbol, price, sizing_equity, leverage=10.0)
                if order_amount <= 0:
                    self.notifier.send(f"⚠️ Skipping {symbol}: unable to size order (equity≈{sizing_equity:.2f} USDT)")
                    continue

                order = self._execute_market_order(symbol, sig, order_amount)
                if not order:
                    continue

                order_id = None
                if isinstance(order, dict):
                    order_id = order.get("id") or order.get("orderId")
                    if not order_id:
                        info = order.get("info", {})
                        if isinstance(info, dict):
                            order_id = info.get("ordId") or info.get("orderId")

                executed_notional = order_amount * price
                exec_msg = (
                    f"✅ EXECUTED {sig.side.upper()} {symbol} @ MARKET\n"
                    f"• Model: {sig.model} | Conf: {sig.confidence:.2f}\n"
                    f"• Qty: {order_amount:.6f} (~{executed_notional:.2f} USDT)\n"
                    f"• TP: {sig.tp:.4f} | SL: {sig.sl:.4f} | R:R = {rr if rr is not None else 'n/a'}\n"
                    f"• Reason: {sig.reason}\n"
                    f"• OrderID: {order_id or 'n/a'}"
                )
                self.notifier.send(exec_msg)
                self.last_alert_ts = time.time()

                self.paper.log_signal(symbol, row, sig, order_amount, executed_notional, rr, self.cfg, regime)
                t = self.paper.open_virtual(symbol, price, sig, self.cfg)
                self.paper.ml_snapshot(t.id, symbol, row, regime)

                self.last_key[symbol] = key
                self.last_time[symbol] = now_utc()
                self._save_state()
                break

            except Exception:
                continue

# =========================
# CLI
# =========================

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Evolving Committee Scalper — OKX Demo Auto Trader (No OpenAI)")
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--quiet", nargs="*", default=None, help="UTC HH:MM times to avoid (e.g., 12:30 18:00)")
    p.add_argument("--top", type=int, default=None, help="Top N USDT perpetuals to scan (override config)")
    p.add_argument("--minconf", type=float, default=None, help="Min confidence to accept (override config)")
    p.add_argument("--agree", type=int, default=None, help="Min base models agreeing (override config)")
    p.add_argument("--dailystop", type=float, default=None, help="Daily stop pct (e.g., 0.02 for 2%)")
    args = p.parse_args()
    cfg = Config()
    cfg.timeframe = args.timeframe
    # اسمح بتجاوز الإعدادات من سطر الأوامر
    if args.top is not None: cfg.top_n_symbols = int(args.top)
    if args.quiet is not None: cfg.quiet_windows_utc = tuple(args.quiet)
    if args.minconf is not None: cfg.min_confidence_accept = float(args.minconf)
    if args.agree is not None: cfg.committee_min_agree = int(args.agree)
    if args.dailystop is not None: cfg.daily_stop_pct = float(args.dailystop)
    ensure_dir(cfg.logs_dir)
    ensure_dir(cfg.signals_csv); ensure_dir(cfg.trades_csv); ensure_dir(cfg.ml_csv); ensure_dir(cfg.models_csv); ensure_dir(cfg.state_json)
    return cfg

def main():
    cfg = parse_args()
    cfg_view = asdict(cfg)
    for secret_key in ("telegram_token", "telegram_chat_id"):
        if cfg_view.get(secret_key):
            cfg_view[secret_key] = "***"
    print("Config:\n", json.dumps(cfg_view, indent=2, default=str))
    bot = Bot(cfg)
    bot.run()

if __name__ == "__main__":
    main()
