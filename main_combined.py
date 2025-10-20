# main_combined.py
"""
Pro Trader AI - Combined Agent (FULL)
- FastAPI server
- Endpoints:
  GET  /health
  GET  /pro_signal?pair=BTCUSDT&tf_main=1h&tf_entry=15m
  GET  /scalp_signal?pair=BTCUSDT&tf=3m
  GET  /analyze_live?exchange=binance&symbol=BTCUSDT&interval=1h
  POST /analyze_chart  (multipart file=@chart.png ; form: pair,timeframe,auto_backtest)
  POST /analyze      (json: {pair, timeframe}) -> run data-based analyze and auto-log
  POST /analyze_csv  (multipart file=@ohlc.csv)
  GET  /openapi.json
- Auto-sends signal JSON to BACKTEST_URL if env BACKTEST_URL is set.
"""

import os, io, math, json, requests
from typing import Optional, List
from datetime import datetime
from fastapi import FastAPI, Query, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import ta
from PIL import Image
import cv2

# Try import pytesseract; if missing, handle gracefully
try:
    import pytesseract
    _HAS_TESSERACT = True
except Exception:
    pytesseract = None
    _HAS_TESSERACT = False

app = FastAPI(title="Pro Trader AI - Combined",
              description="Hybrid SMC + Price Action + Alchemist. API + Chart image analyzer + Backtester integration",
              version="1.0")

BACKTEST_URL = os.environ.get("BACKTEST_URL")  # e.g. https://.../evaluate_signal

# ---------- Utilities & Indicators ----------
def fetch_ohlc_binance(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    r = requests.get(url, params={"symbol": symbol.upper(), "interval": interval, "limit": limit}, timeout=10)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume","close_time","qav","num_trades","tb_base","tb_quote","ignore"
    ])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["open_time","open","high","low","close","volume"]]

def ema(series: pd.Series, n: int):
    return ta.trend.EMAIndicator(series, window=n).ema_indicator()

def rsi(series: pd.Series, n: int=14):
    return ta.momentum.RSIIndicator(series, window=n).rsi()

def atr(df: pd.DataFrame, n: int=14):
    return ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=n).average_true_range()

def detect_sr(df: pd.DataFrame, lookback:int=120):
    recent_h = df['high'].tail(lookback).max()
    recent_l = df['low'].tail(lookback).min()
    return float(recent_h), float(recent_l)

def fib_levels(low: float, high: float):
    diff = high - low
    return {
        "high": high,
        "fib_618": high - diff*0.618,
        "fib_5": high - diff*0.5,
        "fib_382": high - diff*0.382,
        "low": low
    }

def breakout_of_structure(df: pd.DataFrame, window:int=20):
    if df.shape[0] < window+2: return None
    high_sw = df['high'].rolling(window).max().iloc[-2]
    low_sw = df['low'].rolling(window).min().iloc[-2]
    last = df['close'].iloc[-1]
    prev = df['close'].iloc[-2]
    if prev <= high_sw and last > high_sw: return "BOS_UP"
    if prev >= low_sw and last < low_sw: return "BOS_DOWN"
    return None

def score_confidence(parts: List[float]) -> float:
    if not parts: return 0.0
    vals = [max(0.0, min(1.0, float(v))) for v in parts]
    return float(sum(vals)/len(vals))

def pretty_reason(parts: List[str]) -> str:
    return " · ".join([p for p in parts if p])

def fmt(x):
    try:
        if abs(x) >= 1000: return f"{x:,.0f}"
        if abs(x) >= 1: return f"{x:,.2f}"
        return f"{x:.6f}"
    except:
        return str(x)

# ---------- Core hybrid analyzer (OHLC dataframe) ----------
def hybrid_analyze(df: pd.DataFrame, pair:Optional[str]=None, timeframe:Optional[str]=None) -> dict:
    df = df.copy().dropna().reset_index(drop=True)
    if df.shape[0] < 12:
        return {"error":"not_enough_data", "message":"Need >=12 candles"}

    df['ema20'] = ema(df['close'],20)
    df['ema50'] = ema(df['close'],50)
    df['rsi14'] = rsi(df['close'],14)
    df['atr14'] = atr(df,14)
    last = df.iloc[-1]
    price = float(last['close'])
    ema20 = float(last['ema20']) if not np.isnan(last['ema20']) else None
    ema50 = float(last['ema50']) if not np.isnan(last['ema50']) else None
    rsi_now = float(last['rsi14']) if not np.isnan(last['rsi14']) else None
    atr_now = float(last['atr14']) if not np.isnan(last['atr14']) else price*0.001

    recent_high, recent_low = detect_sr(df, lookback=120)
    bos = breakout_of_structure(df, window=20)
    swing_high = df['high'].tail(80).max()
    swing_low  = df['low'].tail(80).min()
    fib = fib_levels(swing_low, swing_high)

    reasons=[]; conf=[]
    trend = None
    if ema20 and ema50:
        trend = "bullish" if ema20 > ema50 else "bearish"

    if bos == "BOS_UP" or (trend == "bullish" and price > ema20):
        entry = price
        prefer_zone = min(max(fib['fib_618'], recent_low), fib['fib_382'])
        sl = recent_low - atr_now*0.6
        rr = entry - sl if entry>sl else price*0.01
        tp1 = entry + rr*1.5
        tp2 = entry + rr*2.5
        reasons.append("Bias LONG — BOS/EMA alignment.")
        reasons.append(f"Support ~{fmt(recent_low)}; fib confluence ~{fmt(prefer_zone)}.")
        if bos=="BOS_UP": reasons.append("Break of Structure UP detected.")
        conf.append(0.9 if trend=="bullish" else 0.6)
        conf.append(0.9 if price >= prefer_zone else 0.65)
        conf.append(1.0 if (rsi_now and 30 < rsi_now < 75) else 0.5)
        signal="LONG"
    elif bos == "BOS_DOWN" or (trend == "bearish" and price < ema20):
        entry = price
        prefer_zone = max(min(fib['fib_618'], recent_high), fib['fib_382'])
        sl = recent_high + atr_now*0.6
        rr = sl - entry if sl>entry else price*0.01
        tp1 = entry - rr*1.5
        tp2 = entry - rr*2.5
        reasons.append("Bias SHORT — BOS/EMA alignment.")
        reasons.append(f"Resistance ~{fmt(recent_high)}; fib confluence ~{fmt(prefer_zone)}.")
        if bos=="BOS_DOWN": reasons.append("Break of Structure DOWN detected.")
        conf.append(0.9 if trend=="bearish" else 0.6)
        conf.append(0.9 if price <= prefer_zone else 0.65)
        conf.append(1.0 if (rsi_now and 25 < rsi_now < 70) else 0.5)
        signal="SHORT"
    else:
        entry = price
        sl = recent_low * 0.995
        tp1 = entry + (entry-sl) * 1.2
        tp2 = entry + (entry-sl) * 2.0
        reasons.append("No clear high-probability structure — WAIT or confirm higher TF.")
        conf.append(0.25)
        signal="WAIT"

    confidence = score_confidence(conf)
    reasoning = pretty_reason(reasons)

    return {
        "pair": pair or "",
        "timeframe": timeframe or "",
        "signal_type": signal,
        "entry": round(entry,8),
        "tp1": round(tp1,8),
        "tp2": round(tp2,8),
        "sl": round(sl,8),
        "confidence": round(confidence,3),
        "reasoning": reasoning
    }

# ---------- Scalp logic (short TF) ----------
def scalp_engine(df: pd.DataFrame, pair:Optional[str]=None, tf:Optional[str]=None) -> dict:
    if df.shape[0] < 30:
        return {"error":"not_enough_bars"}
    df['ema8'] = ema(df['close'],8)
    df['ema21'] = ema(df['close'],21)
    df['rsi14'] = rsi(df['close'],14)
    df['atr14'] = atr(df,14)
    last = df.iloc[-1]
    price = float(last['close'])
    atr_now = float(last['atr14']) if not np.isnan(last['atr14']) else price*0.001
    vol_mean = df['volume'].tail(40).mean() if df.shape[0]>=40 else df['volume'].mean()
    vol_spike = float(last['volume']) > (vol_mean*1.8 if vol_mean>0 else False)

    if float(last['ema8']) > float(last['ema21']) and vol_spike and 35 < float(last['rsi14']) < 75:
        entry = price
        sl = entry - atr_now*0.6
        tp1 = entry + atr_now*0.8
        tp2 = entry + atr_now*1.4
        reason = "Scalp LONG: EMA8>EMA21, volume spike, RSI ok"
        conf = 0.9
        signal="LONG"
    elif float(last['ema8']) < float(last['ema21']) and vol_spike and 25 < float(last['rsi14']) < 65:
        entry = price
        sl = entry + atr_now*0.6
        tp1 = entry - atr_now*0.8
        tp2 = entry - atr_now*1.4
        reason = "Scalp SHORT: EMA8<EMA21, volume spike, RSI ok"
        conf = 0.9
        signal="SHORT"
    else:
        entry = price
        sl = price*0.998
        tp1 = price*1.002
        tp2 = price*1.004
        reason = "No clean scalp conditions"
        conf = 0.2
        signal="WAIT"

    return {
        "pair": pair or "",
        "timeframe": tf or "",
        "signal_type": signal,
        "entry": round(entry,8),
        "tp1": round(tp1,8),
        "tp2": round(tp2,8),
        "sl": round(sl,8),
        "confidence": round(conf,3),
        "reasoning": reason
    }

# ---------- Image digitizer helpers ----------
def ocr_y_axis_prices(img_cv):
    if not _HAS_TESSERACT:
        return {}
    h,w = img_cv.shape[:2]
    left_w = max(60, int(w*0.12))
    crop = img_cv[:, :left_w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    cfg = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.,-'
    data = pytesseract.image_to_data(gray, config=cfg, output_type=pytesseract.Output.DICT)
    ticks = {}
    import re
    for i, txt in enumerate(data['text']):
        if not txt or any(ch.isalpha() for ch in txt): continue
        m = re.search(r'[-+]?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?', txt)
        if m:
            raw = m.group(0).replace(',','.') 
            try:
                val = float(raw)
            except:
                continue
            y = int(data['top'][i] + data['height'][i]/2)
            ticks[y] = val
    return ticks

def detect_candles_from_plot(img_cv, y_map, max_bars=200):
    h,w = img_cv.shape[:2]
    left_w = max(60, int(w*0.12))
    right_w = int(w*0.02)
    top_margin = int(h*0.06)
    bottom_margin = int(h*0.04)
    plot = img_cv[top_margin:h-bottom_margin, left_w:w-right_w].copy()
    ph,pw = plot.shape[:2]
    gray_plot = cv2.cvtColor(plot, cv2.COLOR_BGR2GRAY)
    col_mean = gray_plot.mean(axis=0)
    thr = np.percentile(col_mean, 60)
    peaks = np.where(col_mean < thr)[0]
    centers = []
    i=0
    while i < len(peaks):
        j=i
        while j+1 < len(peaks) and peaks[j+1]==peaks[j]+1:
            j+=1
        centers.append((peaks[i]+peaks[j])//2)
        i=j+1
    centers = centers[-max_bars:]
    rows=[]
    if not y_map or len(y_map) < 2:
        ys_sorted = np.array([0, ph-1])
        prices_sorted = np.array([1.0, 0.9])
        def px_to_price(py):
            return float(1.0 - (py / ph) * 0.1)
    else:
        ys_sorted = np.array(sorted(y_map.keys()))
        prices_sorted = np.array([y_map[y] for y in ys_sorted])
        def px_to_price(py):
            orig_py = py + top_margin
            return float(np.interp(orig_py, ys_sorted, prices_sorted))

    for cx in centers:
        col = plot[:, cx]
        col_g = cv2.cvtColor(col.reshape(-1,1,3), cv2.COLOR_BGR2GRAY).flatten()
        darks = np.where(col_g < np.percentile(col_g,60))[0]
        if len(darks) < 3: continue
        top_px = darks.min()
        bot_px = darks.max()
        high_px = top_px
        low_px = bot_px
        open_px = top_px
        close_px = bot_px
        try:
            high_p = px_to_price(high_px)
            low_p = px_to_price(low_px)
            open_p = px_to_price(open_px)
            close_p = px_to_price(close_px)
        except Exception:
            continue
        rows.append({'open':open_p,'high':high_p,'low':low_p,'close':close_p})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_index().reset_index(drop=True)
    return df

# ---------- Backtester POST ----------
def post_to_backtester(payload: dict):
    if not BACKTEST_URL:
        return {"error":"BACKTEST_URL_not_configured"}
    try:
        r = requests.post(BACKTEST_URL, json=payload, timeout=12)
        try:
            return r.json()
        except:
            return {"status_code": r.status_code, "text": r.text}
    except Exception as e:
        return {"error":"backtester_unreachable", "detail": str(e)}

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status":"ok","service":"Pro Trader AI - Combined"}

@app.get("/pro_signal")
def pro_signal(pair: str = Query(...), tf_main: str = Query("1h"), tf_entry: str = Query("15m"), limit:int = Query(300), auto_log: bool = Query(False)):
    try:
        df_entry = fetch_ohlc_binance(pair, tf_entry, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"fetch_failed: {e}")
    res = hybrid_analyze(df_entry, pair=pair, timeframe=tf_entry)
    # add context main trend
    try:
        df_main = fetch_ohlc_binance(pair, tf_main, limit=200)
        ema20_main = float(ema(df_main['close'],20).iloc[-1])
        ema50_main = float(ema(df_main['close'],50).iloc[-1])
        res['context_main_trend'] = "bullish" if ema20_main>ema50_main else "bearish"
    except:
        pass
    # auto-log to backtester if requested
    if auto_log and 'error' not in res:
        payload = {
            "pair": res["pair"],
            "timeframe": res["timeframe"],
            "side": res["signal_type"],
            "entry": res["entry"],
            "tp1": res["tp1"],
            "tp2": res["tp2"],
            "sl": res["sl"],
            "confidence": res["confidence"],
            "reason": res["reasoning"]
        }
        res['backtest'] = post_to_backtester(payload)
    return JSONResponse(res)

@app.get("/scalp_signal")
def scalp_signal(pair: str = Query(...), tf: str = Query("3m"), limit:int = Query(300), auto_log: bool = Query(False)):
    try:
        df = fetch_ohlc_binance(pair, tf, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"fetch_failed: {e}")
    res = scalp_engine(df, pair=pair, tf=tf)
    if auto_log and 'error' not in res:
        payload = {
            "pair": res["pair"],
            "timeframe": res["timeframe"],
            "side": res["signal_type"],
            "entry": res["entry"],
            "tp1": res["tp1"],
            "tp2": res["tp2"],
            "sl": res["sl"],
            "confidence": res["confidence"],
            "reason": res["reasoning"]
        }
        res['backtest'] = post_to_backtester(payload)
    return JSONResponse(res)

@app.get("/analyze_live")
def analyze_live(exchange: str = Query(...), symbol: str = Query(...), interval: str = Query("1h"), limit:int = Query(200), auto_log: bool = Query(False)):
    if exchange.lower() != "binance":
        raise HTTPException(status_code=400, detail="exchange_not_supported; use binance for live")
    try:
        df = fetch_ohlc_binance(symbol, interval, limit=limit)
        res = hybrid_analyze(df, pair=symbol, timeframe=interval)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if auto_log and 'error' not in res:
        payload = {
            "pair": res["pair"],
            "timeframe": res["timeframe"],
            "side": res["signal_type"],
            "entry": res["entry"],
            "tp1": res["tp1"],
            "tp2": res["tp2"],
            "sl": res["sl"],
            "confidence": res["confidence"],
            "reason": res["reasoning"]
        }
        res['backtest'] = post_to_backtester(payload)
    return JSONResponse(res)

@app.post("/analyze")
def analyze_endpoint(payload: dict):
    pair = payload.get("pair")
    tf_entry = payload.get("timeframe","15m")
    auto_log = bool(payload.get("auto_log", False))
    if not pair:
        raise HTTPException(status_code=400, detail="pair required")
    try:
        df_entry = fetch_ohlc_binance(pair, tf_entry, limit=300)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    res = hybrid_analyze(df_entry, pair=pair, timeframe=tf_entry)
    if auto_log and 'error' not in res:
        payload_bt = {
            "pair": res["pair"],
            "timeframe": res["timeframe"],
            "side": res["signal_type"],
            "entry": res["entry"],
            "tp1": res["tp1"],
            "tp2": res["tp2"],
            "sl": res["sl"],
            "confidence": res["confidence"],
            "reason": res["reasoning"]
        }
        res['backtest'] = post_to_backtester(payload_bt)
    return JSONResponse(res)

@app.post("/analyze_chart")
def analyze_chart(file: UploadFile = File(...), pair: Optional[str] = Form(None), timeframe: Optional[str] = Form(None), auto_backtest: Optional[str] = Form("true")):
    auto_backtest_flag = auto_backtest.lower() != "false"
    try:
        contents = file.file.read()
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
        img_cv = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid_image: {e}")

    # OCR y-axis (best-effort)
    y_map = {}
    if _HAS_TESSERACT:
        try:
            y_map = ocr_y_axis_prices(img_cv)
        except Exception:
            y_map = {}
    else:
        y_map = {}

    df_ohlc = detect_candles_from_plot(img_cv, y_map, max_bars=200)
    if df_ohlc.empty:
        raise HTTPException(status_code=400, detail="digitize_failed - upload CSV for most reliable results")

    # Ensure numeric columns exist
    for col in ['open','high','low','close']:
        df_ohlc[col] = pd.to_numeric(df_ohlc[col], errors='coerce')

    res = hybrid_analyze(df_ohlc, pair=pair or "IMG", timeframe=timeframe or "img")

    if auto_backtest_flag and 'error' not in res:
        payload_bt = {
            "pair": res["pair"],
            "timeframe": res["timeframe"],
            "side": res["signal_type"],
            "entry": res["entry"],
            "tp1": res.get("tp1"),
            "tp2": res.get("tp2"),
            "tp3": res.get("tp3"),
            "sl": res["sl"],
            "confidence": res["confidence"],
            "reason": res["reasoning"],
            "spread_pct": 0.02,
            "slippage_pct": 0.03
        }
        res['backtest'] = post_to_backtester(payload_bt)
    res['bars_used'] = int(df_ohlc.shape[0])
    return JSONResponse(res)

@app.post("/analyze_csv")
def analyze_csv(file: UploadFile = File(...), pair: Optional[str] = Form(None), timeframe: Optional[str] = Form(None), auto_backtest: Optional[str] = Form("true")):
    auto_flag = auto_backtest.lower() != "false"
    try:
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid_csv: {e}")

    # normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]
    # find ohlc columns
    def find_col(k):
        for c in df.columns:
            if k in c: return c
        return None
    o = find_col('open'); h = find_col('high'); l = find_col('low'); ccol = find_col('close')
    if not all([o,h,l,ccol]):
        raise HTTPException(status_code=400, detail="missing_columns: need open, high, low, close")
    df2 = df[[o,h,l,ccol]].rename(columns={o:'open',h:'high',l:'low',ccol:'close'})
    for col in ['open','high','low','close']:
        df2[col] = pd.to_numeric(df2[col], errors='coerce')
    df2 = df2.dropna().reset_index(drop=True)
    res = hybrid_analyze(df2, pair=pair or "CSV", timeframe=timeframe or "csv")
    if auto_flag and 'error' not in res:
        payload_bt = {
            "pair": res["pair"],
            "timeframe": res["timeframe"],
            "side": res["signal_type"],
            "entry": res["entry"],
            "tp1": res.get("tp1"),
            "tp2": res.get("tp2"),
            "sl": res["sl"],
            "confidence": res["confidence"],
            "reason": res["reasoning"]
        }
        res['backtest'] = post_to_backtester(payload_bt)
    return JSONResponse(res)

@app.get("/openapi.json")
def openapi():
    return app.openapi()

# Run with: uvicorn main_combined:app --host 0.0.0.0 --port $PORT
