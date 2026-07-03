#!/usr/bin/env python3
"""
Weekly 52-Week High/Low Screener - Enhanced Edition
-----------------------------------------------------
Screens S&P 500, Russell 2000, and Nasdaq stocks for new 52-week highs/lows.
Enriched with: sector grouping, volume spike, RSI, % from 52-week mark.
Saves an HTML report + CSVs and optionally sends an email brief.

Dependencies:
    pip install yfinance pandas requests lxml
"""

import os
import io
import sys
import smtplib
import datetime as dt
from email.message import EmailMessage
from collections import Counter

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yfinance as yf

# =============================== CONFIG ===============================
INCLUDE_SP500       = True
INCLUDE_RUSSELL2000 = True
INCLUDE_NASDAQ      = True

RECENT_DAYS = 5      # trading days that count as "this week"
WINDOW      = 252    # trailing days for 52-week calculation
BATCH_SIZE  = 200    # tickers per yfinance download batch
AUTO_ADJUST = True   # split/dividend adjusted prices

VOL_WINDOW  = 20     # days for average volume baseline
RSI_WINDOW  = 14     # RSI lookback

# Email (optional) -- set as environment variables, never hardcode
SMTP_HOST  = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT  = int(os.environ.get("SMTP_PORT", "587"))
EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASS = os.environ.get("EMAIL_PASS")
EMAIL_TO   = os.environ.get("EMAIL_TO", EMAIL_USER)
# =====================================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_SP500_CACHE = os.path.join(_SCRIPT_DIR, "sp500_cached.csv")


def make_session(retries=3, backoff=1.5):
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(HEADERS)
    return session


# ----------------------- TICKER + SECTOR DATA ------------------------

def get_sp500_with_sectors():
    """
    Returns (symbols_list, sector_map_dict).
    Tries GitHub CSV first (most reliable from Actions), then Wikipedia, then cache.
    """
    session = make_session()
    github_url = (
        "https://raw.githubusercontent.com/datasets/"
        "s-and-p-500-companies/main/data/constituents.csv"
    )

    for label, fetch in [
        ("GitHub CSV", lambda: session.get(github_url, timeout=30)),
        ("Wikipedia",  lambda: session.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", timeout=30)),
    ]:
        try:
            r = fetch()
            r.raise_for_status()
            if "wikipedia" in r.url:
                df = pd.read_html(io.StringIO(r.text))[0]
                df = df.rename(columns={"Symbol": "Symbol", "GICS Sector": "GICS Sector"})
            else:
                df = pd.read_csv(io.StringIO(r.text))

            df["Symbol"] = df["Symbol"].astype(str).str.replace(".", "-", regex=False)
            sector_col = "GICS Sector" if "GICS Sector" in df.columns else None
            sector_map = (
                dict(zip(df["Symbol"], df[sector_col]))
                if sector_col else {}
            )
            syms = df["Symbol"].tolist()

            # update cache
            cache_df = df[["Symbol"]].copy()
            if sector_col:
                cache_df["Sector"] = df[sector_col]
            cache_df.to_csv(_SP500_CACHE, index=False)

            print(f"  Fetched {len(syms)} S&P 500 tickers from {label}.")
            return syms, sector_map
        except Exception as e:
            print(f"  {label} fetch failed ({e}).")

    # Cache fallback
    if os.path.exists(_SP500_CACHE):
        df = pd.read_csv(_SP500_CACHE)
        df["Symbol"] = df["Symbol"].astype(str)
        sector_map = (
            dict(zip(df["Symbol"], df["Sector"]))
            if "Sector" in df.columns else {}
        )
        print(f"  Loaded {len(df)} S&P 500 tickers from cache.")
        return df["Symbol"].tolist(), sector_map

    raise RuntimeError("All S&P 500 sources failed and no cache found.")


def get_nasdaq_tickers():
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    try:
        r = make_session().get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), sep="|")
        df = df[df.get("Test Issue", "N") == "N"]
        syms = [s for s in df["Symbol"].dropna().astype(str) if s.isalpha()]
        print(f"  Fetched {len(syms)} Nasdaq tickers.")
        return syms
    except Exception as e:
        print(f"  Nasdaq fetch failed ({e}). Skipping.")
        return []


def get_russell2000_tickers():
    url = (
        "https://www.ishares.com/us/products/239710/"
        "ishares-russell-2000-etf/1467271812596.ajax"
        "?fileType=csv&fileName=IWM_holdings&dataType=fund"
    )
    try:
        r = make_session().get(url, timeout=30)
        r.raise_for_status()
        lines = r.text.splitlines()
        start = next(
            i for i, ln in enumerate(lines)
            if ln.replace('"', "").startswith("Ticker")
        )
        df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
        if "Asset Class" in df.columns:
            df = df[df["Asset Class"].astype(str).str.contains("Equity", na=False)]
        syms = []
        for s in df["Ticker"].dropna().astype(str):
            s = s.strip().replace(".", "-")
            if s and s not in ("-", "USD") and any(c.isalpha() for c in s):
                syms.append(s)
        print(f"  Fetched {len(syms)} Russell 2000 tickers.")
        return syms
    except Exception as e:
        print(f"  Russell 2000 fetch failed ({e}). Skipping.")
        return []


# ------------------------- CALCULATIONS ------------------------------

def calc_rsi(close_series, window=RSI_WINDOW):
    delta = close_series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).iloc[-1]


# ---------------------------- SCREEN ---------------------------------

def screen(tickers, sector_map):
    highs, lows = [], []
    total = len(tickers)

    for i in range(0, total, BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        print(f"  Downloading {i + 1}-{min(i + BATCH_SIZE, total)} of {total} ...")
        try:
            data = yf.download(
                batch, period="380d", interval="1d",
                group_by="ticker", auto_adjust=AUTO_ADJUST,
                threads=True, progress=False,
            )
        except Exception as e:
            print(f"  Batch failed: {e}")
            continue

        for t in batch:
            try:
                df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                df = df.dropna(subset=["High", "Low", "Close"])
                if len(df) < max(WINDOW, VOL_WINDOW, RSI_WINDOW) // 2:
                    continue

                roll_high = df["High"].rolling(WINDOW, min_periods=100).max()
                roll_low  = df["Low"].rolling(WINDOW, min_periods=100).min()

                last_close  = round(float(df["Close"].iloc[-1]), 2)
                w52_high    = round(float(roll_high.iloc[-1]), 2)
                w52_low     = round(float(roll_low.iloc[-1]), 2)

                # RSI
                try:
                    rsi = round(float(calc_rsi(df["Close"])), 1)
                except Exception:
                    rsi = None

                # Volume spike: breakout day volume vs 20-day average
                avg_vol = df["Volume"].rolling(VOL_WINDOW).mean()
                last_vol = float(df["Volume"].iloc[-1])
                avg_vol_val = float(avg_vol.iloc[-1]) if not pd.isna(avg_vol.iloc[-1]) else None
                vol_ratio = (
                    round(last_vol / avg_vol_val, 1)
                    if avg_vol_val and avg_vol_val > 0 else None
                )

                sector = sector_map.get(t, "Other")

                # Check 52-week high hit
                recent_hi = df["High"].tail(RECENT_DAYS)
                rh = roll_high.tail(RECENT_DAYS)
                hi_mask = (recent_hi >= rh).values
                if hi_mask.any():
                    pct_from = round((last_close / w52_high - 1) * 100, 2)
                    highs.append({
                        "Ticker":         t,
                        "Sector":         sector,
                        "Last Close":     last_close,
                        "52W High":       w52_high,
                        "% From High":    pct_from,
                        "RSI":            rsi,
                        "Vol vs Avg":     vol_ratio,
                        "Date Hit":       recent_hi.index[hi_mask][-1].date().isoformat(),
                    })

                # Check 52-week low hit
                recent_lo = df["Low"].tail(RECENT_DAYS)
                rl = roll_low.tail(RECENT_DAYS)
                lo_mask = (recent_lo <= rl).values
                if lo_mask.any():
                    pct_from = round((last_close / w52_low - 1) * 100, 2)
                    lows.append({
                        "Ticker":         t,
                        "Sector":         sector,
                        "Last Close":     last_close,
                        "52W Low":        w52_low,
                        "% From Low":     pct_from,
                        "RSI":            rsi,
                        "Vol vs Avg":     vol_ratio,
                        "Date Hit":       recent_lo.index[lo_mask][-1].date().isoformat(),
                    })

            except Exception:
                continue

    h = pd.DataFrame(highs).sort_values(["Sector", "Ticker"]) if highs else pd.DataFrame()
    l = pd.DataFrame(lows).sort_values(["Sector", "Ticker"])  if lows  else pd.DataFrame()
    return h, l


# ---------------------------- REPORT ---------------------------------

STYLE = """
<style>
body { font-family: Arial, Helvetica, sans-serif; color: #1a1a1a; max-width: 1100px; margin: 0 auto; padding: 20px; }
h1   { font-size: 22px; border-bottom: 2px solid #0066cc; padding-bottom: 8px; }
h2   { font-size: 17px; margin-top: 28px; color: #0066cc; }
h3   { font-size: 14px; margin-top: 16px; margin-bottom: 4px; color: #444; }
.summary-box { background: #f8f9fa; border-left: 4px solid #0066cc; padding: 16px 20px; margin: 16px 0; border-radius: 0 4px 4px 0; }
.stat { display: inline-block; margin-right: 32px; }
.stat .num  { font-size: 28px; font-weight: bold; color: #0066cc; }
.stat .lbl  { font-size: 12px; color: #666; display: block; }
.highs .num { color: #1a7a1a; }
.lows  .num { color: #cc2200; }
.sectors    { display: flex; gap: 32px; flex-wrap: wrap; margin: 12px 0; }
.sec-block  { min-width: 180px; }
.sec-block ul { margin: 4px 0; padding-left: 16px; font-size: 13px; }
.notable    { margin: 12px 0; font-size: 13px; }
.tag        { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: bold; margin-right: 4px; }
.tag.ob     { background: #ffe0e0; color: #cc2200; }
.tag.os     { background: #e0ffe0; color: #1a7a1a; }
.tag.vol    { background: #fff3cd; color: #856404; }
.t   { border-collapse: collapse; font-size: 12px; width: 100%; margin-top: 8px; }
.t td, .t th { border: 1px solid #ddd; padding: 5px 9px; text-align: left; }
.t th { background: #f4f4f4; font-size: 11px; }
.t tr:hover { background: #f9f9f9; }
</style>
"""


def sector_counts(df, col="Sector"):
    if df.empty or col not in df.columns:
        return {}
    return dict(Counter(df[col]).most_common())


def notable_flags(df, label):
    """Return HTML tags for notable stocks: high volume, overbought/oversold RSI."""
    flags = []
    if df.empty:
        return ""
    if "Vol vs Avg" in df.columns:
        vols = df[df["Vol vs Avg"].notna() & (df["Vol vs Avg"] >= 2.0)]
        for _, r in vols.iterrows():
            flags.append(
                f'<span class="tag vol">{r["Ticker"]} vol {r["Vol vs Avg"]}x</span>'
            )
    if "RSI" in df.columns:
        if label == "highs":
            ob = df[df["RSI"].notna() & (df["RSI"] >= 70)]
            for _, r in ob.iterrows():
                flags.append(
                    f'<span class="tag ob">{r["Ticker"]} RSI {r["RSI"]}</span>'
                )
        else:
            os_ = df[df["RSI"].notna() & (df["RSI"] <= 30)]
            for _, r in os_.iterrows():
                flags.append(
                    f'<span class="tag os">{r["Ticker"]} RSI {r["RSI"]}</span>'
                )
    return "".join(flags)


def build_html(highs, lows):
    today = dt.date.today().isoformat()

    # --- summary section ---
    h_sectors = sector_counts(highs)
    l_sectors = sector_counts(lows)

    hi_flags  = notable_flags(highs, "highs")
    lo_flags  = notable_flags(lows,  "lows")

    def sector_ul(counts):
        if not counts:
            return "<em>n/a</em>"
        items = "".join(
            f"<li><b>{s}</b>: {n}</li>" for s, n in list(counts.items())[:8]
        )
        return f"<ul>{items}</ul>"

    overbought_count  = int((highs["RSI"] >= 70).sum()) if not highs.empty and "RSI" in highs.columns else 0
    oversold_count    = int((lows["RSI"]  <= 30).sum()) if not lows.empty  and "RSI" in lows.columns  else 0
    high_vol_highs    = int((highs["Vol vs Avg"] >= 2).sum()) if not highs.empty and "Vol vs Avg" in highs.columns else 0
    high_vol_lows     = int((lows["Vol vs Avg"]  >= 2).sum()) if not lows.empty  and "Vol vs Avg" in lows.columns  else 0

    summary_html = f"""
<div class="summary-box">
  <div>
    <span class="stat highs"><span class="num">{len(highs)}</span><span class="lbl">New 52-Week Highs</span></span>
    <span class="stat lows"><span class="num">{len(lows)}</span><span class="lbl">New 52-Week Lows</span></span>
    <span class="stat"><span class="num">{overbought_count}</span><span class="lbl">Overbought (RSI &gt;70) on Highs</span></span>
    <span class="stat"><span class="num">{oversold_count}</span><span class="lbl">Oversold (RSI &lt;30) on Lows</span></span>
    <span class="stat"><span class="num">{high_vol_highs + high_vol_lows}</span><span class="lbl">High-Volume Breakouts (2x+)</span></span>
  </div>
  <div class="sectors" style="margin-top:16px;">
    <div class="sec-block">
      <h3 style="margin-top:0">Highs by Sector</h3>
      {sector_ul(h_sectors)}
    </div>
    <div class="sec-block">
      <h3 style="margin-top:0">Lows by Sector</h3>
      {sector_ul(l_sectors)}
    </div>
  </div>
  {'<div class="notable"><b>Notable high-volume breakouts:</b> ' + hi_flags + lo_flags + '</div>' if hi_flags or lo_flags else ''}
</div>
"""

    # --- table sections ---
    def tbl_section(title, df):
        if df.empty:
            return f"<h2>{title}</h2><p>None this week.</p>"
        return (
            f"<h2>{title} &mdash; {len(df)} stocks</h2>"
            + df.to_html(index=False, border=0, classes="t", justify="left", na_rep="-")
        )

    return (
        STYLE
        + f"<h1>52-Week High / Low Screener &mdash; week of {today}</h1>"
        + f"<p style='color:#666;font-size:13px'>Universe: S&amp;P 500, Russell 2000, Nasdaq-listed. "
          f"A stock qualifies if it hit its trailing {WINDOW}-day extreme in the last {RECENT_DAYS} trading days. "
          f"Sector shown for S&amp;P 500 members; others labelled Other.</p>"
        + summary_html
        + tbl_section("New 52-Week Highs", highs)
        + tbl_section("New 52-Week Lows", lows)
    )


def build_email_brief(highs, lows):
    today = dt.date.today().isoformat()
    h_sectors = sector_counts(highs)
    l_sectors = sector_counts(lows)

    overbought = (
        highs[highs["RSI"] >= 70]["Ticker"].tolist()
        if not highs.empty and "RSI" in highs.columns else []
    )
    oversold = (
        lows[lows["RSI"] <= 30]["Ticker"].tolist()
        if not lows.empty and "RSI" in lows.columns else []
    )
    high_vol = pd.concat([
        highs[highs["Vol vs Avg"] >= 2.0] if not highs.empty and "Vol vs Avg" in highs.columns else pd.DataFrame(),
        lows[lows["Vol vs Avg"]  >= 2.0] if not lows.empty  and "Vol vs Avg" in lows.columns  else pd.DataFrame(),
    ])

    lines = [
        f"Weekly 52-Week Screener | {today}",
        "=" * 45,
        f"New Highs : {len(highs)}",
        f"New Lows  : {len(lows)}",
        "",
    ]

    if h_sectors:
        lines.append("TOP SECTORS -- HIGHS:")
        for s, n in list(h_sectors.items())[:5]:
            lines.append(f"  {s}: {n}")
        lines.append("")

    if l_sectors:
        lines.append("TOP SECTORS -- LOWS:")
        for s, n in list(l_sectors.items())[:5]:
            lines.append(f"  {s}: {n}")
        lines.append("")

    if not high_vol.empty:
        lines.append("HIGH-VOLUME BREAKOUTS (2x+ avg volume):")
        for _, r in high_vol.head(10).iterrows():
            lines.append(f"  {r['Ticker']}: {r.get('Vol vs Avg', '?')}x avg volume")
        lines.append("")

    if overbought:
        lines.append(f"OVERBOUGHT HIGHS (RSI > 70): {', '.join(overbought[:15])}")
        lines.append("")

    if oversold:
        lines.append(f"OVERSOLD LOWS (RSI < 30): {', '.join(oversold[:15])}")
        lines.append("")

    lines.append("Full report and CSVs attached.")
    return "\n".join(lines)


# ----------------------------- EMAIL ---------------------------------

def send_email(html_body, brief_text, attachments):
    msg = EmailMessage()
    msg["Subject"] = (
        f"52-Week Screener {dt.date.today().isoformat()} "
        f"-- Highs: {attachments[0][1].count(chr(10))-1}, "
        f"Lows: {attachments[1][1].count(chr(10))-1}"
    )
    msg["From"] = EMAIL_USER
    msg["To"]   = EMAIL_TO
    msg.set_content(brief_text)
    msg.add_alternative(html_body, subtype="html")
    for name, csv_text in attachments:
        msg.add_attachment(
            csv_text.encode(), maintype="text", subtype="csv", filename=name
        )
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(EMAIL_USER, EMAIL_PASS)
        s.send_message(msg)
    print(f"Email sent to {EMAIL_TO}")


# ------------------------------ MAIN ---------------------------------

def main():
    tickers    = set()
    sector_map = {}

    if INCLUDE_SP500:
        print("Fetching S&P 500 constituents ...")
        syms, smap = get_sp500_with_sectors()
        tickers   |= set(syms)
        sector_map.update(smap)

    if INCLUDE_NASDAQ:
        print("Fetching Nasdaq-listed symbols ...")
        tickers |= set(get_nasdaq_tickers())

    if INCLUDE_RUSSELL2000:
        print("Fetching Russell 2000 (IWM) holdings ...")
        tickers |= set(get_russell2000_tickers())

    tickers = sorted(t for t in tickers if t and t.replace("-", "").isalnum())
    print(f"\nScreening {len(tickers)} unique tickers ...\n")

    highs, lows = screen(tickers, sector_map)
    print(f"\nFound {len(highs)} new highs and {len(lows)} new lows.")

    html  = build_html(highs, lows)
    brief = build_email_brief(highs, lows)

    h_csv = (highs if not highs.empty else pd.DataFrame()).to_csv(index=False)
    l_csv = (lows  if not lows.empty  else pd.DataFrame()).to_csv(index=False)
    attachments = [
        ("new_52w_highs.csv", h_csv),
        ("new_52w_lows.csv",  l_csv),
    ]

    # Save files
    stamp   = dt.date.today().isoformat()
    out_dir = os.environ.get("REPORT_DIR", "reports")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, f"report_{stamp}.html"), "w") as f:
        f.write(html)
    with open(os.path.join(out_dir, f"brief_{stamp}.txt"), "w") as f:
        f.write(brief)
    for name, text in attachments:
        with open(os.path.join(out_dir, f"{stamp}_{name}"), "w") as f:
            f.write(text)

    print(f"Report saved to {out_dir}/")
    print("\n--- WEEKLY BRIEF PREVIEW ---")
    print(brief)

    if EMAIL_USER and EMAIL_PASS:
        send_email(html, brief, attachments)
    else:
        print("\nEmail not configured -- skipping email delivery.")
        print("To enable: set EMAIL_USER and EMAIL_PASS env vars (Gmail App Password).")


if __name__ == "__main__":
    sys.exit(main())
