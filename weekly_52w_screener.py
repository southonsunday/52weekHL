#!/usr/bin/env python3
"""
Weekly 52-Week High/Low Screener
---------------------------------
Screens S&P 500, Russell 2000, and Nasdaq-listed stocks for NEW 52-week
highs and lows made during the trailing week, then saves an HTML report
with CSV attachments to the reports/ folder (and optionally emails them).

Dependencies:
    pip install yfinance pandas requests lxml

Run:
    python weekly_52w_screener.py
"""

import os
import io
import sys
import smtplib
import datetime as dt
from email.message import EmailMessage

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yfinance as yf

# =============================== CONFIG ===============================
INCLUDE_SP500       = True
INCLUDE_RUSSELL2000 = True
INCLUDE_NASDAQ      = True

RECENT_DAYS = 5      # how many recent trading days count as "this week"
WINDOW      = 252    # trailing trading days for the 52-week calc (~1 year)
BATCH_SIZE  = 200    # tickers per yfinance download call
AUTO_ADJUST = True   # True = split/dividend adjusted (avoids split artifacts)

# Email — optional. Set as environment variables, do NOT hardcode.
# For Gmail, create an "App Password": Google Account → Security → App passwords.
SMTP_HOST  = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT  = int(os.environ.get("SMTP_PORT", "587"))
EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASS = os.environ.get("EMAIL_PASS")
EMAIL_TO   = os.environ.get("EMAIL_TO", EMAIL_USER)

# Browser-like headers so sites don't block us as a bot
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
# =====================================================================


def make_session(retries=3, backoff=1.5):
    """HTTP session with retry logic and browser-like headers."""
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


# ----------------------------- TICKERS -------------------------------

def get_sp500_tickers():
    """Fetch S&P 500 tickers from Wikipedia with fallback to cached CSV."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    session = make_session()
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        if not tables:
            raise RuntimeError("No tables found on Wikipedia S&P 500 page")
        df = tables[0]
        syms = df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        # Cache for next time in case Wikipedia blocks
        cache_path = os.path.join(os.path.dirname(__file__) or ".", "sp500_cached.csv")
        pd.DataFrame({"Symbol": syms}).to_csv(cache_path, index=False)
        print(f"  Fetched {len(syms)} S&P 500 tickers from Wikipedia.")
        return syms
    except Exception as e:
        print(f"  Warning: live S&P 500 fetch failed ({e}). Trying cached file.")
        cache_path = os.path.join(os.path.dirname(__file__) or ".", "sp500_cached.csv")
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
            syms = df["Symbol"].astype(str).tolist()
            print(f"  Loaded {len(syms)} S&P 500 tickers from cache.")
            return syms
        raise RuntimeError(
            "Could not fetch S&P 500 tickers from Wikipedia and no cache found. "
            "Add sp500_cached.csv to the repo as a fallback."
        )


def get_nasdaq_tickers():
    """Fetch Nasdaq-listed tickers from Nasdaq's public symbol directory."""
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    session = make_session()
    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), sep="|")
        df = df[df.get("Test Issue", "N") == "N"]
        syms = df["Symbol"].dropna().astype(str).tolist()
        syms = [s for s in syms if s.isalpha()]
        print(f"  Fetched {len(syms)} Nasdaq tickers.")
        return syms
    except Exception as e:
        print(f"  Warning: Nasdaq ticker fetch failed ({e}). Skipping Nasdaq.")
        return []


def get_russell2000_tickers():
    """Fetch Russell 2000 tickers from iShares IWM holdings."""
    url = (
        "https://www.ishares.com/us/products/239710/"
        "ishares-russell-2000-etf/1467271812596.ajax"
        "?fileType=csv&fileName=IWM_holdings&dataType=fund"
    )
    session = make_session()
    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        lines = r.text.splitlines()
        start = next(
            i for i, ln in enumerate(lines)
            if ln.replace('"', "").startswith("Ticker")
        )
        df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
        if "Asset Class" in df.columns:
            df = df[df["Asset Class"].astype(str).str.contains("Equity", na=False)]
        syms = df["Ticker"].dropna().astype(str).tolist()
        out = []
        for s in syms:
            s = s.strip().replace(".", "-")
            if s and s not in ("-", "USD") and any(c.isalpha() for c in s):
                out.append(s)
        print(f"  Fetched {len(out)} Russell 2000 tickers.")
        return out
    except Exception as e:
        print(f"  Warning: Russell 2000 fetch failed ({e}). Skipping Russell 2000.")
        return []


# ----------------------------- SCREEN --------------------------------
def screen(tickers):
    highs, lows = [], []
    total = len(tickers)
    for i in range(0, total, BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        print(f"  Downloading {i + 1}–{min(i + BATCH_SIZE, total)} of {total} ...")
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
                if len(df) < 100:
                    continue

                roll_high = df["High"].rolling(WINDOW, min_periods=100).max()
                roll_low  = df["Low"].rolling(WINDOW, min_periods=100).min()

                recent_hi = df["High"].tail(RECENT_DAYS)
                recent_lo = df["Low"].tail(RECENT_DAYS)
                rh = roll_high.tail(RECENT_DAYS)
                rl = roll_low.tail(RECENT_DAYS)

                last_close = round(float(df["Close"].iloc[-1]), 2)

                hi_mask = (recent_hi >= rh).values
                if hi_mask.any():
                    highs.append({
                        "Ticker":     t,
                        "Last Close": last_close,
                        "52W High":   round(float(roll_high.iloc[-1]), 2),
                        "Date Hit":   recent_hi.index[hi_mask][-1].date().isoformat(),
                    })

                lo_mask = (recent_lo <= rl).values
                if lo_mask.any():
                    lows.append({
                        "Ticker":     t,
                        "Last Close": last_close,
                        "52W Low":    round(float(roll_low.iloc[-1]), 2),
                        "Date Hit":   recent_lo.index[lo_mask][-1].date().isoformat(),
                    })
            except Exception:
                continue

    h = pd.DataFrame(highs).sort_values("Ticker") if highs else pd.DataFrame()
    l = pd.DataFrame(lows).sort_values("Ticker")  if lows  else pd.DataFrame()
    return h, l


# ----------------------------- REPORT --------------------------------
def build_html(highs, lows):
    today = dt.date.today().isoformat()

    def section(title, df):
        if df.empty:
            return f"<h2>{title}</h2><p>None this week.</p>"
        return (
            f"<h2>{title} — {len(df)}</h2>"
            + df.to_html(index=False, border=0, classes="t", justify="left")
        )

    style = (
        "<style>body{font-family:Arial,Helvetica,sans-serif;color:#1a1a1a}"
        "h1{font-size:20px}h2{font-size:16px;margin-top:24px}"
        ".t{border-collapse:collapse;font-size:13px}"
        ".t td,.t th{border:1px solid #ddd;padding:6px 10px;text-align:left}"
        ".t th{background:#f4f4f4}</style>"
    )

    return (
        style
        + f"<h1>52-Week High / Low Screener — week of {today}</h1>"
        + f"<p>Universe: S&P 500, Russell 2000, Nasdaq-listed. "
          f"A 'new high/low' = price reached its trailing {WINDOW}-day "
          f"extreme on one of the last {RECENT_DAYS} trading days.</p>"
        + section("New 52-Week Highs", highs)
        + section("New 52-Week Lows", lows)
    )


def send_email(html, attachments):
    msg = EmailMessage()
    msg["Subject"] = f"Weekly 52-Week Screener — {dt.date.today().isoformat()}"
    msg["From"]    = EMAIL_USER
    msg["To"]      = EMAIL_TO
    msg.set_content("This report is best viewed as HTML. CSVs are attached.")
    msg.add_alternative(html, subtype="html")
    for name, csv_text in attachments:
        msg.add_attachment(
            csv_text.encode(), maintype="text", subtype="csv", filename=name
        )
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(EMAIL_USER, EMAIL_PASS)
        s.send_message(msg)
    print(f"Emailed report to {EMAIL_TO}")


# ------------------------------ MAIN ---------------------------------
def main():
    tickers = set()

    if INCLUDE_SP500:
        print("Fetching S&P 500 constituents ...")
        tickers |= set(get_sp500_tickers())

    if INCLUDE_NASDAQ:
        print("Fetching Nasdaq-listed symbols ...")
        tickers |= set(get_nasdaq_tickers())

    if INCLUDE_RUSSELL2000:
        print("Fetching Russell 2000 (IWM) holdings ...")
        tickers |= set(get_russell2000_tickers())

    tickers = sorted(t for t in tickers if t and t.replace("-", "").isalnum())
    print(f"\nScreening {len(tickers)} unique tickers ...\n")

    highs, lows = screen(tickers)
    print(f"\nFound {len(highs)} new highs and {len(lows)} new lows.")

    html = build_html(highs, lows)
    attachments = [
        ("new_52w_highs.csv", (highs if not highs.empty else pd.DataFrame()).to_csv(index=False)),
        ("new_52w_lows.csv",  (lows  if not lows.empty  else pd.DataFrame()).to_csv(index=False)),
    ]

    # Save report files
    stamp   = dt.date.today().isoformat()
    out_dir = os.environ.get("REPORT_DIR", "reports")
    os.makedirs(out_dir, exist_ok=True)

    html_path = os.path.join(out_dir, f"report_{stamp}.html")
    with open(html_path, "w") as f:
        f.write(html)

    for name, text in attachments:
        with open(os.path.join(out_dir, f"{stamp}_{name}"), "w") as f:
            f.write(text)

    print(f"Report saved to {out_dir}/")

    # Send email if credentials are configured
    if EMAIL_USER and EMAIL_PASS:
        send_email(html, attachments)
    else:
        print("Email not configured — skipping email delivery.")


if __name__ == "__main__":
    sys.exit(main())
