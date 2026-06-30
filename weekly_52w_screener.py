#!/usr/bin/env python3
"""
Weekly 52-Week High/Low Screener
---------------------------------
Screens S&P 500, Russell 2000, and Nasdaq-listed stocks for NEW 52-week
highs and lows made during the trailing week, then emails an HTML report
with CSV attachments (or saves them locally if email isn't configured).

Dependencies:
    pip install yfinance pandas requests lxml

Run:
    python weekly_52w_screener.py

Secrets are read from environment variables — nothing is hardcoded.
"""

import os
import io
import sys
import smtplib
import datetime as dt
from email.message import EmailMessage

import pandas as pd
import requests
import yfinance as yf

# =============================== CONFIG ===============================
INCLUDE_SP500       = True
INCLUDE_RUSSELL2000 = True
INCLUDE_NASDAQ      = True

RECENT_DAYS = 5      # how many recent trading days count as "this week"
WINDOW      = 252    # trailing trading days for the 52-week calc (~1 year)
BATCH_SIZE  = 200    # tickers per yfinance download call
AUTO_ADJUST = True   # True = split/dividend adjusted (avoids split artifacts)

# Email — set these as environment variables, do NOT hardcode secrets.
# For Gmail, create an "App Password" (Google Account > Security > 2FA > App passwords).
SMTP_HOST  = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT  = int(os.environ.get("SMTP_PORT", "587"))
EMAIL_USER = os.environ.get("EMAIL_USER")            # e.g. you@gmail.com
EMAIL_PASS = os.environ.get("EMAIL_PASS")            # the app password
EMAIL_TO   = os.environ.get("EMAIL_TO", EMAIL_USER)  # recipient (defaults to yourself)
# =====================================================================


# ----------------------------- TICKERS -------------------------------
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    syms = pd.read_html(url)[0]["Symbol"].astype(str).tolist()
    return [s.replace(".", "-") for s in syms]   # BRK.B -> BRK-B for Yahoo


def get_nasdaq_tickers():
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    r = requests.get(url, timeout=30)
    df = pd.read_csv(io.StringIO(r.text), sep="|")
    df = df[df.get("Test Issue", "N") == "N"]
    syms = df["Symbol"].dropna().astype(str).tolist()
    # keep common-stock-looking symbols; drop warrants/units/footer rows
    return [s for s in syms if s.isalpha()]


def get_russell2000_tickers():
    # Sourced from the iShares Russell 2000 ETF (IWM) holdings file.
    # NOTE: iShares occasionally changes this URL/format — if it breaks,
    # grab the current "Holdings (CSV)" link from the IWM fund page.
    url = ("https://www.ishares.com/us/products/239710/"
           "ishares-russell-2000-etf/1467271812596.ajax"
           "?fileType=csv&fileName=IWM_holdings&dataType=fund")
    r = requests.get(url, timeout=30)
    lines = r.text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.replace('"', "").startswith("Ticker"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    if "Asset Class" in df.columns:
        df = df[df["Asset Class"].astype(str).str.contains("Equity", na=False)]
    syms = df["Ticker"].dropna().astype(str).tolist()
    out = []
    for s in syms:
        s = s.strip().replace(".", "-")
        if s and s not in ("-", "USD") and any(c.isalpha() for c in s):
            out.append(s)
    return out


# ----------------------------- SCREEN --------------------------------
def screen(tickers):
    highs, lows = [], []
    total = len(tickers)
    for i in range(0, total, BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        print(f"  downloading {i + 1}-{min(i + BATCH_SIZE, total)} of {total} ...")
        try:
            data = yf.download(batch, period="380d", interval="1d",
                               group_by="ticker", auto_adjust=AUTO_ADJUST,
                               threads=True, progress=False)
        except Exception as e:
            print(f"  batch failed: {e}")
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
                        "Ticker": t,
                        "Last Close": last_close,
                        "52W High": round(float(roll_high.iloc[-1]), 2),
                        "Date Hit": recent_hi.index[hi_mask][-1].date().isoformat(),
                    })

                lo_mask = (recent_lo <= rl).values
                if lo_mask.any():
                    lows.append({
                        "Ticker": t,
                        "Last Close": last_close,
                        "52W Low": round(float(roll_low.iloc[-1]), 2),
                        "Date Hit": recent_lo.index[lo_mask][-1].date().isoformat(),
                    })
            except Exception:
                continue

    h = pd.DataFrame(highs).sort_values("Ticker") if highs else pd.DataFrame()
    l = pd.DataFrame(lows).sort_values("Ticker") if lows else pd.DataFrame()
    return h, l


# ----------------------------- REPORT --------------------------------
def build_html(highs, lows):
    today = dt.date.today().isoformat()

    def section(title, df):
        if df.empty:
            return f"<h2>{title}</h2><p>None this week.</p>"
        return (f"<h2>{title} — {len(df)}</h2>"
                + df.to_html(index=False, border=0,
                             classes="t", justify="left"))

    style = ("<style>body{font-family:Arial,Helvetica,sans-serif;color:#1a1a1a}"
             "h1{font-size:20px}h2{font-size:16px;margin-top:24px}"
             ".t{border-collapse:collapse;font-size:13px}"
             ".t td,.t th{border:1px solid #ddd;padding:6px 10px;text-align:left}"
             ".t th{background:#f4f4f4}</style>")

    return (style
            + f"<h1>52-Week High / Low Screener — week of {today}</h1>"
            + f"<p>Universe: S&P 500, Russell 2000, Nasdaq-listed. "
              f"A 'new high/low' = price reached its trailing {WINDOW}-day "
              f"extreme on one of the last {RECENT_DAYS} trading days.</p>"
            + section("New 52-Week Highs", highs)
            + section("New 52-Week Lows", lows))


def send_email(html, attachments):
    msg = EmailMessage()
    msg["Subject"] = f"Weekly 52-Week Screener — {dt.date.today().isoformat()}"
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_TO
    msg.set_content("This report is best viewed as HTML. CSVs are attached.")
    msg.add_alternative(html, subtype="html")
    for name, csv_text in attachments:
        msg.add_attachment(csv_text.encode(), maintype="text",
                           subtype="csv", filename=name)
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
        try:
            tickers |= set(get_russell2000_tickers())
        except Exception as e:
            print(f"  Russell 2000 fetch failed ({e}); continuing without it.")

    tickers = sorted(t for t in tickers if t and t.replace("-", "").isalnum())
    print(f"\nScreening {len(tickers)} unique tickers ...\n")

    highs, lows = screen(tickers)
    print(f"\nFound {len(highs)} new highs and {len(lows)} new lows.")

    html = build_html(highs, lows)
    attachments = [
        ("new_52w_highs.csv", (highs if not highs.empty else pd.DataFrame()).to_csv(index=False)),
        ("new_52w_lows.csv",  (lows  if not lows.empty  else pd.DataFrame()).to_csv(index=False)),
    ]

    if EMAIL_USER and EMAIL_PASS:
        send_email(html, attachments)
    else:
        stamp = dt.date.today().isoformat()
        with open(f"report_{stamp}.html", "w") as f:
            f.write(html)
        for name, text in attachments:
            with open(f"{stamp}_{name}", "w") as f:
                f.write(text)
        print("Email not configured (set EMAIL_USER / EMAIL_PASS). "
              f"Saved report_{stamp}.html and CSVs locally instead.")


if __name__ == "__main__":
    sys.exit(main())
