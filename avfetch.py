"""

avfetch.py
----------

Author: Michael Dickens
Created: 2025-01-16

"""


import csv
import json
import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ALPHAVANTAGE_API_KEY")


def save_all_data(ticker):
    """
    TIME_SERIES_DAILY_ADJUSTED endpoint requires a $50/month subscription,
    but we can reconstruct it for free by combining the TIME_SERIES_DAILY and
    DIVIDENDS and SPLITS endpoints and then calculating adjusted prices
    manually.
    """
    def get_url(function):
        return f"https://www.alphavantage.co/query?apikey={api_key}&function={function}&symbol={ticker}&outputsize=full"

    if os.path.exists(f"data/raw-{ticker}.json"):
        print(f"Skipping fetch of {ticker} because data/raw-{ticker}.json already exists")
        return None

    response = requests.get(get_url("TIME_SERIES_DAILY"))
    response.raise_for_status()
    daily_prices = response.json()
    assert "Time Series (Daily)" in daily_prices, daily_prices
    response = requests.get(get_url("DIVIDENDS"))
    response.raise_for_status()
    dividends = response.json()
    assert "data" in dividends, dividends
    response = requests.get(get_url("SPLITS"))
    response.raise_for_status()
    splits = response.json()
    assert "data" in splits, splits

    with open(f"data/raw-{ticker}.json", "w") as outfile:
        if len(splits["data"]) > 0:
            print(f"{ticker} has splits")
        json.dump({
            "daily_prices": daily_prices,
            "dividends": dividends,
            "splits": splits
        }, outfile)

    print("Successfully fetched", ticker)


def build_adjusted_daily_prices(ticker):
    if not os.path.exists(f"data/raw-{ticker}.json"):
        print(f"File data/raw-{ticker}.json does not exist")
        return None

    with open(f"data/raw-{ticker}.json", "r") as infile:
        data = json.load(infile)
        daily_prices = data["daily_prices"]["Time Series (Daily)"]
        dividends = data["dividends"]["data"]
        splits = data["splits"]["data"]

    for row in dividends:
        # note: ex_dividend_date is the date when the dividend impacts the
        # price. payment_date is the date it is considered paid for tax
        # purposes
        assert row["ex_dividend_date"] is not None
        assert row["amount"] is not None
        date = datetime.strptime(row["ex_dividend_date"], "%Y-%m-%d")
        while date.strftime("%Y-%m-%d") not in daily_prices:
            date += timedelta(days=1)
        daily_prices[date.strftime("%Y-%m-%d")]["dividend"] = float(row["amount"])

    for row in splits:
        assert row["effective_date"] is not None
        assert row["split_factor"] is not None
        date = datetime.strptime(row["effective_date"], "%Y-%m-%d")
        while date.strftime("%Y-%m-%d") not in daily_prices:
            date += timedelta(days=1)
        daily_prices[date.strftime("%Y-%m-%d")]["split_factor"] = float(row["split_factor"])

    has_NAV = False
    if os.path.exists(f"data/morningstar-{ticker}.csv"):
        has_NAV = True
        with open(f"data/morningstar-{ticker}.csv", "r") as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                date = datetime.strptime(row["Date"], "%m/%d/%Y")
                formatted_date = date.strftime("%Y-%m-%d")
                if formatted_date in daily_prices:
                    daily_prices[formatted_date]["NAV"] = float(row["NAV"])
                    daily_prices[formatted_date]["NAV With Dividend"] = float(row["NAV With Dividend"])

    adj_ratio = 1
    split_factor = 1
    daily_prices_list = list(reversed(sorted(daily_prices.items(), key=lambda x: x[0])))
    for date, row in daily_prices_list:
        price = float(row["4. close"])
        row["adj_close"] = adj_ratio / split_factor * price
        row["dividend"] = row.get("dividend", 0)
        row["split_factor"] = row.get("split_factor", 1)
        split_factor *= row["split_factor"]
        new_price = price - row["dividend"]
        adj_ratio *= new_price / price
    daily_prices_list.reverse()

    # put into Marketshares format
    adjusted_daily_prices = []
    for date, row in daily_prices_list:
        data = {
            "date": date,
            "close": float(row["4. close"]),
            "adj_close": row["adj_close"],
            "dividend": row["dividend"],
            "split_factor": row["split_factor"],
        }
        if has_NAV:
            data["NAV"] = row["NAV"]
            data["NAV With Dividend"] = row["NAV With Dividend"]
        adjusted_daily_prices.append(data)

    with open(f"data/adjusted-{ticker}.csv", "w") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=adjusted_daily_prices[0].keys())
        writer.writeheader()
        writer.writerows(adjusted_daily_prices)

    return adjusted_daily_prices


def test_adjustment():
    """Test that my adjusted price calculations look correct by comparing them
    to Marketstack adjusted prices.

    Note: Turns out my adjusted prices are right and Marketstack's are wrong so
    this test produces a false negative. My numbers agree with Yahoo Finance
    and Marketstack's numbers are clearly wrong (like the adjusted price does
    not change when a dividend payment occurs).
    """
    with open(f"msdata/SPY.json", "r") as infile:
        json_data = json.load(infile)
    for row in json_data:
        row["date"] = datetime.strptime(row["date"].split("T")[0], "%Y-%m-%d")
    json_data.sort(key=lambda x: x["date"])

    adj_ratio = 1
    for row in json_data:
        price = row["close"]
        dividend = row["dividend"]
        split_factor = row["split_factor"]

        new_price = price + dividend
        adj_ratio *= new_price / price
        row["my_adj_close"] = adj_ratio * split_factor * price

    prev_error = 1
    print("date     \tclose\tadj\tmy_adj\tdiv\terror")
    for row in json_data:
        row["my_adj_close"] /= adj_ratio
        error = row["adj_close"] / row["my_adj_close"]
        # if abs(error - prev_error) > 1e-6:
        if row["date"] >= datetime(2024, 9, 18) and row["date"] <= datetime(2024, 9, 23):
        # if row["date"] >= datetime(2024, 12, 17) and row["date"] <= datetime(2024, 12, 23):
            print(f"{row['date'].strftime('%Y-%m-%d')}\t{row['close']:.2f}\t{row['adj_close']:.2f}\t{row['my_adj_close']:.2f}\t{row['dividend']}\t{error:.3f}")


main_tickers = "SPY IJH IWM EFA VGK EWJ EEM SPXL UPRO UMDD URTY EFO EURL EZJ EET EDC TQQQ QQQ SSO EEM EET EDC RSSB GOVT VT NTSX IEF NTSI VEA TLT TMF TYD".split()

optional_tickers = "DBMF".split()

all_tickers = main_tickers + optional_tickers

for ticker in all_tickers:
    save_all_data(ticker)

for ticker in all_tickers:
    build_adjusted_daily_prices(ticker)
