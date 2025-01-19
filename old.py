"""
old.py
------

Author: Michael Dickens <mdickens93@gmail.com>
Created: 2021-02-05

A (somewhat modified) copy of the script I originally used in 2021 to calculate the true cost of leveraged ETFs. The new and improved script is in true_cost.py.

The reason there are two scripts is that I lost this one and wrote a new
one, then later I found this one again.

"""

import csv
import json
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ALPHAVANTAGE_API_KEY")

today = datetime(2021, 1, 1)
NUM_YEARS = 5
# DAYS_PER_YEAR = 365  # this is what I had before but it's wrong
DAYS_PER_YEAR = 252

def get_tbills_old():
    """3 month T-bills"""
    tbills = {}
    with open("resources/t-bills.csv", "r") as fp:
        for line in fp:
            date_str, ret_str = line.strip().split(",")
            date = datetime.strptime(date_str, "%m/%d/%y")
            ret = (1 + float(ret_str) / 100)**(1/DAYS_PER_YEAR) - 1
            tbills[date] = ret

    return tbills


def get_tbills():
    tbills = {}
    with open("data/Treasury.csv", "r") as infile:
        reader = csv.reader(infile)
        header = next(reader)
        for row in reader:
            row_dict = dict(zip(header, row))
            date = datetime.strptime(row_dict["Date"], "%m/%d/%Y")
            tbill_yield = float(row_dict["3 Mo"]) / 100
            tbills[date] = (1 + tbill_yield)**(1/DAYS_PER_YEAR) - 1
    return tbills


def get_price_history_alphavantage(ticker):
    http_response = requests.get(
        'https://www.alphavantage.co/query?apikey={}&function=TIME_SERIES_DAILY_ADJUSTED&symbol={}&outputsize=full'.format(
            api_key, ticker
        ))

    result = http_response.json()
    dailies = result["Time Series (Daily)"]
    price_series = {}
    for k in dailies:
        price_series[datetime.strptime(k, "%Y-%m-%d")] = float(dailies[k]["5. adjusted close"])

    return price_series


def get_price_history(ticker):
    with open(f"avdata/adjusted-{ticker}.json", "r") as infile:
        json_data = json.load(infile)
        # adj_close is adjusted for dividends and splits, so no need to account
        # for them manually
        prices = {
            datetime.strptime(x["date"].split("T")[0], "%Y-%m-%d"): x["adj_close"]
            for x in json_data
        }
    return prices


def prices_to_returns(dates, prices):
    # dates = list(sorted(prices.keys()))
    return {
        dates[i]: prices[dates[i]] / prices[dates[i-1]] - 1
        for i in range(1, len(dates))
    }


def returns_to_prices(rets):
    # takes a dict but returns a list
    prices = [1]
    for k in rets:
        prices.append(prices[-1] * (1 + rets[k]))
    return prices


def filter_dates(date_range, dictionary):
    return {d: dictionary[d] for d in date_range}


def dict_to_list(dictionary):
    """Convert a dict into a list of values, sorted by the order of the keys."""
    items = list(dictionary.items())
    items.sort(key=lambda pair: pair[0])
    return [pair[1] for pair in items]


def total_return(return_series):
    if type(return_series) == dict:
        return total_return(dict_to_list(return_series))
    else:
        return np.prod([1 + x for x in return_series]) - 1


def annual_return(return_series):
    return (1 + total_return(return_series))**(1 / NUM_YEARS) - 1


def leveraged_etf_cost(leveraged_ticker, base_ticker, leverage):
    date_range_with_weekends = [today - timedelta(days=i) for i in range(365 * NUM_YEARS + 2)]
    date_range_with_weekends.reverse()

    base_prices = get_price_history(base_ticker)
    rf = get_tbills()

    # TODO: there are a few days in base_prices that aren't in rf due to federal
    # holidays. leaving these out will make the results slightly wrong
    date_range = [d for d in date_range_with_weekends if d in base_prices and d in rf]

    leveraged_prices = get_price_history(leveraged_ticker)
    base_rets = prices_to_returns(date_range, base_prices)
    simulated_leveraged_rets = {
        d: leverage * base_rets[d] - (leverage - 1) * rf[d] for d in date_range[1:]
    }
    # print([(x[0].strftime("%Y-%m-%d"), x[1]) for x in sorted(simulated_leveraged_rets.items(), key=lambda x: x[0])])

    leveraged_rets = prices_to_returns(date_range, leveraged_prices)

    simulated_annual_ret = annual_return(simulated_leveraged_rets)
    leveraged_annual_ret = annual_return(leveraged_rets)
    extra_cost = (simulated_annual_ret - leveraged_annual_ret)

    correl = np.corrcoef(
        returns_to_prices(simulated_leveraged_rets), returns_to_prices(leveraged_rets)
    )[0, 1]
    print("{}*{:<3} - {:<4}: {:.2f}%, adjusted {:.2f}% (r={:.4})".format(
        leverage, base_ticker, leveraged_ticker,
        100 * extra_cost, 100 * extra_cost / (leverage - 1), correl,
    ))
    return extra_cost / (leverage - 1)


def leveraged_etf_costs():
    extra_costs = []
    triples = [
        ('SPXL', 'SPY', 3),
        ('UPRO', 'SPY', 3),
        ('UMDD', 'IJH', 3),
        ('URTY', 'IWM', 3),
        ('EFO', 'EFA', 2),
        ('EURL', 'VGK', 3),
        ('EZJ', 'EWJ', 2),
        ('EET', 'EEM', 2),
        ('EDC', 'EEM', 3),
        # ('TQQQ', 'QQQ', 3),
    ]
    for triple in triples:
        extra_costs.append(leveraged_etf_cost(*triple))

    print("Average extra cost: {:.2f}%".format(100 * np.mean(extra_costs)))


leveraged_etf_costs()
