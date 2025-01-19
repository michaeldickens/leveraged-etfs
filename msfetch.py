"""
msfetch.py
--------

Author: Michael Dickens
Created: 2025-01-15

Fetch data from the Marketstack API.

I stopped using Marketstack because it turned out to have severe data quality
issues (e.g. for many ETFs, the adjusted close price does not correctly account
for dividends).

"""

import json
import os
import requests

from dotenv import load_dotenv

load_dotenv()


class Fetcher:
    base_url = "https://api.marketstack.com/v2"
    api_key = os.getenv("MARKETSTACK_API_KEY")

    def fetch(self, ticker, date_from=None):
        offset = 0
        full_data = []
        if date_from is None:
            date_from = "2021-01-01"
        while True:
            url = (
                f"{self.base_url}eod?access_key={self.api_key}"
                + f"&symbols={ticker}"
                + f"&date_from={date_from}"
                + f"&date_to=2025-01-05"
                + f"&offset={offset}"
            )
            response = requests.get(url)
            response.raise_for_status()
            json_data = response.json()
            full_data.extend(json_data["data"])
            offset += json_data["pagination"]["count"]
            print(
                f"Downloaded {offset} records for {ticker}, back to {json_data['data'][-1]['date'].split('T')[0]}"
            )
            if json_data["pagination"]["count"] < 100:
                break

        with open(f"data/{ticker}.json", "w") as outfile:
            json.dump(full_data, outfile)

    def supplemental_fetch(self, ticker):
        with open(f"data/{ticker}.json", "r") as infile:
            full_data = json.load(infile)

        offset = 0
        while True:
            url = (
                f"{self.base_url}/eod?access_key={self.api_key}"
                + f"&symbols={ticker}"
                + f"&date_from=2016-01-01"
                + f"&date_to=2021-01-05"
                + f"&offset={offset}"
            )
            response = requests.get(url)
            response.raise_for_status()
            json_data = response.json()
            full_data.extend(json_data["data"])
            offset += json_data["pagination"]["count"]
            if len(json_data['data']) == 0:
                print(f"Downloaded 0 records for {ticker}")
            else:
                print(
                    f"Downloaded {offset} records for {ticker}, back to {json_data['data'][-1]['date'].split('T')[0]}"
                )
            if json_data["pagination"]["count"] < 100:
                break

        # remove duplicate entries
        seen_dates = set()
        deduped_data = []
        for entry in full_data:
            if entry["date"] not in seen_dates:
                seen_dates.add(entry["date"])
                deduped_data.append(entry)

        deduped_data.sort(key=lambda x: x["date"])

        with open(f"data/{ticker}.json", "w") as outfile:
            json.dump(deduped_data, outfile)


def fetch_etfs(fetch):
    for ticker in "SPY IJH IWM EFA VGK EWJ EEM SPXL UPRO UMDD URTY EFO EURL EZJ EET EDC RSSB".split():
        fetch(ticker)


fetcher = Fetcher()
fetcher.supplemental_fetch("SSO")
# func = fetcher.supplemental_fetch
# fetch_etfs(func)
