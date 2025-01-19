# Leveraged ETFs

Scripts used to calculate [the true cost of leveraged ETFs](https://mdickens.me/2021/03/04/true_cost_of_leveraged_etfs/).

- true_cost.py: The main script that calculates ETF costs and performs other calculations.
- avfetch.py: Fetch historical data using the Alpha Vantage API.
- mvfetch.py (unused): Fetch historical data using the Marketstack API.
- old.py (unused): A copy of the script I used for the original 2021 version of my [article](https://mdickens.me/2021/03/04/true_cost_of_leveraged_etfs/).

To install requirements, run

    pip install -r requirements.txt

## ETF price data

This script relies on having price data in a `data/` directory inside the repository directory. The data is not included in this repository, but you can download a zip file from my website [here](https://mdickens.me/materials/leveraged-etfs-data.zip).

If you have an [Alpha Vantage](https://www.alphavantage.co/) API key, you can also download the data from Alpha Vantage by following these steps:

1. Inside this repository, create a directory called `data`.
1. Then create a file called `.env`.
2. Inside `.env`, add this line:

       ALPHAVANTAGE_API_KEY=<your API key goes here>

3. Run `python avfetch.py`.

You can do this with a free API key, but you're rate-limited to 25 requests per day. Fetching data for all ETFs requires running `avfetch.py` once per day for four days.
