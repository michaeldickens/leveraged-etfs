# Leveraged ETFs

Scripts used to calculate [the true cost of leveraged ETFs](https://mdickens.me/2021/03/04/true_cost_of_leveraged_etfs/) and [return stacked funds](https://mdickens.me/2025/02/04/return_stacked_funds/).

- true_cost.py: The main script that calculates ETF costs and performs other calculations.
- avfetch.py: Fetch historical data using the Alpha Vantage API.
- mvfetch.py (deprecated): Fetch historical data using the Marketstack API.
- old.py (deprecated): A copy of the script I used for the original 2021 version of my [article](https://mdickens.me/2021/03/04/true_cost_of_leveraged_etfs/).

To install requirements, run

    pip install -r requirements.txt

## ETF price data

This script relies on having price data in a `data/` directory inside the repository directory. The data is not included in this repository, but you can download a zip file from my website [here](https://mdickens.me/materials/leveraged-etfs-data.zip).

- ETF historical prices downloaded using AlphaVantage API.
- Treasury yields downloaded from the US Treasury.
- S&P Treasury Futures prices downloaded from the S&P Global website.
- NAV data (in morningstar-\<ticker\>.csv) downloaded from Morningstar (who could've guessed?).
