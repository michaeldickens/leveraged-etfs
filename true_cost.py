"""

true_cost.py
------------

Author: Michael Dickens
Created: 2025-01-15

"""

import csv
import json
import sys
from datetime import datetime, timedelta
from functools import reduce
from scipy import optimize


def days_per_year(year):
    if year % 400 == 0:
        return 366
    if year % 100 == 0:
        return 365
    if year % 4 == 0:
        return 366
    return 365


# If True, go back to 2016 and exclude all ETFs that have missing data.
# If False, only go back to 2021 and include all ETFs.
date_range = (datetime(2016, 1, 1), datetime(2021, 1, 4))
# date_range = (datetime(2021, 1, 1), datetime(2025, 1, 4))
# date_range = (datetime(2016, 1, 1), datetime(2025, 1, 4))


def setup_globals():
    global date_range, standard_dates, year_bounds, tbill_daily_yields

    # create:
    # - standard_dates: list of all trading days in the date range
    # - year_bounds: list of tuples (start, end) giving the (inclusive)
    #   start and (exclusive) end of each year
    with open("data/adjusted-SPY.json", "r") as infile:
        standard_dates = sorted(
            [
                datetime.strptime(x["date"].split("T")[0], "%Y-%m-%d")
                for x in json.load(infile)
            ]
        )
        standard_dates = [
            date
            for date in standard_dates
            if date >= date_range[0] and date <= date_range[1]
        ]
        year_bounds = []
        prev_year_index = None
        prev_year = None
        for i, date in enumerate(standard_dates):
            year = date.year
            if year != prev_year:
                if prev_year is not None:
                    year_bounds.append((prev_year_index, i))
                prev_year_index = i
                prev_year = year

    # create tbill_daily_yields: list of daily yields for 3-month T-bills,
    # to represent the risk-free rate
    with open("data/Treasury.csv", "r") as infile:
        reader = csv.reader(infile)
        header = next(reader)
        tbill_yields = {}
        for row in reader:
            row_dict = dict(zip(header, row))
            date = datetime.strptime(row_dict["Date"], "%m/%d/%Y")
            tbill_yield = float(row_dict["3 Mo"]) / 100
            tbill_yields[date] = tbill_yield

    # Calculate yield for each trading day as follows:
    # 1. calculate daily yield for every individual day. for non-trading days,
    #    take the last known yield
    # 2. for each trading day, take the cumulative daily yield for all days
    #    since the last trading day
    all_tbill_daily_yields = {}
    date = standard_dates[0]
    annualized_yield = tbill_yields[date]
    while date <= standard_dates[-1]:
        if date in tbill_yields:
            annualized_yield = tbill_yields[date]
        daily_yield = (1 + annualized_yield) ** (1 / days_per_year(date.year)) - 1
        all_tbill_daily_yields[date] = daily_yield
        date += timedelta(days=1)

    tbill_daily_yields = []
    for i in range(len(standard_dates) - 1):
        date = standard_dates[i]
        accum_yield = 0
        while date < standard_dates[i + 1]:
            accum_yield += all_tbill_daily_yields[date]
            date += timedelta(days=1)
        tbill_daily_yields.append(accum_yield)


setup_globals()


def correlation(xs, ys):
    xmean = sum(xs) / len(xs)
    ymean = sum(ys) / len(ys)
    xstd = (sum((x - xmean) ** 2 for x in xs) / len(xs)) ** 0.5
    ystd = (sum((y - ymean) ** 2 for y in ys) / len(ys)) ** 0.5
    return (
        sum((x - xmean) * (y - ymean) for x, y in zip(xs, ys)) / len(xs) / xstd / ystd
    )


def return_improvement(mu, sigma, cost):
    """Calculate the improvement in geometric return of an optimally-leveraged
    portfolio compared to an unlevered portfolio."""
    leverage = (mu + sigma**2 / 2 - cost) / (sigma**2)
    return (
        leverage * (mu + sigma**2 / 2)
        - (leverage * sigma) ** 2 / 2
        - (leverage - 1) * cost
        - mu
    )


def load_prices(ticker):
    filename = f"data/adjusted-{ticker}.json"
    with open(filename, "r") as infile:
        json_data = json.load(infile)
        # adj_close is adjusted for dividends and splits, so no need to account
        # for them manually
        prices = {
            datetime.strptime(x["date"].split("T")[0], "%Y-%m-%d"): x["adj_close"]
            for x in json_data
        }

        earliest_date = min(prices.keys())
        if earliest_date > date_range[0]:
            print(f"{ticker} does not have data back to {date_range[0]}.")
            return None

        if "data" not in filename and ticker == "UPRO":
            # UPRO did a 2:1 share split on 2022-01-13, but this is not
            # accurately reflected in the data, so calculate it manually.
            for date in prices:
                if date >= datetime(2022, 1, 13):
                    prices[date] *= 2

        price0 = prices[standard_dates[0]]
        normalized_prices = {date: price / price0 for date, price in prices.items()}
        price_series = []
        num_missing_dates = 0
        for date in standard_dates:
            if date in normalized_prices:
                price_series.append(normalized_prices[date])
            else:
                price_series.append(price_series[-1])
                num_missing_dates += 1

        # allow one missing date per year
        threshold = (date_range[1] - date_range[0]).days / 365
        if num_missing_dates > threshold:
            print(f"Warning: {ticker} has {num_missing_dates} missing dates.")
        return price_series


def prices_to_returns(prices):
    rets = []
    for i in range(1, len(prices)):
        rets.append(prices[i] / prices[i - 1] - 1)
    return rets


def returns_to_prices(rets):
    price = 1
    prices = [price]
    for ret in rets:
        price *= 1 + ret
        prices.append(price)
    return prices


def add_leverage(prices, leverage_ratio):
    base_rets = prices_to_returns(prices)
    leveraged_rets = [
        leverage_ratio * ret - (leverage_ratio - 1) * rf
        for ret, rf in zip(base_rets, tbill_daily_yields)
    ]
    leveraged_prices = returns_to_prices(leveraged_rets)
    return leveraged_prices


def simulate_return_stacking_v1(stocks, bonds, prop1, prop2):
    rf = tbill_daily_yields
    extra_leverage = prop1 + prop2 - 1
    simulated_prices = [1]
    for i in range(1, len(stocks)):
        stocks_ret = stocks[i] / stocks[i - 1] - 1
        bonds_ret = bonds[i] / bonds[i - 1] - 1
        cost_of_leverage = extra_leverage * rf[i - 1]
        total_ret = prop1 * stocks_ret + prop2 * bonds_ret - cost_of_leverage
        simulated_prices.append(simulated_prices[-1] * (1 + total_ret))
    return simulated_prices


def simulate_return_stacking_v2(stocks, bonds, prop1, prop2):
    rf = tbill_daily_yields
    simulated_prices = [1]
    simulated_props = [(prop1, prop2)]
    for i in range(1, len(stocks)):
        vt_factor = stocks[i] / stocks[i - 1]
        govt_factor = bonds[i] / bonds[i - 1]
        new_props = (
            simulated_props[-1][0] * vt_factor,
            simulated_props[-1][1] * govt_factor,
        )
        net_earnings = (new_props[0] + new_props[1]) - (
            simulated_props[-1][0] + simulated_props[-1][1]
        )
        extra_leverage = (new_props[0] + new_props[1]) - 1
        cost_of_leverage = extra_leverage * rf[i - 1]
        new_price = simulated_prices[-1] + net_earnings - cost_of_leverage
        simulated_prices.append(new_price)

        # only rebalance if the allocation drifts by 5%
        rebalanced_new_props = (
            prop1 * new_price
            if new_props[0] <= 0.95 * prop1 * new_price or new_props[0] >= 1.05 * prop1 * new_price
            else new_props[0],
            prop2 * new_price
            if new_props[1] <= 0.95 * prop2 * new_price or new_props[1] >= 1.05 * prop2 * new_price
            else new_props[1],
        )
        simulated_props.append(rebalanced_new_props)

    return simulated_prices


def calculate_true_cost(
    leveraged_etf, index_etf, leverage_ratio, excess_fee, print_style
):
    extra_leverage = leverage_ratio - 1
    etf_prices = load_prices(leveraged_etf)
    index_prices = load_prices(index_etf)
    if etf_prices is None or index_prices is None:
        return None
    leveraged_index = add_leverage(index_prices, leverage_ratio)
    return calculate_true_cost_inner(
        leveraged_etf,
        etf_prices,
        leveraged_index,
        extra_leverage,
        excess_fee,
        print_style,
    )


def calculate_true_cost_inner(
    leveraged_etf, etf_prices, leveraged_index, extra_leverage, excess_fee, print_style
):
    correl = correlation(
        prices_to_returns(etf_prices), prices_to_returns(leveraged_index)
    )
    true_cost = 0
    excess_costs = []
    if print_style == "by_year":
        print(f"| {leveraged_etf} |", end="")
    for start, end in year_bounds:
        etf_bounds = (etf_prices[start], etf_prices[end])
        index_bounds = (leveraged_index[start], leveraged_index[end])
        etf_ret = etf_bounds[1] / etf_bounds[0] - 1
        leveraged_index_ret = index_bounds[1] / index_bounds[0] - 1
        ret_deficit = (leveraged_index_ret - etf_ret) / extra_leverage
        excess_costs.append(ret_deficit)
        if print_style == "by_year":
            print(f" {100 * ret_deficit:.2f} |", end="")

    # Four different ways of calculating average annual cost:
    # 1. avg_deficit = average of each year's deficit
    # 2. extra_cost_old = annualized return of leveraged ETF - annualized return of index ETF
    # 3. annualized_extra_cost = total deficit, annualized
    # 4. optimized_cost = number such that if you deduct it from the index once per year, you get the leveraged ETF price
    #
    # #2 is how my article did it originally. I believe #4 is the best.

    avg_deficit = sum(excess_costs) / len(excess_costs)

    etf_total_ret = etf_prices[year_bounds[-1][1]] / etf_prices[year_bounds[0][0]] - 1
    leveraged_index_total_ret = (
        leveraged_index[year_bounds[-1][1]] / leveraged_index[year_bounds[0][0]] - 1
    )
    etf_annualized = (1 + etf_total_ret) ** (1 / len(year_bounds)) - 1
    leveraged_index_annualized = (1 + leveraged_index_total_ret) ** (
        1 / len(year_bounds)
    ) - 1
    extra_cost_old = leveraged_index_annualized - etf_annualized
    annualized_extra_cost = (1 + leveraged_index_total_ret - etf_total_ret) ** (
        1 / len(year_bounds)
    ) - 1

    def price_after_cost(annual_cost):
        price = etf_prices[year_bounds[0][0]]
        for start, end in year_bounds:
            price *= leveraged_index[end] / leveraged_index[start]
            price *= 1 - annual_cost
        return price

    optimized_cost = optimize.minimize(
        lambda x: (price_after_cost(x) - etf_prices[year_bounds[-1][1]]) ** 2,
        0,
    ).x[0]

    if print_style == "avg":
        print(
            f"| {leveraged_etf} | {100 * optimized_cost / extra_leverage:.2f}% | {100 * (optimized_cost - excess_fee) / extra_leverage:.2f}% | {correl:.3f} |"
        )
        return [
            optimized_cost / extra_leverage,
            (optimized_cost - excess_fee) / extra_leverage,
        ]
    elif print_style == "by_year":
        print()
        return excess_costs


def true_costs(print_style):
    excess_costs = []

    # Note: EFA and EEM have high fees, I'm using the fee numbers for VEA and
    # VWO instead. But VEA/VWO use different benchmarks so the correlations
    # aren't as good, so I use EFA/EEM as the benchmarks for testing purposes.
    tuples = [
        ("SPXL", "SPY", 3, 0.91, 0.09),
        ("UPRO", "SPY", 3, 0.91, 0.09),
        ("SSO", "SPY", 2, 0.89, 0.09),
        ("UMDD", "IJH", 3, 0.95, 0.05),
        ("URTY", "IWM", 3, 0.95, 0.19),
        ("EFO", "EFA", 2, 0.95, 0.06),
        ("EURL", "VGK", 3, 1.06, 0.09),
        ("EZJ", "EWJ", 2, 1.12, 0.50),
        ("EET", "EEM", 2, 0.95, 0.08),
        ("EDC", "EEM", 3, 1.13, 0.08),
        ("TQQQ", "QQQ", 3, 0.84, 0.20),
    ]

    for ltic, itic, leverage, fee1, fee2 in tuples:
        excess_fee = (fee1 - fee2 * leverage) / 100
        excess_costs.append(
            calculate_true_cost(ltic, itic, leverage, excess_fee, print_style)
        )

    excess_costs = [x for x in excess_costs if x]

    avg_costs = [
        sum(excess_costs[i][j] for i in range(len(excess_costs))) / len(excess_costs)
        for j in range(len(excess_costs[0]))
    ]
    print("| Average |", end="")
    for cost in avg_costs:
        print(f" {100 * cost:.2f} |", end="")
    print()


def print_true_costs():
    print("| ETF | Excess Cost | After Fee | Correlation |")
    print("|---|---|---|---|")
    true_costs(print_style="avg")

    years = []
    for start, end in year_bounds:
        years.append(standard_dates[start].year)

    print("\n| ETF |", end="")
    for year in years:
        print(f" {year} |", end="")
    print("\n|" + "---|" * (len(years) + 1))
    true_costs(print_style="by_year")


def print_return_stacked_costs():
    global date_range
    date_range = (datetime(2024, 1, 1), datetime(2025, 1, 4))
    setup_globals()
    vt = load_prices("VT")
    govt = load_prices("GOVT")
    rssb = load_prices("RSSB")
    excess_fee = (0.36 - 0.07 - 0.05) / 100
    sim_rssb1 = simulate_return_stacking_v1(vt, govt, 1, 1)
    sim_rssb2 = simulate_return_stacking_v2(vt, govt, 1, 1)
    calculate_true_cost_inner("RSSB", rssb, sim_rssb1, 1, excess_fee, "avg")
    calculate_true_cost_inner("RSSB", rssb, sim_rssb2, 1, excess_fee, "avg")
    calculate_true_cost_inner("RSSB", rssb, sim_rssb2, 1, excess_fee, "by_year")

    date_range = (datetime(2019, 1, 1), datetime(2025, 1, 4))
    setup_globals()
    spy = load_prices("SPY")
    govt = load_prices("GOVT")
    ntsx = load_prices("NTSX")
    excess_fee = (0.20 - 0.09 * 0.9 - 0.05 * 0.6) / 100
    sim_ntsx1 = simulate_return_stacking_v1(spy, govt, 0.9, 0.6)
    sim_ntsx2 = simulate_return_stacking_v2(spy, govt, 0.9, 0.6)
    calculate_true_cost_inner("NTSX", ntsx, sim_ntsx2, 0.5, excess_fee, "avg")
    calculate_true_cost_inner("NTSX", ntsx, sim_ntsx2, 0.5, excess_fee, "by_year")

    date_range = (datetime(2022, 1, 1), datetime(2025, 1, 4))
    setup_globals()
    vea = load_prices("VEA")
    govt = load_prices("GOVT")
    ntsi = load_prices("NTSI")
    excess_fee = (0.26 - 0.06 * 0.9 - 0.05 * 0.6) / 100
    sim_ntsi1 = simulate_return_stacking_v1(vea, govt, 0.9, 0.6)
    sim_ntsi2 = simulate_return_stacking_v2(vea, govt, 0.9, 0.6)
    calculate_true_cost_inner("NTSI", ntsi, sim_ntsi2, 0.5, excess_fee, "avg")
    calculate_true_cost_inner("NTSI", ntsi, sim_ntsi2, 0.5, excess_fee, "by_year")

    date_range = (datetime(2016, 1, 1), datetime(2025, 1, 4))
    setup_globals()
    spy = load_prices("SPY")
    govt = load_prices("GOVT")
    psldx = load_prices("PSLDX")
    excess_fee = (0.59 - 0.07 * 1 - 0.05 * 1) / 100
    sim_psldx1 = simulate_return_stacking_v1(spy, govt, 1, 1)
    sim_psldx2 = simulate_return_stacking_v2(spy, govt, 1, 1)
    calculate_true_cost_inner("PSLDX", psldx, sim_psldx1, 1, excess_fee, "avg")
    calculate_true_cost_inner("PSLDX", psldx, sim_psldx2, 1, excess_fee, "avg")
    calculate_true_cost_inner("PSLDX", psldx, sim_psldx2, 1, excess_fee, "by_year")


def print_geometric_mean_improvements():
    tuples = [
        ("US large", 2.0, 5.9, 15.4),
        ("US small", 4.3, 0, 20.7),
        ("EAFE", 6.3, 6.5, 17.3),
        ("Europe", 6.4, 6.3, 19.2),
        ("Japan", 5.8, 6.3, 17.6),
        ("emerging", 7.6, 7.0, 21.7),
    ]

    cost = 1.5
    for name, ra_ret, aqr_ret, stdev in tuples:
        print(f"| {name} | {100 * return_improvement(ra_ret/100, stdev/100, cost/100):.1f}% | {100 * return_improvement(aqr_ret/100, stdev/100, cost/100):.1f}% |")


print_return_stacked_costs()
