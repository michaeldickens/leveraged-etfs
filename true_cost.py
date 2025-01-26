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


# End date is 1-4 instead of 1-1 because 1-4 is the first trading day of the
# year. Calculating returns for a year requires having the price for the first
# day in the following year.
# date_range = (datetime(2016, 1, 1), datetime(2021, 1, 4))
# date_range = (datetime(2021, 1, 1), datetime(2025, 1, 4))
date_range = (datetime(2016, 1, 1), datetime(2025, 1, 4))


def setup_globals(start_date, end_date):
    """Set up global variables. If you want to change `date_range`, you need'
    to call this function again. You will also have to re-load any prices
    series that you've loaded because `load_prices` changes its behavior based
    on these global variables.

    (This is not a great design and it wouldn't be hard to fix but this is a
    500 line script, it doesn't need to be a masterpiece.)
    """
    global date_range, standard_dates, year_bounds, tbill_daily_yields

    date_range = (start_date, end_date)

    # create:
    # - standard_dates: list of all trading days in the date range
    # - year_bounds: list of tuples (start, end) giving the (inclusive)
    #   start and (exclusive) end of each year
    with open("data/adjusted-SPY.csv", "r") as infile:
        reader = csv.DictReader(infile)
        standard_dates = sorted(
            [datetime.strptime(x["date"], "%Y-%m-%d") for x in reader]
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
        reader = csv.DictReader(infile)
        tbill_yields = {}
        for row in reader:
            date = datetime.strptime(row["Date"], "%m/%d/%Y")
            tbill_yield = float(row["3 Mo"]) / 100
            tbill_yields[date] = tbill_yield

    # calculate yield for each trading day as follows:
    # 1. calculate daily interest for every individual day. for non-trading days,
    #    use the last known interest rate
    # 2. for each trading day, set the yield to equal the cumulative daily
    #    interest for all days since the last trading day
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


setup_globals(*date_range)


def correlation(xs, ys):
    # Originally I was gonna import numpy but why add a 36 megabyte dependency
    # when I could write a 5 line function instead?
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


def load_prices(ticker, column="adj_close"):
    with open(f"data/adjusted-{ticker}.csv", "r") as infile:
        reader = csv.DictReader(infile)
        # adj_close is adjusted for dividends and splits, so no need to account
        # for them manually
        prices = {
            datetime.strptime(row["date"], "%Y-%m-%d"): float(row[column])
            for row in reader
        }

    earliest_date = min(prices.keys())
    if earliest_date > date_range[0]:
        print(f"{ticker} does not have data back to {date_range[0]}.")
        return None

    price_series = []
    num_missing_dates = 0
    for date in standard_dates:
        if date in prices:
            price_series.append(prices[date])
        else:
            if column in ["close", "adj_close", "split_factor"]:
                # fill using the last known value
                price_series.append(price_series[-1])
                num_missing_dates += 1
            elif column == "dividend":
                # fill as zero
                price_series.append(0)
            else:
                raise ValueError(f"Unknown column: {column}")

    # allow one missing date per year before printing a warning
    threshold = (date_range[1] - date_range[0]).days / 365
    if num_missing_dates > threshold:
        print(f"Warning: {ticker} has {num_missing_dates} missing dates.")
    return price_series


def back_out_dividends(adj_prices, dividends):
    """Given a list of adjusted prices and a list of dividends, return the
    original prices."""
    # Confirmed that this exactly reproduces prices from adj_prices.
    adj_ratio = 1
    prices = []
    for i in reversed(range(len(adj_prices))):
        price = adj_prices[i] / adj_ratio
        new_price = price - dividends[i]
        adj_ratio *= new_price / price
        prices.append(price)

    return list(reversed(prices))


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


def yields_to_prices(yields, maturity_years):
    """Convert a list of bond yields to a list of raw_prices."""
    raw_prices = []
    for yield_ in yields:
        price = 1 / (1 + yield_) ** maturity_years
        raw_prices.append(price)

    # add interest
    adj_prices = [1]
    assert len(yields) == len(standard_dates)
    for i in range(1, len(yields)):
        daily_yield_factor = (1 + yields[i - 1]) ** (
            ((standard_dates[i] - standard_dates[i - 1]).days)
            / days_per_year(standard_dates[i].year)
        )
        raw_ret_factor = raw_prices[i] / raw_prices[i - 1]
        adj_prices.append(adj_prices[-1] * raw_ret_factor * daily_yield_factor)

    return adj_prices


def get_treasury_prices(maturity_str):
    # TODO: I think this is not accurate because it doesn't match the ETFs that
    # well and it produces a lower correlation with RSSB than the ETFs do.
    # Maybe it's because the ETFs and RSSB use a collection of bonds with close
    # to but not exactly N years of maturity, whereas this method gives the
    # price for the bond with exactly N years of maturity.
    with open("data/Treasury.csv", "r") as infile:
        reader = csv.DictReader(infile)
        tbill_yields = {}
        for row in reader:
            date = datetime.strptime(row["Date"], "%m/%d/%Y")
            yield_ = float(row[maturity_str]) / 100
            tbill_yields[date] = yield_

    maturity_years = {
        "2 Yr": 2,
        "5 Yr": 5,
        "10 Yr": 10,
        "30 Yr": 30,
    }[maturity_str]

    filled_yields = []
    for date in standard_dates:
        if date in tbill_yields:
            filled_yields.append(tbill_yields[date])
        else:
            filled_yields.append(filled_yields[-1])

    return yields_to_prices(filled_yields, maturity_years)


def load_treasury_futures(maturity_str):
    maturity_col = {
        "2 Yr": "S&P 2-Year U.S. Treasury Note Futures Total Return Index",
        "5 Yr": "S&P 5-Year U.S. Treasury Note Futures Total Return Index",
        "10 Yr": "S&P 10-Year U.S. Treasury Note Futures Total Return Index",
        # RSSB presentation benchmark uses US Treasury Bond Index as the 4th
        # bond position, but I think Ultra Futures better matches the actual
        # holding.
        # https://www.returnstackedetfs.com/wp-content/uploads/2024/02/RSSB-Presentation.pdf
        "Total": "S&P U.S. Treasury Bond Index",
        "30 Yr": "S&P Ultra T-Bond Futures Total Return Index",
    }[maturity_str]

    with open("data/Treasury-Futures.csv", "r") as infile:
        reader = csv.DictReader(infile)
        bond_prices = {}
        for row in reader:
            date = datetime.strptime(row["Effective date"], "%m/%d/%Y")
            price = row[maturity_col]
            if price != "":
                bond_prices[date] = float(price)

    filled_prices = []
    for date in standard_dates:
        if date in bond_prices:
            filled_prices.append(bond_prices[date])
        else:
            filled_prices.append(filled_prices[-1])

    return filled_prices


def add_leverage(prices, leverage_ratio):
    """Given a price series for a security, return the price series that you
    would get from applying leverage to that security. Assumes the borrowing
    rate equals the risk-free rate."""
    base_rets = prices_to_returns(prices)
    leveraged_rets = [
        leverage_ratio * ret - (leverage_ratio - 1) * rf
        for ret, rf in zip(base_rets, tbill_daily_yields)
    ]
    leveraged_prices = returns_to_prices(leveraged_rets)
    return leveraged_prices


def return_stacking_daily_rebalance(securities, target_weights):
    """Simulate the return of a return stacked ETF that rebalances daily.

    securities: A list of lists where each sub-list is the price series for a
    security.

    target_weights: The proportions of the fund's liquidation value invested in each
    security. Can add up to more than 1.
    """
    # TODO: this reinvests dividends into the thing that paid the dividend, but
    # what it's supposed to do is reinvest the dividend into the whole
    # portfolio
    rf = tbill_daily_yields
    extra_leverage = sum(target_weights) - 1
    simulated_prices = [1]
    for i in range(1, len(securities[0])):
        security_rets = [sec[i] / sec[i - 1] - 1 for sec in securities]
        cost_of_leverage = extra_leverage * rf[i - 1]
        total_ret = (
            sum(prop * ret for prop, ret in zip(target_weights, security_rets))
            - cost_of_leverage
        )
        simulated_prices.append(simulated_prices[-1] * (1 + total_ret))
    return simulated_prices


def return_stacking_quarterly_rebalance(securities, target_weights):
    """Simulate the return of a return stacked ETF that rebalances quarterly."""
    # TODO: this reinvests dividends into the thing that paid the dividend, but
    # what it's supposed to do is reinvest the dividend into the whole
    # portfolio
    rf = tbill_daily_yields
    extra_leverage = sum(target_weights) - 1
    liquidation_value = [1]
    notional_values = [target_weights]
    rebalance_days = []
    for i, date in enumerate(standard_dates):
        if date.month in [3, 6, 9, 12] and (
            len(rebalance_days) == 0 or rebalance_days[-1][1].month != date.month
        ):
            rebalance_days.append((i, date))
    rebalance_days = set([x[0] for x in rebalance_days])

    for i in range(1, len(securities[0])):
        security_ret_factors = [sec[i] / sec[i - 1] for sec in securities]
        new_weights = [
            prop * factor
            for prop, factor in zip(notional_values[-1], security_ret_factors)
        ]
        net_earnings = sum(new_weights) - sum(notional_values[-1])
        extra_leverage = sum(new_weights) - 1
        cost_of_leverage = extra_leverage * rf[i - 1]
        new_price = liquidation_value[-1] + net_earnings - cost_of_leverage
        liquidation_value.append(new_price)

        if i in rebalance_days:
            new_weights = [prop * new_price for prop in target_weights]
        notional_values.append(new_weights)

    return liquidation_value


def return_stack(tickers_or_prices, target_weights, rebalance_method, drip=False):
    """Simulate the return of a return stacked ETF.

    tickers_or_prices: A list where each item is either a ticker symbol or a
    list of prices.

    target_weights: The proportions of the fund's liquidation value invested in
    each security. Can add up to more than 1.

    rebalance_method: One of the following strings:
    - "daily": rebalance daily
    - "quarterly": rebalance quarterly
    - "5pct": rebalance whenever the allocation drifts by 5%

    drip (default=False): If True, reinvest a security's dividends into itself.
    Otherwise, reinvest dividends into the whole portfolio.

    """
    raw_prices = [
        load_prices(x, "close") if isinstance(x, str) else x for x in tickers_or_prices
    ]
    dividends = [
        load_prices(x, "dividend") if isinstance(x, str) else [0] * len(x)
        for x in tickers_or_prices
    ]
    splits = [
        load_prices(x, "split_factor") if isinstance(x, str) else [1] * len(x)
        for x in tickers_or_prices
    ]
    yields = [
        [div / close for div, close in zip(dividends_list, raw_prices_list)]
        for dividends_list, raw_prices_list in zip(dividends, raw_prices)
    ]

    # adjust prices for splits but not dividends
    for prices, splits in zip(raw_prices, splits):
        for i in range(len(prices)):
            prices[i] *= splits[i]

    if drip:
        # reinvest dividends into the fund that paid the dividends
        raw_prices = [
            load_prices(x, "adj_close") if isinstance(x, str) else x
            for x in tickers_or_prices
        ]
        yields = [[0] * len(x) for x in raw_prices]
        splits = [[1] * len(x) for x in raw_prices]

    # calculate rebalance days
    rebalance_days = []
    for i, date in enumerate(standard_dates):
        if (
            "quarterly" in rebalance_method
            and date.month in [3, 6, 9, 12]
            and (len(rebalance_days) == 0 or rebalance_days[-1][1].month != date.month)
        ):
            rebalance_days.append((i, date))
        else:
            rebalance_days.append((i, date))
    rebalance_days = set([x[0] for x in rebalance_days])

    rf = tbill_daily_yields
    liquidation_value = [1]
    notional_values = target_weights
    for i in range(1, len(raw_prices[0])):
        price_return_factors = [sec[i] / sec[i - 1] for sec in raw_prices]
        total_dividend = sum(
            value * yield_[i] for value, yield_ in zip(notional_values, yields)
        )
        new_weights = [
            prop * factor
            for prop, factor in zip(notional_values, price_return_factors)
        ]
        net_earnings = sum(new_weights) - sum(notional_values)
        extra_leverage = sum(notional_values) - liquidation_value[-1]
        cost_of_leverage = extra_leverage * rf[i - 1]
        new_price = (
            liquidation_value[-1] + net_earnings + total_dividend - cost_of_leverage
        )
        liquidation_value.append(new_price)

        if "5pct" in rebalance_method:
            # only rebalance if the allocation drifts by 5%
            notional_values = [
                prop * new_price
                if new_prop <= 0.95 * prop * new_price
                or new_prop >= 1.05 * prop * new_price
                else new_prop
                for prop, new_prop in zip(target_weights, new_weights)
            ]
        elif "5pp" in rebalance_method:
            # only rebalance if the allocation drifts by 5 percentage points
            notional_values = [
                prop * new_price
                if abs(new_prop - prop * new_price) > 0.05
                else new_prop
                for prop, new_prop in zip(target_weights, new_weights)
            ]
        elif i in rebalance_days:
            notional_values = [prop * new_price for prop in target_weights]
        else:
            # TODO: Previously I didn't have this line which I think was a bug,
            # but the returns are identical with or without it, which I don't
            # understand. Should only matter for quarterly rebalancing but even
            # then the returns are identical.
            notional_values = new_weights

    return liquidation_value


def calculate_true_cost(
    leveraged_etf, index_etf, leverage_ratio, excess_fee, print_style
):
    """Calculate the true cost of a leveraged ETF as the difference in return
    between the actual leveraged ETF and a simulated fund that levers up the
    index ETF.

    leveraged_etf: A string representing the ticker of the leveraged ETF.

    index_etf: A string representing the ticker of the index ETF that can be
    levered up to match the leveraged ETF.

    leverage_ratio: The amount of leverage that the leveraged ETF uses (e.g.
    leverage_ratio of 2 = 2:1 leverage).

    excess_fee: The fee (expense ratio) of the leveraged ETF minus [the expense
    ratio of the index ETF multiplied by leverage_ratio].

    print_style: One of "avg" or "by_year". If "avg", print the average cost of
    each ETF. If "by_year", print a table showing the annual cost for each ETF
    in each year.
    """
    extra_leverage = leverage_ratio - 1
    index_prices = load_prices(index_etf)
    if index_prices is None:
        return None
    leveraged_index = add_leverage(index_prices, leverage_ratio)
    return calculate_true_cost_inner(
        leveraged_etf,
        extra_leverage,
        excess_fee,
        print_style,
        leveraged_index,
    )


def calculate_true_cost_inner(
    leveraged_etf_ticker,
    extra_leverage,
    excess_fee,
    print_style,
    leveraged_index,
    etf_prices=None,
):
    """Helper function for calculate_true_cost that performs the actual
    calculations."""
    if etf_prices is None:
        etf_prices = load_prices(leveraged_etf_ticker)
    correl = correlation(
        prices_to_returns(etf_prices), prices_to_returns(leveraged_index)
    )
    true_cost = 0
    excess_costs = []
    if print_style == "by_year":
        print(f"| {leveraged_etf_ticker} |", end="")
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

    # I believe there's a closed form solution for this number, but it's
    # complicated so it's easier to just do gradient descent.
    optimized_cost = optimize.minimize(
        lambda x: (price_after_cost(x) - etf_prices[year_bounds[-1][1]]) ** 2,
        0,
    ).x[0]

    if print_style == "avg":
        import numpy as np
        print(f"| Index = {((1 + np.mean(prices_to_returns(leveraged_index)))**(252) - 1) * 100:.6f}% |", end="")
        print(
            f"| {leveraged_etf_ticker} | {100 * optimized_cost / extra_leverage:.2f}% | {100 * (optimized_cost - excess_fee) / extra_leverage:.2f}% | {correl:.3f} |"
        )
        return [
            optimized_cost / extra_leverage,
            (optimized_cost - excess_fee) / extra_leverage,
        ]
    elif print_style == "by_year":
        print()
        return excess_costs


def true_costs(print_style):
    """Calculate true costs for a list of leveraged ETFs."""
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
    setup_globals(datetime(2024, 1, 1), datetime(2025, 1, 4))
    calculate_true_cost_inner(
        "RSSB",
        1,
        (0.36 - 0.03 * 0.64 - 0.08 * 0.36) / 100,
        "avg",
        return_stack(
            [
                "VTI",
                "VXUS",
                load_treasury_futures("2 Yr"),
                load_treasury_futures("5 Yr"),
                load_treasury_futures("10 Yr"),
                load_treasury_futures("30 Yr"),
            ],
            [0.62, 0.38, 0.25, 0.25, 0.25, 0.25],
            "5pp",
        ),
        etf_prices=load_prices("RSSB", "NAV With Dividend"),
    )
    calculate_true_cost_inner(
        "RSSB",
        1,
        (0.36 - 0.07 * 1) / 100,
        "avg",
        return_stack(
            [
                "VT",
                load_treasury_futures("2 Yr"),
                load_treasury_futures("5 Yr"),
                load_treasury_futures("10 Yr"),
                load_treasury_futures("30 Yr"),
            ],
            [1, 0.25, 0.25, 0.25, 0.25],
            "5pp",
        ),
        etf_prices=load_prices("RSSB", "NAV With Dividend"),
    )

    setup_globals(datetime(2019, 1, 1), datetime(2025, 1, 4))
    for print_type in ["avg", "by_year"]:
        calculate_true_cost_inner(
            "NTSX",
            0.5,
            (0.20 - 0.09 * 0.9) / 100,
            print_type,
            return_stack(
                [
                    "SPY",
                    load_treasury_futures("2 Yr"),
                    load_treasury_futures("5 Yr"),
                    load_treasury_futures("10 Yr"),
                    load_treasury_futures("30 Yr"),
                ],
                [0.9, 0.12, 0.12, 0.24, 0.12],
                "quarterly 5pp",
            ),
            etf_prices=load_prices("NTSX", "NAV With Dividend"),
        )


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
        print(
            f"| {name} | {100 * return_improvement(ra_ret/100, stdev/100, cost/100):.1f}% | {100 * return_improvement(aqr_ret/100, stdev/100, cost/100):.1f}% |"
        )


# print_true_costs()
print_return_stacked_costs()
# print_geometric_mean_improvements()
