import datetime

from scipy.interpolate import CubicSpline
from scipy.optimize import root
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

################ 4a ###############


def cash_flow(today, issue_date, maturity_date, coupon):
    '''
    given issue date, maturity date and coupon rate, and current date,
    construct array of cash flows and timing of cash flows and accrued interest

    '''

    dates = []
    date_flag = maturity_date
    while date_flag > today:
        dates.insert(0, date_flag)
        date_flag = date_flag - relativedelta(months=6)

    if issue_date > date_flag:
        date_flag = issue_date

    accrued_days = (today - date_flag).days
    next_coupon_date = dates[0]
    days_between_coupons = (next_coupon_date - date_flag).days
    accrued_interest = accrued_days / days_between_coupons * coupon

    days = np.array([(date - today).days for date in dates])
    # time as fraction of years
    years = days / 365
    cf = np.ones_like(years) * coupon
    cf[-1] += 100

    return days, years, cf, accrued_interest


def bond_cash_flow(bond, today, df_bonds, df_prices):
    '''
    retreat cash flow information of a bond
    :param bond: name of bond[str], e.g. 'CA135087K296'
    :param today:  the current date[datetime]
    :return: dirty price of today, the cash flows and its timing
    '''
    price = df_prices[df_prices['Date'] == today][bond].item()
    temp = df_bonds[df_bonds['ISIN'] == bond]
    coupon = temp['coupon'].item()
    maturity = temp['maturity_date'].item()
    issue_date = temp['issue_date'].item()
    _, time, cf, accrued = cash_flow(today, issue_date, maturity, coupon)
    dirty = price + accrued
    return time, cf, dirty


def h(r, time, cf, dirty):
    # r = 0.01
    return dirty - sum(cf * np.exp(-r * time))


def h_der(r, time, cf, dirty):
    return sum(time * cf * np.exp(-r * time))


# import data
dtype = {'coupon': float, 'ISIN': str}
df_bonds = pd.read_csv('info.csv', parse_dates=['maturity_date', 'issue_date'],
                       usecols=['coupon', 'maturity_date', 'issue_date', 'ISIN'])

df_prices = pd.read_csv('prices.csv', parse_dates=['Date'], )

bonds = ['CA135087F254', 'CA135087F585', 'CA135087G328', 'CA135087L369', 'CA135087A610', 'CA135087J546', 'CA135087J967',
         'CA135087K528', 'CA135087K940', 'CA135087L518'
         ]
dates = df_prices['Date'].to_list()


def create_df_ytm(dates, bonds, df_bonds, df_prices):
    df_ytm = []
    for today in dates:
        # given a date, construct a yield curve
        term = []
        ytm = []
        for bond in bonds:
            temp = df_bonds[df_bonds['ISIN'] == bond]
            maturity = temp['maturity_date'].item()
            term.append((maturity - today).days) # note that this is in days. The output "term" will be in days!!

            bond_args = bond_cash_flow(bond, today, df_bonds, df_prices)
            res = root(h, 0.01, args=bond_args)
            ytm.append(res['x'].item())

        today_str = datetime.datetime.strftime(today, '%b %d, %Y')
        df_ytm.append({'date': today_str, 'rate': ytm, 'term': term})
    return df_ytm


df_ytm = create_df_ytm(dates, bonds, df_bonds, df_prices)

# no interpolation
fig, ax = plt.subplots()
for curve in df_ytm:
    ax.plot(curve['term'], curve['rate'], label=curve['date'])
plt.legend()
ax.set_title('yield curve without smoothing')
ax.set_xticks([i * 365 for i in range(6)])
ax.set_xticklabels(range(6))
ax.set_xlabel('term in years')
ax.set_ylabel('YTM')
plt.show()

# cubic spline interpolation
fig, ax = plt.subplots()
for curve in df_ytm:
    term = curve['term']
    rate = curve['rate']
    cs = CubicSpline(term, rate)
    xrange = np.arange(min(term), max(term))
    ax.plot(xrange, cs(xrange), label=curve['date'])
plt.legend()
ax.set_title('yield curve with cubic spline smoothing')
ax.set_xticks([i * 365 for i in range(6)])
ax.set_xticklabels(range(6))
ax.set_xlabel('term in years')
ax.set_ylabel('YTM')
plt.show()



############## 4b ###############

def lookup_rate(time, inventory_time, inventory_rate):
    '''
    helper function to find the two time points in inventory that can provide interpolation.
    if time is not covered by inventory, then return False
    :param invenotry:
    :param time:
    :return: smallest interval of time in inventory that continas time, or return False
    '''
    if time > inventory_time[-1]:
        return False
    else:
        for i in range(len(inventory_time)):
            if time > inventory_time[i]:
                continue
            elif time == inventory_time[i]:
                return inventory_rate[i]
            else:
                interpolated_rate = (1 - (time - inventory_time[i - 1]) / (inventory_time[i] - inventory_time[i - 1])) * \
                                    inventory_rate[
                                        i] + (time - inventory_time[i - 1]) / (
                                            inventory_time[i] - inventory_time[i - 1]) * inventory_rate[
                                        i - 1]
                return interpolated_rate


def g(r, *args):
    args = list(*args)
    p = args.pop(0)
    for i in range(len(args)//2):
        f1 = args[2*i]
        f2 = args[2*i+1]
        p = p-f1*f2**r
    return p



def create_df_spot(dates, bonds, df_bonds, df_prices):
    df_spot = []
    for today in dates:
        inventory_time = []
        inventory_rate = []
        term = []
        for bond in bonds:
            temp = df_bonds[df_bonds['ISIN'] == bond]
            maturity = temp['maturity_date'].item()
            term.append((maturity - today).days)

            time, cf, price = bond_cash_flow(bond, today, df_bonds, df_prices)
            if not inventory_time:
                r = np.log(cf / price) / time
                inventory_time.append(time[0])
                inventory_rate.append(r[0])
            else:
                args = []
                t_new = time[-1]
                t_last = inventory_time[-1]
                r_last = inventory_rate[-1]
                for i in range(len(time)):
                    r = lookup_rate(time[i], inventory_time, inventory_rate)
                    if r:
                        dcf = cf[i]*np.exp(-r* time[i])
                        price = price - dcf
                    else:
                        if t_last<time[i]<=t_new:
                            proportion_last = (t_new - time[i])/(t_new - t_last)
                            proportion_new = 1- proportion_last
                            factor1 = cf[i]*np.exp(-proportion_last*r_last*time[i])
                            factor2 = np.exp(-proportion_new*time[i])
                            args.append(factor1)
                            args.append(factor2)
                args.insert( 0, price)
                res = root(g, 0.02, args=args)
                inventory_time.append(t_new)
                inventory_rate.append(res['x'].item())
        today_str = datetime.datetime.strftime(today, '%b %d, %Y')
        df_spot.append({'date': today_str, 'rate': inventory_rate, 'term': term})
    return df_spot


df_spot = create_df_spot(dates, bonds, df_bonds, df_prices)


# cubic spline interpolation
fig, ax = plt.subplots()
for curve in df_spot:
    term = curve['term']
    rate = curve['rate']
    cs = CubicSpline(term, rate)
    xrange = np.arange(min(term), max(term))
    ax.plot(xrange, cs(xrange), label=curve['date'])
plt.legend()
ax.set_title('spot curve with cubic spline smoothing')
ax.set_xticks([i * 365 for i in range(6)])
ax.set_xticklabels(range(6))
ax.set_xlabel('term in years')
ax.set_ylabel('spot rate')
plt.show()

################ 4c ###############

def create_df_foward(df_spot):
    df_forward = []

    for curve in df_spot:
        inventory_time = []
        inventory_rate = []
        length = len(curve['rate'])

        term = curve['term']
        rate = curve['rate']
        cs = CubicSpline(term, rate)
        r1 = cs(365)
        for i in range(1, length):
            term = curve['term'][i]
            rate = curve['rate'][i]
            t = term/365
            if t<1: continue
            forward_rate = (rate*t-r1)/(t-1)
            inventory_time.append(term-365)
            inventory_rate.append(forward_rate)
        df_forward.append({'date': curve['date'], 'rate': inventory_rate, 'term': inventory_time})
    return df_forward

df_forward = create_df_foward(df_spot)

# cubic spline interpolation
fig, ax = plt.subplots()
for curve in df_forward:
    term = curve['term']
    rate = curve['rate']
    cs = CubicSpline(term, rate)
    xrange = np.arange(min(term), max(term))
    ax.plot(xrange, cs(xrange), label=curve['date'])
plt.legend()
ax.set_title('1-year forward curve with cubic spline smoothing')
ax.set_xticks([i * 365 for i in range(5)])
ax.set_xticklabels(range(5))
ax.set_xlabel('term in years')
ax.set_ylabel('forward rate')
plt.show()


dates_str = [datetime.datetime.strftime(today, '%b %d, %Y') for today in dates]

################ q5 ###############

def interpolate(df, targets):
    '''

    :param df: curve data from different dates list[dict{date, rate, term}]
    :param targets: points you want to interpolate
    :return: interpolated data from different dates list[dict{}]
    '''
    result = []

    for curve in df:
        date = curve['date']
        term = curve['term']
        rate = curve['rate']
        cs = CubicSpline(term, rate)

        new_rate = []
        for target in targets:
            new_rate.append(cs(target).item())
        result.append({date: new_rate})
        # result.append(new_rate)

    return result

def calculate_cov(df, targets):
    '''
    first interpolate data for targets, then calculate covariance matrix
    :param df: dataframe that contain rates information from different dates, list[dict{date, rate, term}]
    :param targets: e.g. [1,2,3] means year1, year2, year3
    :return: matrix
    '''
    targets = 365*np.array(targets)
    df_interpolated = interpolate(df, targets)
    x = []
    for i in range(len(df_interpolated) - 1):
        r2 = np.array(list(df_interpolated[i].values())[0])
        r1 = np.array(list(df_interpolated[i + 1].values())[0])
        x.insert(0, np.log(np.array(r2/r1)))

    X = np.array(x)
    for i in range(X.shape[1]):
        X[:,i] = X[:,i]- np.mean(X[:,i])

    cov = X.transpose()@X
    return cov

targets = [1,2,3,4,5]
cov_spot = calculate_cov(df_spot, targets)

targets = [1,2,3,4]
cov_forward = calculate_cov(df_forward, targets)



################ q6 ###############

eigvalues, eigvectors = np.linalg.eig(cov_spot)
temp = eigvectors.T@eigvectors # confirm that columns of eigenvectors are orthogonal
eigvalues[0]/sum(eigvalues)


eigvalues, eigvectors = np.linalg.eig(cov_forward)
temp = eigvectors.T@eigvectors # confirm that columns of eigenvectors are orthogonal
eigvalues[0]/sum(eigvalues)