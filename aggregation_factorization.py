#!/usr/bin/python
# coding: utf-8
# author: Nadav Barkai

"""
This script simulates an aggregated data of 2 fields for 2 given dimensions (pivot tables)
and additional table which contains combination of instances from different dimensions
along with their actual value of one of the fields (performance table).
The algorithm tries to find the latent value for each instance such that the aggregated dimension values are maintained.
The simulation receives the number of instances per dimension as program arguments.
"""

import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.linear_model import LinearRegression, Ridge
from scipy.optimize import minimize
from IPython.display import display
import argparse


def random_ints_with_sum(_sum, n):
    vals = np.random.multinomial(_sum, np.ones(n)/n, size=1)[0]
    return vals


def build_data(N1, N2, dims, measures):
    """Generate 2D random data"""
    table1_values = [dims[0]+str(n) for n in range(N1)]
    table2_values = [dims[1]+str(n) for n in range(N2)]

    arr = []
    for d0 in table1_values:
        for d1 in table2_values:
            arr.append([d0, d1, np.random.randint(1, 50)])
    arr = np.array(arr, dtype=np.object)
    performance = pd.DataFrame({dims[0]: arr[:,0], dims[1]: arr[:,1], measures[0]: arr[:,-1]})

    v = np.random.randint(1000, 2000, 1)[0]
    table_list = []
    for k in range(len(dims)):
        dv = random_ints_with_sum(v, performance[dims[k]].nunique())
        table_list.append(pd.DataFrame({
            measures[0]: performance.groupby(dims[k]).sum()[measures[0]], 
            measures[1]: dv
            }))
    multi_pivot = pd.concat(table_list, sort=False)

    return multi_pivot, performance


def get_matrix_for_linear_model(pivot, performance, dims, measures):
    matrix = np.zeros((N1+N2,N1+N2))
    for row in range(N1+N2):
        matrix[row, row] = pivot[measures[0]].iloc[row]
        for col in range(N1+N2):
            if(col != row):
                off_diag_val = performance[(performance[dims[0]]==pivot.index[row]) & (performance[dims[1]]==pivot.index[col])][measures[0]]
                matrix[row, col] = off_diag_val
                matrix[col, row] = off_diag_val
    
    return matrix


def make_linear_prediction(X, y):
    m = LinearRegression(fit_intercept=False, n_jobs=-1)
    m.fit(X, y)
    return m.predict(X)


def loss(w, *args):
    performance, response, w_dict, dims, measures = args[0], args[1], args[2], args[3], args[4]
    d0 = performance[dims[0]].unique()
    d1 = performance[dims[1]].unique()
    measure1_per_tuple = np.zeros((len(d0), len(d1)))
    for row, d in enumerate(d0):
        for col, dd in enumerate(d1):
            measure0 = performance[(performance[dims[0]]==d) & (performance[dims[1]]==dd)][measures[0]]
            measure1_per_tuple[row, col] = w[w_dict[d]] * w[w_dict[dd]] * measure0

    measure1_per_d0_instance = measure1_per_tuple.sum(axis=1)
    measure1_per_d1_instance = measure1_per_tuple.sum(axis=0)
    measure_per_instance = np.concatenate([measure1_per_d0_instance, measure1_per_d1_instance], axis=0)
    loss1 = np.sum(np.abs(measure_per_instance - response))
    loss2 = np.sum((measure_per_instance[:len(d0)].sum() - response[:len(d0)].sum())**2)
    loss3 = np.sum((measure_per_instance[len(d0):].sum() - response[len(d0):].sum())**2)
    total_loss = loss1 + loss2 + loss3
    return total_loss


def init(n, performance, pivot, dims, measures):
    x0 = np.random.random_sample(size=n)
    costs = pivot[measures[1]].values
    w_dict = dict(zip(pivot.index, range(len(pivot.index))))
    args = (performance, costs.reshape(n,), w_dict, dims, measures)
    bounds = [(0, None) for _ in range(n)]
    return x0, args, bounds


def sanity(performance, weights, pivot, measure_peformance, measure_to_check):
    dummy = pd.get_dummies(performance[[c for c in performance.columns if c != measure_peformance]], prefix='', prefix_sep='')
    mult = np.multiply(dummy, weights)
    mult[mult==0] = 1
    rows_product = np.product(mult, axis=1)
    arr = performance[measure_peformance].values*rows_product
    for c in performance.columns:
        for v in performance[c].unique():
            if (c == measure_peformance): break
            inds = performance[performance[c]==v].index
            try:
                exp_val = pivot.iloc[pivot.index==v][measure_to_check][0]
                calc_val = np.round(np.sum(arr[inds]))
                assert calc_val == exp_val
            except:
                print("\x1b[1;31m!!!!! sanity test FAILED when calculating %s on %s: %s !!!!!\x1b[0m"%(measure_to_check, c, v))
                print("%s on %s: %s - got %f expecting %f"%(measure_to_check, c, v, calc_val, exp_val))
                return False
    return True
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--N1', help='Size of first dimension', required=True)
    parser.add_argument('--N2', help='Size of second dimension', required=True)
    parse_args = parser.parse_args()
    print()
    N1, N2 = np.int(parse_args.N1), np.int(parse_args.N2)
    dims=['dim0', 'dim1']
    measures=['measure0', 'measure1']
    pivot, performance = build_data(N1, N2, dims=dims, measures=measures)
    print('==========pivot==========')
    display(pivot)
    print('=======performance=======')
    display(performance)
    x0, args, bounds = init(N1+N2, performance, pivot, dims, measures)
    res = minimize(loss, x0, args=args, method="SLSQP", options={'maxiter':100}, bounds=bounds)
    if(res.success & sanity(performance, res.x, pivot, measures[0], measures[1])):
        print("\x1b[1;32m**** Sanity check SUCCEEDED! *****\x1b[0m")
