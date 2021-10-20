import time
import pickle
from numpy.random import choice
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
plt.rcParams['image.cmap'] = 'binary'


def run_diploid_mp(cells,end_time, rule, lambdas):
    rule_dict = create_rule_dict(rule)
    pool = initiate_mp_pool()
    results = run_diploid_on_pool(pool,cells,end_time, rule_dict, lambdas)
    with open(f'results{rule}.pickle', 'wb') as f:
        pickle.dump(results, f)
    return cells, end_time


def create_rule_dict(number):
    binary = [int(n) for n in format(number, 'b').zfill(8)]
    rule_dict = {
        7:binary[0],
        6:binary[1],
        5:binary[2],
        4:binary[3],
        3:binary[4],
        2:binary[5],
        1:binary[6],
        0:binary[7],
    }
    return rule_dict


def initiate_mp_pool():
    cores = mp.cpu_count()
    print(f'You have {cores} logical cores')
    pool = mp.Pool(cores - 2)
    return pool


def run_diploid_on_pool(pool,cells, endtime, rule_dict, lambdas,):
    results = pool.starmap(run_diploid,
        [(cells, endtime, _lambda, rule_dict) for _lambda in lambdas])
    pool.close()
    return results


def run_diploid(cells, end_time, _lambda, rule_dict):
    rule_choices_array = choice([0,1], (end_time,cells), p=[1-_lambda,_lambda])
    state = initial_state(cells)
    spacetime_array = np.empty([end_time,cells])
    spacetime_array[0] = state
    density_over_time = [calculate_density(state)]
    for t in range(end_time):
        # determine choice array for this iteration
        rule_choice = rule_choices_array[t]
        state = apply_rule(state, rule_dict, rule_choice)
        spacetime_array[t] = state
        density_over_time.append(calculate_density(state))
    return _lambda, density_over_time, spacetime_array


def initial_state(cells):
    initial_state = np.random.randint(2, size=cells)
    return initial_state


def calculate_density(state):
    density = state.mean()
    return density


def apply_rule(state, rule_dict, rule_choice):
    state_right_neighbor = np.roll(state, -1)
    state_left_neighbor = np.roll(state, 1)
    neighborhoods_num = 4*state_left_neighbor + 2* state + state_right_neighbor
    neighborhood_rule = neighborhoods_num * rule_choice
    next_state = np.array([update_cell(cell,rule_dict) for cell in neighborhood_rule])
    return next_state


def update_cell(cell,rule_dict):
    new_cell = rule_dict[cell]
    return new_cell


if __name__ == '__main__':
    #Parameters
    lambdas = np.arange(0.0,1.05,0.05)
    print(f'lambdas: {lambdas}')
    cells = 1000
    end_time = 500
    rule = 110
    print(f'rule = {rule} , cells = {cells} , end_time = {end_time}')
    
    tic = time.time()
    run_diploid_mp(cells, end_time, rule, lambdas)
    toc = time.time()
    print(f'time:{toc - tic}')



    
