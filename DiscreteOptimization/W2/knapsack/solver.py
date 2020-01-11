#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
import pdb
import random
import time
random.seed(1847859218408232171737)
from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

global SORTED_RATIO


def solve_it_tmp(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')
    scheme = []

    first_line = lines[0].split()
    item_count = int(first_line[0])
    capacity = int(first_line[1])
    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

#
    # a trivial greedy algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    final_output = ''
    final_val = 0
    # Greedy - Highest Value First
    if 0 in scheme:
        output_data_g1, val = greedy_hvf(items, capacity)
        if val >= final_val:
            final_val = val
            final_output = output_data_g1
        # Greedy - Highest Value First END
        # Greedy - Smallest Weight First
    if 1 in scheme:
        output_data_g2, val = greedy_swf(items, capacity)
        if val >= final_val:
            final_val = val
            final_output = output_data_g2
    # Greedy - Smallest Weight First END
    # Greedy - Highest Density First
    if 2 in scheme:
        output_data_g3, val = greedy_hdf(items, capacity)
        if val >= final_val:
            final_val = val
            final_output = output_data_g3
    # # Greedy - Highest Density First END
    # DP
    if 3 in scheme:
        output_data_dp, val = dp(items, capacity)
        if val >= final_val:
            final_val = val
            final_output = output_data_dp
    # DP END

    return final_output


def greedy_swf(items, capacity):
    chosen = []
    tmp = items[0]
    cap = capacity
    val = 0
    while 1:
        min_weight = 1000000
        for item in items:
            if item.weight < min_weight and item.index not in chosen:
                min_weight = item.weight
                tmp = item

        cap -= tmp.weight
        if cap > 0:
            val += tmp.value
            chosen.append(tmp.index)
        else:
            break
    taken = []
    for i in range(len(items)):
        if i in chosen:
            taken.append(1)
        else:
            taken.append(0)
    output_data_g2 = str(val) + ' ' + str(0) + '\n'
    output_data_g2 += ' '.join(map(str, taken))
    return output_data_g2, val


def greedy_hdf(items, capacity):
    chosen = []
    tmp = items[0]
    cap = capacity
    val = 0
    while 1:
        max_den = 0
        for item in items:
            if item.value / item.weight > max_den and item.index not in chosen:
                max_den = item.value / item.weight
                tmp = item

        cap -= tmp.weight
        if cap > 0:
            val += tmp.value
            chosen.append(tmp.index)
        else:
            break
    taken = []
    for i in range(len(items)):
        if i in chosen:
            taken.append(1)
        else:
            taken.append(0)
    output_data_g3 = str(val) + ' ' + str(0) + '\n'
    output_data_g3 += ' '.join(map(str, taken))
    return output_data_g3, val


def greedy_hvf(items, capacity):
    chosen = []
    tmp = items[0]
    cap = capacity
    val = 0
    while 1:
        max_value = 0
        for item in items:
            if item.value > max_value and item.index not in chosen:
                max_value = item.value
                tmp = item

        cap -= tmp.weight
        if cap > 0:
            val += tmp.value
            chosen.append(tmp.index)
        else:
            break
    taken = []
    for i in range(len(items)):
        if i in chosen:
            taken.append(1)
        else:
            taken.append(0)
    output_data_g1 = str(val) + ' ' + str(0) + '\n'
    output_data_g1 += ' '.join(map(str, taken))

    return output_data_g1, val


def dp(items, capacity):
    chosen = []
    k = [[0 for _ in range(capacity + 1)] for _ in range(len(items) + 1)]
    for i in range(len(items) + 1):
        for w in range(capacity + 1):
            if i == 0 or w == 0:
                k[i][w] = 0
            elif items[i - 1].weight <= w:
                k[i][w] = max(items[i - 1].value + k[i - 1][w - items[i - 1].weight],
                              k[i - 1][w])
            else:
                k[i][w] = k[i - 1][w]
    val = k[len(items)][capacity]

    c = capacity
    for i in range(len(items), 0, -1):
        if val <= 0:
            break
        if val == k[i - 1][c]:
            continue
        else:
            chosen.append(i - 1)
            val -= items[i - 1].value
            c -= items[i - 1].weight

    val = k[len(items)][capacity]
    taken = []
    for i in range(len(items)):
        if i in chosen:
            taken.append(1)
        else:
            taken.append(0)
    output_data_dp = str(val) + ' ' + str(1) + '\n'
    output_data_dp += ' '.join(map(str, taken))

    return output_data_dp, val


def bb(items, capacity):

    def est(str_in):
        return sum([int(str_in[i])*items[i].value for i in range(len(str_in))]) + sum([items[i].value for i in range(len(str_in), len(items))])

    values = {
        'bs_val': 0,
        'bs_est': est(''),
        'chosen_s': '',
        'chosen' : []
    }

    def dfs(curr_str, val, weight):
        if len(curr_str) <= len(items) and values['bs_est'] <= est(curr_str):
            if weight + items[len(curr_str)].weight <= capacity:
                if val + items[len(curr_str)].value > values['bs_val']:
                    values['bs_val'] = val + items[len(curr_str)].value
                    values['chosen_s'] = curr_str + '1'
                    values['bs_est'] = est(curr_str+'1')
                dfs(curr_str + '1')
            dfs(curr_str + '0')



def solve_it(input_data):
    global SORTED_RATIO
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i - 1, int(parts[0]), int(parts[1])))

    # default solution

    SORTED_RATIO = make_sorted_ratio(items)
    opt = 1 # is the solution provably optimal?
    if len(items) <= 200:
        # dynamic progr
        value, taken, tab = dynamic_prog(capacity, items)
        # DF-search
    elif len(items) <= 400:
        value, taken, tab = dynamic_prog_2(capacity, items, eps=0.01)
    elif len(items) <= 1000:
        value, taken, tab = dynamic_prog(capacity, items)
    else:
        opt = 0
        value, taken, visited = DFSearch(capacity, items)

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(opt) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


def make_sorted_ratio(items):
    ratio = [(0, 0)] * len(items)
    for idx in range(len(items)):
        ratio[idx] = (items[idx].value /
                     items[idx].weight, idx)
    ratio.sort(key=lambda x: x[0])
    return ratio


def greedy(K, items, ordering='ratio'):
    elems = _order_elem(items, k=ordering)
    sol = [0] * len(items)
    v = 0
    w = 0
    for r in elems:
        if w + items[r[1]].weight > K:
            break
        v += items[r[1]].value
        w += items[r[1]].weight
        sol[r[1]] = 1
    assert_sol(K, items, v, sol)
    return v, sol


def _order_elem(items, k='ratio'):
    elem = [(0, 0)] * len(items)
    if k == 'ratio':
        for idx in range(len(items)):
            elem[idx] = (items[idx].value / items[idx].weight, idx)
        elem.sort(key=lambda x: x[0])
    elif k == 'value':
        for idx in range(len(items)):
            elem[idx] = (items[idx].value, idx)
        elem.sort(key=lambda x: x[0], reverse=True)
    elif k == 'size':
        for idx in range(len(items)):
            elem[idx] = (items[idx].weight, idx)
        elem.sort(key=lambda x: x[0])
    return elem


def dynamic_prog(K, items):
    N = len(items)
    tab = [[0 for i in range(N + 1)] for j in range(K + 1)]

    for i in range(1, K + 1):
        for j in range(1, N + 1):
            if items[j - 1].weight > i:
                tab[i][j] = tab[i][j - 1]
            else:
                tab[i][j] = max(
                    tab[i][j - 1], tab[i - items[j - 1].weight][j - 1] + items[j - 1].value)

    opt = tab[K][N]
    taken = [0] * N
    i, j2 = (K, N)
    while j2 >= 1:
        if tab[i][j2] == tab[i][j2 - 1]:
            taken[j2 - 1] = 0
        else:
            taken[j2 - 1] = 1
            i -= items[j2 - 1].weight
        j2 -= 1
    assert_sol(K, items, opt, taken)
    return opt, taken, tab


def dynamic_prog_2(K, items2, eps=0.2):
    N = len(items2)
    Lbound = max([i.value for i in items2])
    items = []
    for i in items2:
        items.append(Item(i.index, math.ceil(
            i.value / ((eps / N) * Lbound)), i.weight))
    P = sum([i.value for i in items])

    tab = [[max(P, K + 1) for p in range(P + 1)] for i in range(N + 1)]
    tab[0][0] = 0
    for i in range(1, N + 1):
        tab[i][0] = 0
    for i in range(1, N + 1):
        for j in range(1, P + 1):
            if items[i - 1].value <= j:
                tab[i][j] = min(tab[i - 1][j], items[i - 1].weight +
                                tab[i - 1][j - items[i - 1].value])
            else:
                tab[i][j] = tab[i - 1][j]
    opt = -1
    for p in range(P):
        if tab[N][p] <= K:
            opt = max(opt, p)
    sol = [0] * len(items)
    p = opt
    for i in range(N, 0, -1):
        if items[i - 1].value <= p:
            if items[i - 1].weight + tab[i - 1][p - items[i - 1].value] < tab[i - 1][p]:
                sol[i - 1] = 1
                p -= items[i - 1].value
    opt = 0
    for i in range(N):
        if sol[i] == 1:
            opt += items2[i].value
    assert_sol(K, items2, opt, sol)
    return opt, sol, tab


def DFSearch(k, items):
    n1 = Node(k, items, [])
    items2 = sorted(items, key=lambda x: x.value/x.weight, reverse=True)
    best_node, visited = search(k, items2, [n1])
    sol = [0] * len(best_node.sol)
    for i in range(len(best_node.sol)):
        sol[items2[i].index] = best_node.sol[i]
    assert_sol(k, items, best_node.value, sol)
    return best_node.value, sol, visited


def search(k, items, node_list):
    curr_best = None
    visited = 0
    time_limit = 300 # seconds
    start_time = time.time()
    while len(node_list) > 0:
        node = node_list.pop()
        if node.feasible:
            if not node.is_leaf:
                if curr_best is None or node.estimate > curr_best.value:
                    visit(node, node_list, k, items)
                    visited += 1
            else:
                if curr_best is None or node.value > curr_best.value:
                    curr_best = node
        if time.time() - start_time > time_limit:
            print("Time interruption")
            break
    return curr_best, visited


def visit(node, node_list, k, items):
    sol1 = node.sol.copy()
    sol_l = sol1 + [1]
    sol_r = sol1 + [0]
    left = Node(k, items, sol_l)
    right = Node(k, items, sol_r)
    depth_first(node_list, left, right)
    #best_first(node_list, left, right)
    #rand_first(node_list, left, right)


def depth_first(node_list, left, right):
    if right.feasible:
        node_list.append(right)
    if left.feasible:
        node_list.append(left)


def best_first(node_list, left, right):
    if right.feasible:
        node_list.append(right)
    if left.feasible:
        node_list.append(left)
    if left.feasible or right.feasible:
        node_list.sort(key=lambda x: x.value)


def rand_first(node_list, left, right):
    r = random.random()
    if r > 0.5:
        l = [left, right]
    else:
        l = [right, left]
    if l[0].feasible:
        node_list.append(l[0])
    if l[1].feasible:
        node_list.append(l[1])


class Node:
    def __init__(self, k, items, sol):
        w, v = (0, 0)
        self.is_leaf = (len(sol) == len(items))
        for n in range(len(sol)):
            if sol[n]:
                w += items[n].weight
                v += items[n].value
        if w > k:
            self.feasible = False
        else:
            self.value = v
            self.feasible = True
            self.sol = sol
        if self.feasible:
            if not self.is_leaf:
                self.estimate = relax(k, items, sol, self.value)
            else:
                self.estimate = self.value


def relax(k, items, curr_sol, curr_value):
    global SORTED_RATIO
    ratio = SORTED_RATIO[len(curr_sol):len(items)]
    v = curr_value
    w = 0
    for i in range(len(curr_sol)):
        if curr_sol[i] == 1:
            w += items[i].weight
    for r in ratio:
        w += items[r[1]].weight
        if w <= k:
            v += items[r[1]].value
        else:
            extra = k - w
            perc = 1 - extra/items[r[1]].weight
            v += perc * items[r[1]].value
            return v
    return v


def assert_sol(k, items, opt_value, sol):
    v = 0
    w = 0
    for i in range(len(sol)):
        if sol[i] == 1:
            v += items[i].value
            w += items[i].weight
    assert(v == opt_value)
    assert(w <= k)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file. '
              'Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')
