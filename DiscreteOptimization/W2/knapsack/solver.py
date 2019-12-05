#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

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
    if val >= final_val:
        final_val = val
        final_output = output_data_g1
    # Greedy - Highest Value First END
    # Greedy - Smallest Weight First
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
    if val >= final_val:
        final_val = val
        final_output = output_data_g2
    # Greedy - Smallest Weight First END
    # Greedy - Highest Density First
    chosen = []
    tmp = items[0]
    cap = capacity
    val = 0
    while 1:
        max_den = 0
        for item in items:
            if item.value/item.weight > max_den and item.index not in chosen:
                max_den = item.value/item.weight
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
    if val >= final_val:
        final_val = val
        final_output = output_data_g3
    # Greedy - Highest Density First END
    # DP



    
    
    # prepare the solution in the specified output format
    # output_data = str(value) + ' ' + str(0) + '\n'
    # output_data += ' '.join(map(str, taken))
    return final_output


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

