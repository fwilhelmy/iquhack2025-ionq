## IonQ, Inc., Copyright (c) 2025,
# All rights reserved.
# Use in source and binary forms of this software, without modification,
# is permitted solely for the purpose of activities associated with the IonQ
# Hackathon at iQuHack2025 hosted by MIT and only during the Feb 1-2, 2025 
# duration of such event.

import numpy as np
import networkx as nx
from qiskit import QuantumCircuit, transpile

# Brute-force approach with conditional checks
def generate_solutions(graph, verbose = False):
    G = graph
    n = len(G.nodes())
    w = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            temp = G.get_edge_data(i, j, default=0)
            if temp != 0:
                w[i, j] = 1.0
    if verbose:
        print(w)

    best_cost_brute = 0
    best_cost_balanced = 0
    best_cost_connected = 0

    for b in range(2**n):
        x = [int(t) for t in reversed(list(bin(b)[2:].zfill(n)))]

        # Create subgraphs based on the partition
        subgraph0 = G.subgraph([i for i, val in enumerate(x) if val == 0])
        subgraph1 = G.subgraph([i for i, val in enumerate(x) if val == 1])

        bs = "".join(str(i) for i in x)
        
        # Check if subgraphs are not empty
        if len(subgraph0.nodes) > 0 and len(subgraph1.nodes) > 0:
            cost = 0
            for i in range(n):
                for j in range(n):
                    cost = cost + w[i, j] * x[i] * (1 - x[j])
            if best_cost_brute < cost:
                best_cost_brute = cost
                xbest_brute = x
                XS_brut = []
            if best_cost_brute == cost:
                XS_brut.append(bs)

            outstr = "case = " + str(x) + " cost = " + str(cost)

            if (len(subgraph1.nodes)-len(subgraph0.nodes))**2 <= 1:
                outstr += " balanced"
                if best_cost_balanced < cost:
                    best_cost_balanced = cost
                    xbest_balanced = x
                    XS_balanced = []
                if best_cost_balanced == cost:
                    XS_balanced.append(bs)

            if nx.is_connected(subgraph0) and nx.is_connected(subgraph1):
                outstr += " connected"
                if best_cost_connected < cost:
                    best_cost_connected = cost
                    xbest_connected = x
                    XS_connected = []
                if best_cost_connected == cost:
                    XS_connected.append(bs)
            if verbose:
                print(outstr)

    return {"base": {"best_cost": best_cost_brute, "best_solution": xbest_brute, "solutions": XS_brut},
            "balanced": {"best_cost": best_cost_balanced, "best_solution": xbest_balanced, "solutions": XS_balanced},
            "connected": {"best_cost": best_cost_connected, "best_solution": xbest_connected, "solutions": XS_connected}, }

def final_score(graph, xs,counts,shots,ansatz,challenge):
    sum_counts = 0
    for bs in counts:
        if bs in xs:
            sum_counts += counts[bs]
    
    transpiled_ansatz = transpile(ansatz, basis_gates = ['cx','rz','sx','x'])
    cx_count = transpiled_ansatz.count_ops()['cx']
    score = (4*2*graph.number_of_edges())/(4*2*graph.number_of_edges() + cx_count) * sum_counts/shots

    return np.round(score,5)