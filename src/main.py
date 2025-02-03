## IonQ, Inc., Copyright (c) 2025,
# All rights reserved.
# Use in source and binary forms of this software, without modification,
# is permitted solely for the purpose of activities associated with the IonQ
# Hackathon at iQuHack2025 hosted by MIT and only during the Feb 1-2, 2025 
# duration of such event.

import Graphs
from QITEvolver import QITEvolver
from CircuitBuilder import (
    build_ansatz, build_new_ansatz,
    build_maxcut_hamiltonian,
    build_balanced_maxcut_hamiltonian,
    build_maxcut_connectivity_hamiltonian
)
from qiskit_aer import AerSimulator
from utils import compute_cut_size, interpret_solution
from Scoring import generate_solutions, final_score
import matplotlib.pyplot as plt

def run_maxcut_experiment(graph,
                          ansatz,
                          challenge='base',
                          learning_rate=0.2,
                          learning_decay=0,
                          momentum=0,
                          verbose=False,
                          num_steps=50,
                          shots=100_000):
    # Generate the Hamiltonian based on the selected challenge
    if challenge == 'base':
        hamiltonian = build_maxcut_hamiltonian(graph)
    elif challenge == 'balanced':
        hamiltonian = build_balanced_maxcut_hamiltonian(graph, 1)
    elif challenge == 'connected':
        hamiltonian = build_maxcut_connectivity_hamiltonian(graph, 5.0, 1, 0)
    else:
        raise ValueError("challenge must be one of: 'base', 'balanced', or 'connected'")
    
    if verbose:
        print("Hamiltonian:")
        print(hamiltonian)
        
    # Set up and run QITE evolution with the selected Hamiltonian and ansatz
    qit_evolver = QITEvolver(hamiltonian, ansatz)
    qit_evolver.evolve(num_steps=num_steps,
                       init_lr=learning_rate,
                       lr_decay=learning_decay,
                       momentum=momentum,
                       verbose=False)
    
    if verbose:
        qit_evolver.plot_convergence()
    
    # Prepare the optimized state for sampling
    backend = AerSimulator()
    optimized_state = ansatz.assign_parameters(qit_evolver.param_vals[-1])
    optimized_state.measure_all()
    counts = backend.run(optimized_state, shots=shots).result().get_counts()
    
    # Determine the best bitstring by computed cut value
    cut_vals = sorted(((bs, compute_cut_size(graph, bs)) for bs in counts),
                      key=lambda t: t[1])
    best_bs = cut_vals[-1][0]
    
    # Generate classical solutions for all variants
    bests = generate_solutions(graph)
    
    # Compute the selected score based on the challenge variant
    score = final_score(graph, bests[challenge]["solutions"], counts, shots, ansatz, challenge)

    if verbose:
        print("\nMeasurement counts:")
        print(counts)
        print("\nInterpreting best bitstring:")
        interpret_solution(graph, best_bs)
        print("Cut value:", compute_cut_size(graph, best_bs))
    
    return {
        "challenge": challenge,
        "counts": counts,
        "best_bitstring": best_bs,
        "bests": bests,
        "score": score,
        "qitevolution": qit_evolver.param_vals
    }

if __name__ == '__main__':
    # graph = Graphs.cycle_graph_c8() 
    # graph = Graphs.complete_bipartite_graph_k88() 
    # graph = Graphs.complete_bipartite_graph_k_nn(5) 
    # graph = Graphs.regular_graph_4_8() 
    # graph = Graphs.cubic_graph_3_16() 
    graph = Graphs.random_connected_graph_16(p=0.18)
    # graph7 = Graphs.expander_graph_n(16) 
    # graph8 = -> make your own cool graph
    
    # Build the ansatz from the graph
    ansatz = build_ansatz(graph)
    
    #challenges = ['balanced']
    challenges = ['balanced', 'connected', 'base']
    results = []
    for challenge in challenges:
        result = run_maxcut_experiment(graph,
                                        ansatz,
                                        challenge=challenge,
                                        learning_rate=0.2,
                                        learning_decay=0,
                                        momentum=0,
                                        num_steps=50,
                                        shots=100_000,
                                        verbose=False)
        
        results.append(result)
        print(f"\nSelected Challenge ({result['challenge']}) Score: {result['score']}")

    for result in results:
        # Print only the selected challenge score.
        print(f"\nSelected Challenge ({result['challenge']}) Score: {result['score']}")
