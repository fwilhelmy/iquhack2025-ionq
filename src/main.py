import Graphs
from QITEvolver import QITEvolver
from CircuitBuilder import (
    build_ansatz, build_new_ansatz,
    build_maxcut_hamiltonian,
    build_balanced_maxcut_hamiltonian, build_balanced_maxcut_hamiltonian2,
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
    """
    Run the QITE MaxCut experiment using a Hamiltonian and scoring variant
    determined by the selected challenge.

    Parameters:
        graph: The graph object defining your MaxCut instance.
        ansatz: The quantum circuit ansatz to be optimized.
        challenge: The challenge variant to run ('base', 'balanced', or 'connected').
            - 'base' uses build_maxcut_hamiltonian and brute-force solutions.
            - 'balanced' uses build_balanced_maxcut_hamiltonian and balanced solutions.
            - 'connected' uses build_maxcut_connectivity_hamiltonian and connectivity solutions.
        learning_rate: Initial learning rate for the QITE evolution.
        learning_decay: Learning rate decay factor.
        momentum: Momentum term for the evolution.
        verbose: If True, prints and plots extra information.
        num_steps: Number of QITE evolution steps.
        shots: Number of measurement shots for the simulator.

    Returns:
        A dictionary containing:
          - counts: The measurement counts from the simulation.
          - best_bitstring: The bitstring with the largest computed cut value.
          - bests: The classical solutions for various challenges.
          - selected_score: The score corresponding to the selected challenge.
          - qitevolution: The evolution parameters from QITE.
    """
    # Generate the Hamiltonian based on the selected challenge
    if challenge == 'base':
        hamiltonian = build_maxcut_hamiltonian(graph)
    elif challenge == 'balanced':
        hamiltonian = build_balanced_maxcut_hamiltonian2(graph, 0.5)
    elif challenge == 'connected':
        hamiltonian = build_maxcut_connectivity_hamiltonian(graph)
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

# Example usage:
if __name__ == '__main__':
    # Create a sample graph (you can choose any available graph)
    # graph = Graphs.cycle_graph_c8() 
    # graph = Graphs.complete_bipartite_graph_k88() 
    graph = Graphs.complete_bipartite_graph_k_nn(5) 
    # graph = Graphs.regular_graph_4_8() 
    # graph = Graphs.cubic_graph_3_16() 
    # graph6 = Graphs.random_connected_graph_16(p=0.18)
    # graph7 = Graphs.expander_graph_n(16) 
    #graph8 = -> make your own cool graph
    
    # Build the ansatz from the graph
    ansatz = build_new_ansatz(graph)
    
    challenges = ['balanced']
    #challenges = ['base', 'balanced', 'selected']
    results = []
    for challenge in challenges:
        result = run_maxcut_experiment(graph,
                                        ansatz,
                                        challenge='balanced',
                                        learning_rate=0.2,
                                        learning_decay=0,
                                        momentum=0,
                                        num_steps=50,
                                        shots=100_000,
                                        verbose=True)
        
        results.append(result)

    for result in results:
        # Print only the selected challenge score.
        print(f"\nSelected Challenge ({result['challenge']}) Score: {result['score']}")
