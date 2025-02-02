import Graphs
from QITEvolver import QITEvolver
from CircuitBuilder import build_ansatz, build_maxcut_hamiltonian, build_balanced_maxcut_hamiltonian
from qiskit_aer import AerSimulator
from utils import compute_cut_size, interpret_solution
from Scoring import generate_solutions, final_score
import matplotlib.pyplot as plt

graph = Graphs.cycle_graph_c8() 
# graph2 = Graphs.complete_bipartite_graph_k88() 
# graph3 = Graphs.complete_bipartite_graph_k_nn(5) 
# graph4 = Graphs.regular_graph_4_8() 
# graph5 = Graphs.cubic_graph_3_16() 
# graph6 = Graphs.random_connected_graph_16(p=0.18)
# graph7 = Graphs.expander_graph_n(16) 
#graph8 = -> make your own cool graph

ansatz = build_ansatz(graph)
ansatz.draw("mpl", fold=-1)

ham = build_balanced_maxcut_hamiltonian(graph)
ham

# Set up your QITEvolver and evolve!
qit_evolver = QITEvolver(ham, ansatz)
qit_evolver.evolve(num_steps=40, lr=0.2, verbose=True) # lr was 0.5

# Visualize your results!
# qit_evolver.plot_convergence()

shots = 100_000

# Sample your optimized quantum state using Aer
backend = AerSimulator()
optimized_state = ansatz.assign_parameters(qit_evolver.param_vals[-1])
optimized_state.measure_all()
counts = backend.run(optimized_state, shots=shots).result().get_counts()

# Find the sampled bitstring with the largest cut value
cut_vals = sorted(((bs, compute_cut_size(graph, bs)) for bs in counts), key=lambda t: t[1])
best_bs = cut_vals[-1][0]

# Now find the most likely MaxCut solution as sampled from your optimized state
# We'll leave this part up to you!!!
most_likely_soln = ""

print(counts)

interpret_solution(graph, best_bs)
print("Cut value: "+str(compute_cut_size(graph, best_bs)))
print(graph, best_bs)

bests = generate_solutions(graph)

# This is classical brute force solver results:
interpret_solution(graph, bests["brute"]["best_solution"])

print("\nBest solution = " + str(bests["brute"]["best_solution"]) + " cost = " + str(bests["brute"]["best_cost"]))
print(graph, bests["brute"]["best_solution"])
print("Solutions : " + str(bests["brute"]["solutions"]))

print(graph, bests["balanced"]["best_solution"])
print("\nBest solutions = " + str(bests["balanced"]["best_solution"]) + " cost = " + str(bests["balanced"]["best_cost"]))
print("Solutions : " + str(bests["balanced"]["solutions"]))

print(graph, bests["connected"]["best_solution"])
print("\nBest connected = " + str(bests["connected"]["best_solution"]) + " cost = " + str(bests["connected"]["best_cost"]))
print("Solutions : " + str(bests["balanced"]["solutions"]))
plt.show()

sum_counts = 0
for bs in counts:
    if bs in bests["brute"]["solutions"]:
        sum_counts += counts[bs]

print(f"Pure max-cut: {sum_counts} out of {shots}")

sum_balanced_counts = 0
for bs in counts:
    if bs in bests["balanced"]["solutions"]:
        sum_balanced_counts += counts[bs]

print(f"Balanced max-cut: {sum_balanced_counts} out of {shots}")

sum_connected_counts = 0
for bs in counts:
    if bs in bests["connected"]["solutions"]:
        sum_connected_counts += counts[bs]

print(f"Connected max-cut: {sum_connected_counts} out of {shots}")

print("Base score: " + str(final_score(graph, bests["brute"]["solutions"], counts,shots,ansatz,'base')))
print("Balanced score: " + str(final_score(graph,bests["balanced"]["solutions"],counts,shots,ansatz,'balanced')))
print("Connected score: " + str(final_score(graph,bests["connected"]["solutions"],counts,shots,ansatz,'connected')))