## IonQ, Inc., Copyright (c) 2025,
# All rights reserved.
# Use in source and binary forms of this software, without modification,
# is permitted solely for the purpose of activities associated with the IonQ
# Hackathon at iQuHack2025 hosted by MIT and only during the Feb 1-2, 2025 
# duration of such event.

import networkx as nx
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

from copy import deepcopy

def build_ansatz(graph: nx.Graph) -> QuantumCircuit:
    
    ansatz = QuantumCircuit(graph.number_of_nodes())
    ansatz.h(range(graph.number_of_nodes()))

    theta = ParameterVector(r"$\theta$", graph.number_of_edges())
    for t, (u, v) in zip(theta, graph.edges):
        ansatz.cx(u, v)
        ansatz.ry(t, v)
        ansatz.cx(u, v)

    ansatz.barrier()

    return ansatz

def build_new_ansatz(graph: nx.Graph) -> QuantumCircuit:
    num_qubits = graph.number_of_nodes()
    ansatz = QuantumCircuit(num_qubits)
    ansatz.h(range(num_qubits))

    remaining_edges = list(graph.edges)
    theta = ParameterVector(r"$\theta$", len(remaining_edges))
    param_index = 0

    # While there are still edges to schedule
    while remaining_edges:
        # Find a maximal set of edges with no shared qubits (a maximal matching)
        layer = []
        used_nodes = set()
        for edge in remaining_edges:
            u, v = edge
            if u not in used_nodes and v not in used_nodes:
                layer.append(edge)
                used_nodes.update(edge)
        
        # For all edges in the current layer, add gates in parallel
        for u, v in layer:
            t = theta[param_index]
            param_index += 1
            ansatz.cx(u, v)
            ansatz.ry(t, v)
            ansatz.cx(u, v)
        
        # Remove the edges that have been scheduled
        remaining_edges = [edge for edge in remaining_edges if edge not in layer]
        
        # Optionally, add a barrier between layers for clarity
        ansatz.barrier()
    
    return ansatz

def build_maxcut_hamiltonian(graph: nx.Graph) -> SparsePauliOp:
    num_qubits = len(graph.nodes)
    edges = list(graph.edges())
    num_edges = len(edges)

    pauli_terms = ["I"*num_qubits] # start with identity
    coeffs = [-num_edges / 2]

    for (u, v) in edges: # for each edge, add -(1/2)*Z_i Z_j
        z_term = ["I"] * num_qubits
        z_term[u] = "Z"
        z_term[v] = "Z"
        pauli_terms.append("".join(z_term))
        coeffs.append(0.5)

    return SparsePauliOp.from_list(list(zip(pauli_terms, coeffs)))

def build_balanced_maxcut_hamiltonian(graph: nx.Graph, balance_coeff: float = 1.0) -> SparsePauliOp:
    num_qubits = len(graph.nodes)
    nodes = list(graph.nodes)
    # Map node labels to indices (in case nodes are not 0, 1, 2, ...)
    node_to_index = {node: i for i, node in enumerate(nodes)}
    edges = list(graph.edges())
    num_edges = len(edges)
    
    # Use a dictionary to accumulate Pauli strings and their coefficients.
    pauli_dict = {}
    
    # --- Identity term ---
    # Contribution: balance_coeff*|V| (from (∑_v Z_v)²) and - (|E|/2) from the edge penalty.
    identity = "I" * num_qubits
    pauli_dict[identity] = balance_coeff * num_qubits - num_edges / 2
    
    # --- (∑_v Z_v)² term ---
    # Expand (∑_v Z_v)² = ∑_v Z_v² + 2∑_(u<v) Z_u Z_v.
    # Since Z² = I, the first term gives |V| I (already included above) and the off-diagonals contribute:
    # 2*balance_coeff * Z_u Z_v for all u < v.
    for u in range(num_qubits):
        for v in range(u + 1, num_qubits):
            pauli_list = ["I"] * num_qubits
            pauli_list[u] = "Z"
            pauli_list[v] = "Z"
            pauli_str = "".join(pauli_list)
            pauli_dict[pauli_str] = pauli_dict.get(pauli_str, 0) + 2.0 * balance_coeff

    # --- Edge penalty term ---
    # For each edge, add ½*Z_u Z_v. (The constant -½ from -½*(1 - Z_uZ_v) is already absorbed in the identity term.)
    for (u, v) in edges:
        # Use the mapping in case the node labels are not integers starting at 0.
        i = node_to_index[u]
        j = node_to_index[v]
        # Ensure i < j for consistency.
        if i > j:
            i, j = j, i
        pauli_list = ["I"] * num_qubits
        pauli_list[i] = "Z"
        pauli_list[j] = "Z"
        pauli_str = "".join(pauli_list)
        pauli_dict[pauli_str] = pauli_dict.get(pauli_str, 0) + 0.5

    # --- Build the SparsePauliOp ---
    pauli_terms = list(pauli_dict.keys())
    coeffs = [pauli_dict[term] for term in pauli_terms]
    
    return SparsePauliOp.from_list(list(zip(pauli_terms, coeffs)))


def build_maxcut_connectivity_hamiltonian(graph: nx.Graph, lam: float, r: int, s: int) -> SparsePauliOp:
    num_qubits = len(graph.nodes)
    # Dictionary to collect Pauli terms (pauli string -> coefficient)
    terms = {}

    def add_term(pauli_str: str, coeff: float):
        if pauli_str in terms:
            terms[pauli_str] += coeff
        else:
            terms[pauli_str] = coeff

    # ---------------------------
    # 1. Build H_maxcut
    # ---------------------------
    # H_maxcut = (|E|/2)*I - (1/2)*sum_{(u,v) in E} Z_u Z_v
    edges = list(graph.edges())
    num_edges = len(edges)
    identity = "I" * num_qubits
    add_term(identity, -num_edges/2)
    for (u, v) in edges:
        pauli_list = ["I"] * num_qubits
        pauli_list[u] = "Z"
        pauli_list[v] = "Z"
        add_term("".join(pauli_list), 0.5)

    # ---------------------------
    # 2. Helper: convert a set of qubit indices (which get a 'Z') to a Pauli string.
    # ---------------------------
    def pauli_string_from_set(pset: set) -> str:
        return "".join(["Z" if i in pset else "I" for i in range(num_qubits)])
    
    # ---------------------------
    # 3. Build connectivity penalty for partition S
    #    For each v ≠ r, add the term:
    #      (I - Z_v)/2 · ∏_{w in N(v)} ((I + Z_w)/2)
    # ---------------------------
    for v in graph.nodes():
        if v == r:
            continue
        neighbors = list(graph.neighbors(v))
        # Expand (I - Z_v)/2:
        # Two base terms: choosing I on v (coefficient 1/2) or -Z on v (coefficient -1/2)
        base_terms = [(1/2, set()), (-1/2, {v})]
        # For each neighbor w, multiply by the factor (I + Z_w)/2.
        # This factor yields two choices: I with coefficient 1/2 or Z with coefficient 1/2.
        for w in neighbors:
            new_base_terms = []
            for coeff, pset in base_terms:
                # Option 1: choose I on w (factor 1/2)
                new_coeff = coeff * (1/2)
                new_pset = deepcopy(pset)
                new_base_terms.append((new_coeff, new_pset))
                # Option 2: choose Z on w (factor 1/2).
                # (Since Z*Z = I, we “toggle” w: if already present, remove it; else add it.)
                new_coeff = coeff * (1/2)
                new_pset = deepcopy(pset)
                if w in new_pset:
                    new_pset.remove(w)
                else:
                    new_pset.add(w)
                new_base_terms.append((new_coeff, new_pset))
            base_terms = new_base_terms
        # Each expanded term gets multiplied by the penalty coefficient lam.
        for coeff, pset in base_terms:
            p_str = pauli_string_from_set(pset)
            add_term(p_str, lam * coeff)

    # ---------------------------
    # 4. Build connectivity penalty for partition T
    #    For each v ≠ s, add the term:
    #      (I + Z_v)/2 · ∏_{w in N(v)} ((I - Z_w)/2)
    # ---------------------------
    for v in graph.nodes():
        if v == s:
            continue
        neighbors = list(graph.neighbors(v))
        # Expand (I + Z_v)/2:
        base_terms = [(1/2, set()), (1/2, {v})]
        # For each neighbor w, multiply by the factor (I - Z_w)/2.
        # This gives: I with coefficient 1/2 and Z with coefficient -1/2.
        for w in neighbors:
            new_base_terms = []
            for coeff, pset in base_terms:
                # Option 1: choose I on w (factor 1/2)
                new_coeff = coeff * (1/2)
                new_pset = deepcopy(pset)
                new_base_terms.append((new_coeff, new_pset))
                # Option 2: choose Z on w (factor -1/2)
                new_coeff = coeff * (-1/2)
                new_pset = deepcopy(pset)
                if w in new_pset:
                    new_pset.remove(w)
                else:
                    new_pset.add(w)
                new_base_terms.append((new_coeff, new_pset))
            base_terms = new_base_terms
        for coeff, pset in base_terms:
            p_str = pauli_string_from_set(pset)
            add_term(p_str, lam * coeff)
    
    # ---------------------------
    # 5. Prepare the final operator.
    # ---------------------------
    pauli_list = []
    coeff_list = []
    for p_str, coeff in terms.items():
        if abs(coeff) > 1e-10:
            pauli_list.append(p_str)
            coeff_list.append(coeff)
    
    return SparsePauliOp.from_list(list(zip(pauli_list, coeff_list)))