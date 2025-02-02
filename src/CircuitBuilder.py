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

def build_ansatz(graph: nx.Graph) -> QuantumCircuit:
    
    ansatz = QuantumCircuit(graph.number_of_nodes())
    ansatz.h(range(graph.number_of_nodes()))

    theta = ParameterVector(r"$\theta$", graph.number_of_edges())
    for t, (u, v) in zip(theta, graph.edges):
        ansatz.cx(u, v)
        ansatz.ry(t, v)
        ansatz.cx(u, v)

    return ansatz

def build_maxcut_hamiltonian(graph: nx.Graph) -> SparsePauliOp:
    """
    Build the MaxCut Hamiltonian for the given graph H = (|E|/2)*I - (1/2)*Σ_{(i,j)∈E}(Z_i Z_j)
    """
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

def build_balanced_maxcut_hamiltonian(graph: nx.Graph) -> SparsePauliOp:
    """
    Build the balanced MaxCut Hamiltonian for the given graph:
    
        H = (∑_v Z_v)² - ½ ∑_(u,v)∈E (1 - Z_u Z_v)
    
    which expands to:
    
        H = (|V| - |E|/2)*I + 2∑_(u<v) Z_u Z_v + ½∑_(u,v)∈E Z_u Z_v.
    
    Returns:
        SparsePauliOp: the Pauli-string representation of H.
    """
    num_qubits = len(graph.nodes)
    nodes = list(graph.nodes)
    # Map node labels to indices (in case nodes are not 0, 1, 2, ...)
    node_to_index = {node: i for i, node in enumerate(nodes)}
    edges = list(graph.edges())
    num_edges = len(edges)
    
    # Use a dictionary to accumulate Pauli strings and their coefficients.
    # Keys will be strings like "IZZI" meaning Z on qubits 1 and 2, Identity elsewhere.
    pauli_dict = {}
    
    # --- Identity term ---
    # Contribution: (|V| - (|E|/2)) * I
    identity = "I" * num_qubits
    pauli_dict[identity] = num_qubits - num_edges / 2
    
    # --- (sum Z_v)^2 term ---
    # Expand (∑_v Z_v)² = ∑_v Z_v² + 2∑_(u<v) Z_u Z_v. Since Z² = I, the first term
    # gives |V|*I which we already included above (note that the |V| is part of the identity term).
    # Now add 2*Z_uZ_v for all u < v.
    for u in range(num_qubits):
        for v in range(u + 1, num_qubits):
            # Build the Pauli string with Z on positions u and v and I elsewhere.
            pauli_list = ["I"] * num_qubits
            pauli_list[u] = "Z"
            pauli_list[v] = "Z"
            pauli_str = "".join(pauli_list)
            pauli_dict[pauli_str] = pauli_dict.get(pauli_str, 0) + 2.0

    # --- Edge penalty term ---
    # For each edge, add ½*Z_u Z_v. (Recall that the term was -½*(1 - Z_uZ_v), and
    # the constant part -½ is absorbed into the identity term.)
    for (u, v) in edges:
        # Use the mapping in case the node labels are not integers starting at 0.
        i = node_to_index[u]
        j = node_to_index[v]
        # For consistency, ensure i < j.
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