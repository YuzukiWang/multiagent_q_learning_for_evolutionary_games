import numpy as np
from itertools import product
from sympy import Matrix, Rational
import csv

# ---------- Tool function（C=1, D=0） ----------

def decode_state(index):
    bits = list(map(int, format(index, '04b')))
    bits = [1 - b for b in bits] 
    return [bits[:2], bits[2:]]

def encode_state(state):
    bits = state[0] + state[1]
    bits = [1 - b for b in bits]
    return int(''.join(map(str, bits)), 2)

def apply_strategy(strategy, state_index):
    return strategy[state_index]

# ---------- State transition matrix ----------

def build_transition_matrix_exact(p1_strategy, p2_strategy):
    P = [[Rational(0) for _ in range(16)] for _ in range(16)]
    for s in range(16):
        current = decode_state(s)
        a1 = apply_strategy(p1_strategy, s)
        a2 = apply_strategy(p2_strategy, s)
        next_state = [current[1], [a1, a2]]
        s_next = encode_state(next_state)
        P[s_next][s] = Rational(1)
    return Matrix(P)

# ---------- Stationary distribution ----------

def find_stationary_distribution_exact(P):
    I = Matrix.eye(16)
    nullspace = (P - I).nullspace()
    if not nullspace:
        raise ValueError("No nullspace found — matrix has no eigenvalue 1.")
    v = nullspace[0]
    v = v / sum(v)
    return v

# ---------- main ----------

def run_against_all_pure_strategies_exact(player1_strategy):
    all_pure = [np.array(bits) for bits in product([0, 1], repeat=16)]
    all_distributions = []
    for i, p2_strategy in enumerate(all_pure):
        P = build_transition_matrix_exact(player1_strategy, p2_strategy)
        v = find_stationary_distribution_exact(P)
        all_distributions.append(v)
    return all_distributions  # list of 16-element sympy Vectors

# ---------- save ----------

def save_symbolic_vector_csv(path, all_vectors):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for vec in all_vectors:
            writer.writerow([str(x) for x in vec])

# ---------- MTBR ----------

player1_MTBR = np.array([
    1, 0, 1, 0,
    1, 0, 1, 0,
    1, 0, 1, 0,
    1, 0, 1, 1
])

# ---------- Run ----------

all_distributions = run_against_all_pure_strategies_exact(player1_MTBR)
csv_filename = "invariantDistribution_MTBR_exact.csv"
save_symbolic_vector_csv(csv_filename, all_distributions)
