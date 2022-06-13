"""Basic definitions."""
# Ville Bergholm 2008-2014

import numpy as np


__all__ = ['I', 'sx', 'sy', 'sz', 'p0', 'p1', 'H',
           'Q_Bell', 'TOLERANCE']


# Pauli matrices
I  = np.eye(2)
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])

# qubit projectors
p0 = np.array([[1, 0], [0, 0]])
p1 = np.array([[0, 0], [0, 1]])

# easy Hadamard
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

# magic basis
Q_Bell = np.array([[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]]) / np.sqrt(2)

# numerical error tolerance
TOLERANCE = max(1e-8, np.finfo(float).eps)
