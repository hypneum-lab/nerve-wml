"""
Core routines for the Hopfield model with Hebbian learning, dreaming
(unlearning) and synaptic clipping, as used in

    E. Marinari, S. Rossi, F. Zamponi,
    "Dreaming improves memorization in a Hopfield model with bounded
     synaptic strength".

Conventions
-----------
* N binary neurons s_i = +-1, couplings J (N x N, symmetric, zero diagonal).
* Learning step (Eq. 2):  J_ij <- J_ij + xi_i xi_j / (tau_l sqrt(N))
* Dreaming step (Eq. 3):  J_ij <- J_ij - s*_i s*_j / (tau_d sqrt(N)),
  where s* is the fixed point reached from a random configuration.
* Clipping: after every update, J_ij is set to +-A if |J_ij| > A.
  A very large A (e.g. 1e7) gives the unclipped model.
* A memory counts as retrieved if the dynamics started from it ends in a
  fixed point that differs from it in fewer than 2% of the neurons (Eq. 4).
  The recognition rate rho is the number of retrieved memories divided by N
  (Eq. 5).

All arrays are float32. The heavy routines are compiled with numba.
"""
import numpy as np
from numba import njit


@njit
def seed_numba(seed):
    """Seed the random generator used inside numba-compiled functions."""
    np.random.seed(seed)


@njit
def relax(S, J):
    """Zero-temperature asynchronous dynamics (Eq. 1), applied to each row of S
    until it is a fixed point. At every sweep the neurons are visited in a new
    random order; a neuron flips if it is not aligned with its local field.

    The local field h = J s is updated incrementally after each flip, so a
    sweep without flips costs O(N) instead of O(N^2). J must be symmetric.

    S : (P, N) array of initial states, modified in place and returned.
    J : (N, N) symmetric coupling matrix.

    Safety cap: the routine returns after 1e5 sweeps in total.
    """
    P, N = S.shape
    sweeps = 0
    for p in range(P):
        s = S[p]
        h = J @ s
        relaxed = False
        while not relaxed:
            order = np.random.permutation(np.arange(N))
            relaxed = True
            sweeps += 1
            if sweeps > 100000:
                return S
            for i in order:
                if s[i] * h[i] < 0:
                    relaxed = False
                    old = s[i]
                    s[i] = -old
                    delta = -2.0 * old
                    Ji = J[i]                      # row i == column i (J symmetric)
                    for k in range(N):
                        h[k] += Ji[k] * delta
    return S


@njit
def learn(J, X, A, tau_l):
    """Learning (Eq. 2): present the patterns in X one after the other, clipping
    every coupling to [-A, A] after each pattern.

    J : (N, N) couplings, modified in place and returned.
    X : (L, N) patterns to present, in order.
    """
    L, N = X.shape
    c = 1.0 / (np.sqrt(N) * tau_l)
    for mu in range(L):
        for i in range(N):
            for j in range(i + 1, N):
                v = J[i, j] + X[mu, i] * X[mu, j] * c
                if v > A:
                    v = A
                elif v < -A:
                    v = -A
                J[i, j] = v
                J[j, i] = v
    for i in range(N):
        J[i, i] = 0.0
    return J


def dream(J, A, tau_d):
    """One dreaming step (Eq. 3): relax a random configuration to a fixed point
    s* and remove it from the couplings, then clip to [-A, A].
    J is modified in place and returned."""
    N = J.shape[0]
    s = np.random.choice(np.array([-1, 1], dtype=np.float32), size=(1, N))
    s = relax(s, J)[0]
    J -= np.outer(s, s) * np.float32(1.0 / (tau_d * np.sqrt(N)))
    np.clip(J, -A, A, out=J)
    np.fill_diagonal(J, 0.0)
    return J


def recognition_rate(J, X, tol=0.02):
    """Recognition rate rho (Eq. 5): number of patterns in X whose fixed point
    differs from them in fewer than tol*N neurons, divided by N."""
    N = J.shape[0]
    S = relax(X.copy(), J)
    flipped = (S != X).sum(axis=1)
    return np.sum(flipped < tol * N) / N
