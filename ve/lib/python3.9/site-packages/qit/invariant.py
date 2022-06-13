"""
Local invariants
================

This module contains tools for computing and plotting the values of
various gate and state local invariants.

.. currentmodule:: qit.invariant

Contents
--------

.. autosummary::
   state_inv
   canonical_inv
   canonical_inv_normalize
   makhlin_inv
   gate_max_concurrence
   gate_adjoint_rep
   gate_leakage_inv
   plot_makhlin_2q
   plot_weyl_2q
"""
# Ville Bergholm 2011

import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from .base import sy, Q_Bell
from .lmap import Lmap
from .utils import tensorbasis, majorize, trisurf



def state_inv(rho: 'State', k: int, perms: 'Sequence[Sequence[int]]') -> complex:
    r"""Local unitary polynomial invariants of quantum states.

    Computes the permutation invariant :math:`I_{k; \pi_1, \pi_2, \ldots, \pi_n}` for the state :math:`\rho`,
    defined as :math:`\trace(\rho^{\otimes k} \Pi)`, where :math:`\Pi` permutes all k copies of the i:th subsystem using :math:`\pi_i`.

    Args:
      rho: quantum state with ``n`` subsystems
      k: order of the invariant, ``k >= 1``
      perms: Permutations. ``len(perms) == n``, each element must be
          a full ``k``-permutation (or an empty sequence denoting the identity permutation).

    Returns:
      invariant

    Example: :math:`I_{3; (123),(12)}(\rho) =` ``state_inv(rho, 3, [(1, 2, 0), (1, 0, 2)])``

    This function can be very inefficient for some invariants, since
    it does no partial traces etc. which might simplify the calculation.

    Uses the algorithm in :cite:`BBL2012`.
    """
    # Ville Bergholm 2011-2016

    n = len(perms)
    if n != rho.subsystems():
        raise ValueError('Need one permutation per subsystem.')

    # convert () to identity permutation
    id_perm = np.arange(k)

    # idea: tensor k copies of rho together, then permute all the copies of the i:th subsystem using perms[i] on one side.
    # cols of r correspond to subsystems, rows to copies. the initial order is
    r = np.arange(n * k).reshape((k, n))
    for j, p in enumerate(perms):
        if len(p) == 0:
            p = id_perm
        elif len(p) != k:
            raise ValueError('Permutation #{0} does not have {1} elements.'.format(j, k))
        r[:,j] = r[np.asarray(p),j]  # apply the jth permutation
    r = r.flatten()

    # TODO this could be done much more efficiently
    temp = Lmap(rho.to_op().tensorpow(k))
    return temp.reorder((r, None)).trace()


def canonical_inv_normalize(
        c: 'np.array[float]',
        fix_jittering: bool = True,
) -> 'np.array[float]':
    r"""Normalizes canonical local invariants into the default Weyl chamber.

    The default chamber is defined by :math:`1 \ge c_1 \ge c_2 \ge c_3 \ge 0, \quad c_2 \le 1-c_1`.

    Args:
        c: canonical invariants, `c.shape == (3,)`

    Returns:
        normalized invariants
    """
    # Into the [0, 1) unit cell (contains 24 Weyl chambers).
    c = np.mod(c, 1)
    # Weyl group can produce all permutations plus modular negations of any two invariants.
    # Hence, the orbit of [x, y, z] is all 24 permutations of {[x, y, z], [1-x, 1-y, z], [1-x, y, 1-z], [x, 1-y, 1-z]}.
    # Weyl reflections (approximately binary space partitioning):
    if c[2] > c[0]:  # 24 -> 12
        c[[0, 2]] = c[[2, 0]]
    if c[2] > 1 - c[0]:  # 12 -> 6
        c[[0, 2]] = 1 - c[[2, 0]]
    if c[1] > c[0]:  # 6 -> 3
        c[[0, 1]] = 1 - c[[0, 1]]
    if c[2] > c[1]:  # 3 -> 2
        c[[1, 2]] = c[[2, 1]]
    if c[1] > 1 - c[0]:  # 2 -> 1
        c[[0, 1]] = 1 - c[[1, 0]]

    if fix_jittering:
        # rotate points just above the second half of the XY plane under the first half,
        # to avoid points approximately on the XY plane jumping back and forth
        if c[2] < 1e-6 and c[0] > 0.5:
            c[0] = 1 - c[0]
            c[2] = -c[2]

    return c


def canonical_inv(U: 'np.array[complex]') -> 'np.array[float]':
    """Canonical local invariants of a two-qubit gate.

    Computes a vector of three real canonical local invariants for the :math:`U(4)`
    matrix U, normalized to the range :math:`[0,1]`.

    Args:
      U: :math:`U(4)` matrix

    Returns:
      canonical local invariants of U

    Uses the algorithm in :cite:`Childs`.
    """
    # Ville Bergholm 2004-2010

    sigma = np.kron(sy, sy)
    U_flip = (sigma @ U.transpose()) @ sigma  # spin flipped U
    temp = (U @ U_flip) / np.sqrt(complex(npl.det(U)))

    Lambda = npl.eigvals(temp) #[exp(i*2*phi_1), etc]
    # logarithm to the branch (-1/2, 3/2]
    Lambda = np.angle(Lambda) / np.pi # divide pi away
    for k in range(len(Lambda)):
        if Lambda[k] <= -0.5:
            Lambda[k] += 2
    S = Lambda / 2
    S = np.sort(S)[::-1]  # descending order

    n = int(round(sum(S)))  # sum(S) must be an integer
    # take away extra translations-by-pi
    S -= np.r_[np.ones(n), np.zeros(4-n)]
    # put the elements in the correct order
    S = np.roll(S, -n)

    M = [[1, 1, 0], [1, 0, 1], [0, 1, 1]] # scaled by factor 2
    c = M @ S[:3]
    # now 0.5 >= c[0] >= c[1] >= |c[2]|
    # and into the Berkeley chamber using a translation and two Weyl reflections
    if c[2] < 0:
        c[0] = 1 - c[0]
        c[2] = -c[2]
    c = np.mod(c, 1)
    return c


def makhlin_inv(U: 'np.array[complex]') -> 'np.array[float]':
    """Makhlin local invariants of a two-qubit gate.

    Computes a vector of the three real Makhlin invariants :cite:`Makhlin` corresponding
    to the :math:`U(4)` gate U.
    Alternatively, given a vector of canonical invariants normalized to :math:`[0, 1]`,
    returns the corresponding Makhlin invariants :cite:`Zhang`.

    Args:
      U: :math:`U(4)` matrix

    Returns:
      Makhlin local invariants of U

    Alternatively, U may be given in terms of a vector of three
    canonical local invariants.
    """
    # Ville Bergholm 2004-2010
    if U.shape[-1] == 3:
        c = U
        # array consisting of vectors of canonical invariants
        c *= np.pi
        g = np.empty(c.shape)

        g[..., 0] = (np.cos(c[..., 0]) * np.cos(c[..., 1]) * np.cos(c[..., 2])) ** 2 -(np.sin(c[..., 0])\
            * np.sin(c[..., 1]) * np.sin(c[..., 2])) ** 2
        g[..., 1] = 0.25 * np.sin(2 * c[..., 0]) * np.sin(2 * c[..., 1]) * np.sin(2 * c[..., 2])
        g[..., 2] = 4 * g[..., 0] - np.cos(2 * c[..., 0]) * np.cos(2 * c[..., 1]) * np.cos(2*c[..., 2])
    else:
        # U(4) gate matrix
        V = Q_Bell.conj().transpose() @ (U @ Q_Bell)
        M = V.transpose() @ V

        t1 = np.trace(M) ** 2
        t2 = t1 / (16 * npl.det(U))
        g = np.array([t2.real, t2.imag, ((t1 -np.trace(M @ M)) / (4 * npl.det(U))).real])
    return g


def gate_max_concurrence(U: 'np.array[complex]') -> float:
    """Maximum concurrence generated by a two-qubit gate.

    Returns the maximum concurrence generated by the two-qubit
    gate U (see :cite:`Kraus`), starting from a tensor state.

    Args:
      U: :math:`U(4)` matrix

    Returns:
      maximum concurrence generated by U

    Alternatively, U may be given in terms of a vector of three
    canonical local invariants.
    """
    # Ville Bergholm 2006-2010
    if U.shape[-1] == 4:
        # gate into corresponding invariants
        c = canonical_inv(U)
    else:
        c = U
    temp = np.roll(c, 1, axis=-1)
    return np.max(abs(np.sin(np.pi * np.concatenate((c -temp, c +temp), axis=-1))), axis=-1)


def gate_adjoint_rep(U: 'np.array[complex]', dim: 'Sequence[int]', only_local: bool = True) -> 'np.array[float]':
    """Adjoint representation of a unitary gate in the hermitian tensor basis.

    Args:
      U: unitary gate
      dim: dimension vector defining the basis
      only_local: if True, only return the local part of the matrix

    Returns:
      adjoint representation of U

    See :cite:`koponen2006`.
    """
    D = len(U)
    if D != np.prod(dim):
        raise ValueError('Dimension of the gate {} does nor match the dimension vector {}.'.format(D, dim))
    # generate the local part of \hat{U}
    B = tensorbasis(dim, d=None, get_locality=False, only_local=only_local)
    n = len(B)
    W = np.empty((n, n), dtype=float)
    for j, y in enumerate(B):
        temp = (U @ y) @ U.T.conj()
        for i, x in enumerate(B):
            # elements of B are hermitian
            W[i, j] = np.trace(x @ temp).real
    return W


def gate_leakage_inv(U: 'np.array[complex]', dim: 'Sequence[int]', Z=None, W=None) -> 'np.array[float]':
    """Local degrees of freedom leaked by a unitary gate.

    Args:
      U: unitary gate
      dim: dimension vector
    Returns:
      cosines of the principal angles between

    TODO FIXME

    See :cite:`koponen2006`.
    """
    #import pdb; pdb.set_trace()
    # generate the local part of \hat{U}
    ULL = gate_adjoint_rep(U, dim, only_local=True)
    M = ULL
    #M = W.T @ ULL @ Z
    u, s, vh = npl.svd(M, full_matrices=False)
    #_, ref, __ = npl.svd(ULL, full_matrices=False)
    #print(s, ref)
    #print(majorize(s, ref[:len(s)]))
    return s, u, vh


def plot_makhlin_2q(
        ax: 'Axes3D' = None,
        perfect_entanglers: bool = True,
        *,
        sdiv: int = 21,
        tdiv: int = 21
) -> 'Axes3D':
    """Plots the set of two-qubit gates in the space of Makhlin invariants.

    Plots the set of two-qubit gates in the space of Makhlin
    invariants (see :func:`makhlin_inv`), returns the Axes3D object.

    Args:
        ax: axes to plot in
        perfect_entanglers: iff True, plot also the set of perfect entanglers
        sdiv, tdiv: number of s and t divisions in the mesh
    Returns:
        plot axes
    """
    # Ville Bergholm 2006-2021

    if ax is None:
        ax = plt.subplot(111, projection='3d')

    s = np.linspace(0, 1,   sdiv)
    t = np.linspace(0, 0.5, tdiv)

    # more efficient than meshgrid
    #g1 = kron(np.cos(s).^2, np.cos(t).^4) - kron(np.sin(s).^2, np.sin(t).^4)
    #g2 = 0.25*kron(np.sin(2*s), np.sin(2*t).^2)
    #g3 = 4*g1 - kron(np.cos(2*s), np.cos(2*t).^2)
    #S = kron(s, ones(size(t)))
    #T = kron(ones(size(s)), t)

    # canonical coordinate plane (s, t, t) gives the entire surface of the set of gate equivalence classes
    S, T = np.meshgrid(s, t)
    c = np.c_[S.ravel(), T.ravel(), T.ravel()]
    G = makhlin_inv(c).reshape((tdiv, sdiv, 3))
    C = gate_max_concurrence(c).reshape((sdiv, tdiv))

    polyc = ax.plot_surface(
        G[:, :, 0], G[:, :, 1], G[:, :, 2],
        rstride=1, cstride=1,
        #cmap = cm.jet,
        #norm = colors.Normalize(vmin=0, vmax=1, clip=True),
        alpha=0.2)
    #polyc.set_array(C.ravel() ** 2)  # FIXME colors

    def plot_triangles(st, plane):
        """Transforms the (s, t) triangles in ``st`` into the given plane in the space
        of canonical invariants, and the transformed triangles into surfaces in the space
        of Makhlin invariants, which are then plotted."""
        for points in st:
            # create a triangle mesh for plotting the surface plane, since
            # in the space of the Makhlin coordinates it is a curved surface
            p = trisurf(points, 0.05)
            print(len(p))
            c = plane(p)
            c = makhlin_inv(c).T
            triang = tri.Triangulation(*p.T)
            ax.plot_trisurf(*c, triangles=triang.triangles, color='r', alpha=0.2)
            #ax.scatter(*c, color='k')

    if perfect_entanglers:
        # Two triangles in the canonical coordinate plane (s, t, 0.5-t)
        # form the part of the surface of the set of perfect entanglers
        # that is not on the surface of the Weyl chamber.
        st = np.array([
            [[0.5, 0], [0.75, 0.25], [0.25, 0.25]],
            [[0, 0], [0.25, 0.25], [-0.25, 0.25]],
        ], dtype=float)
        plot_triangles(st, lambda p: np.c_[p[:, 0], p[:, 1], 0.5 - p[:, 1]])
        # The rest of the surface (also two triangles) is shared with the Weyl chamber.
        st = np.array([
            [[0.5, 0], [0.75, 0.25], [0.25, 0.25]],
            [[0.25, 0.25], [0, 0.5], [-0.25, 0.25]],
        ], dtype=float)
        plot_triangles(st, lambda p: np.c_[p[:, 0], p[:, 1], p[:, 1]])

    ax.set_xlabel('$g_1$')
    ax.set_ylabel('$g_2$')
    ax.set_zlabel('$g_3$')
    ax.set_title('Makhlin stingray')

    # labels
    ax.text(1.05, 0, 2.7, 'I')
    ax.text(-1.05, 0, -2.7, 'SWAP')
    ax.text(-0.1, 0, 1.2, 'CNOT')
    ax.text(0.1, 0, -1.2, 'iSWAP')
    ax.text(0.1, 0.26, 0, 'SWAP$^{1/2}$')
    ax.text(0, -0.26, 0, 'SWAP$^{-1/2}$')

    #fig.colorbar(polyc, ax=ax)
    return ax


def plot_weyl_2q(
        ax: 'Axes3D' = None,
        perfect_entanglers: bool = False
) -> 'Axes3D':
    """Plots the two-qubit Weyl chamber.

    Plots the Weyl chamber for the local invariants
    of 2q gates. See :cite:`Zhang`.

    Args:
        ax: axes to plot in
        perfect_entanglers: iff True, plot also the set of perfect entanglers

    Returns:
        plot axes
    """
    # Ville Bergholm 2005-2021
    if ax is None:
        ax = plt.subplot(111, projection='3d')

    # points: O, A1, A2, A3
    points = np.array([
        [0, 1, 0.5, 0.5],
        [0, 0, 0.5, 0.5],
        [0, 0, 0, 0.5],
    ])
    ax.plot_trisurf(*points, triangles=[[0, 1, 2], [0, 3, 2], [1, 2, 3]], color='g', alpha=0.2)

    if perfect_entanglers:
        # points: L P Q N M A2
        points = np.array([
            [0.5, 0.25, 0.25, 0.75, 0.75, 0.5],
            [0, 0.25, 0.25, 0.25, 0.25, 0.5],
            [0, 0.25, 0, 0.25, 0, 0],
        ])
        ax.plot_trisurf(*points, triangles=[
            [0, 1, 2], [0, 3, 1], [0, 4, 3], [5, 3, 4], [5, 2, 1], [5, 1, 3]
        ], color='r', alpha=0.2, linewidth=0.5, edgecolor='k')

    ax.set_xlabel('$c_1/\\pi$')
    ax.set_ylabel('$c_2/\\pi$')
    ax.set_zlabel('$c_3/\\pi$')
    ax.set_title('Two-qubit Weyl chamber')

    ax.text(-0.05, -0.05, 0, 'I')
    ax.text(1.05, -0.05, 0, 'I')
    ax.text(0.5, 0.51, 0.51, 'SWAP')
    ax.text(0.45, -0.05, 0, 'CNOT')
    ax.text(0.45, 0.55, -0.05, 'iSWAP')
    ax.text(0.20, 0.25, 0.25, 'SWAP$^{1/2}$')
    ax.text(0.75, 0.25, 0.25, 'SWAP$^{-1/2}$')
    ax.text(0.20, 0.25, 0, 'iSWAP$^{1/2}$')
    ax.text(0.75, 0.25, 0, 'iSWAP$^{1/2}$')
    return ax
