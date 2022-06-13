"""
Utilities
=========

.. currentmodule:: qit.utils

This module contains utility functions which do not logically fit anywhere else.


Mathematical utilities
----------------------

.. autosummary::

   gcd
   lcm
   majorize


Matrix functions
----------------

.. autosummary::

   comm
   acomm
   mkron
   tensorsum
   rank
   orth
   nullspace
   nullspace_hermitian
   projector
   eighsort
   expv


Random matrices
---------------

.. autosummary::

   rand_hermitian
   rand_U
   rand_SU
   rand_U1
   rand_pu
   rand_positive
   rand_GL
   rand_SL


Superoperators
--------------

.. autosummary::

   vec
   inv_vec
   lmul
   rmul
   lrmul
   superop_lindblad
   superop_fp
   superop_to_choi

Physics
-------

.. autosummary::

   angular_momentum
   boson_ladder
   fermion_ladder


Bases, decompositions
---------------------

.. autosummary::

   spectral_decomposition
   gellmann
   tensorbasis


Miscellaneous
-------------

.. autosummary::

   copy_memoize
   op_list
   cdot
   qubits
   R_nmr
   Rx
   Ry
   Rz
   trisurf
"""
# Ville Bergholm 2008-2021
from __future__ import annotations

from copy import deepcopy

import numpy as np
import scipy.linalg as spl

from .base import sx, sy, sz, TOLERANCE

__all__ = ['copy_memoize',
           'gcd', 'lcm', 'majorize',
           'comm', 'acomm', 'mkron', 'tensorsum',
           'projector', 'eighsort', 'expv',
           'rank', 'orth', 'nullspace', 'nullspace_hermitian',
           'rand_hermitian', 'rand_U', 'rand_SU', 'rand_U1', 'rand_pu', 'rand_positive', 'rand_GL', 'rand_SL',
           'vec', 'inv_vec', 'lmul', 'rmul', 'lrmul', 'superop_lindblad', 'superop_fp', 'superop_to_choi',
           'angular_momentum', 'boson_ladder', 'fermion_ladder',
           'spectral_decomposition',
           'gellmann', 'tensorbasis',
           'R_nmr', 'Rx', 'Ry', 'Rz',
           'op_list', 'cdot',
           'qubits']


# the functions in this module return ndarrays, not lmaps, for now


# internal utilities

def _warn(s):
    """Prints a warning."""
    print('Warning: ' + s)



def copy_memoize(func):
    """Memoization decorator for functions with immutable args, returns deep copies."""
    cache = {}
    def wrapper(*args):
        """Nonsense, this is an election year."""
        if args in cache:
            value = cache[args]
        else:
            value = func(*args)
            cache[args] = value

        return deepcopy(value)

    # so that the help system still works
    wrapper.__name__ = func.__name__
    wrapper.__doc__  = func.__doc__
    return wrapper



# math functions

def gcd(a, b):
    """Greatest common divisor.

    Args:
        a (int): integer
        b (int): integer

    Returns:
        int: largest positive integer that divides both a and b

    Uses the Euclidean algorithm.
    """
    while b:
        a, b = b, a % b
    return a


def lcm(a, b):
    """Least common multiple.

    Args:
        a (int): integer
        b (int): integer

    Returns:
        int: smallest positive integer that is divided by both a and b
    """
    return a * (b // gcd(a, b))


def rank(A, tol=None):
    """Matrix rank.

    Args:
        A (array[complex]): matrix
        tol (float): numerical tolerance

    Returns:
        int: matrix rank of ``A``, or the number of singular values of ``A`` larger than ``tol``

    """
    s = spl.svdvals(A)
    if tol is None:
        tol = max(A.shape) * np.amax(s) * np.finfo(A.dtype).eps
    return np.sum(s > tol, dtype=int)


def orth(A, tol=None, kernel=False, spectrum=False):
    """Construct an orthonormal basis for the range of A using SVD

    Args:
        A (array[complex]): a (M, N) matrix
        tol (float): numerical tolerance
        kernel (bool): iff True, return a basis for the kernel of A instead of the range of A
        spectrum (bool): iff True, also return the singular values of A

    Returns:
        array[complex]: shape (M, K): orthonormal basis for the range of A

    ``K`` is the effective rank of A, as determined by ``tol``.

    Adaptation of :func:`scipy.linalg.orth` with optional absolute tolerance.
    """
    U, s, Vh = spl.svd(A, full_matrices=False)
    if tol is None:
        tol = max(A.shape) * np.amax(s) * np.finfo(A.dtype).eps
    num = np.sum(s > tol, dtype=int)

    if not kernel:
        # range of A
        ret = U[:, :num]
    else:
        # kernel of A
        ret = Vh[num:, :].conj().transpose()

    if spectrum:
        return ret, s
    return ret


def nullspace(A, tol=None, spectrum=False):
    """Kernel of a matrix.

    Given a matrix A (and optionally a tolerance tol), returns a basis
    for the kernel (null space) of A in the columns of Z.
    """
    # the computation is almost the same as in orth, so...
    return orth(A, tol, kernel=True, spectrum=spectrum)


def nullspace_hermitian(A, tol=None):
    r"""Kernel of a superoperator matrix restricted to the Hermitian subspace.

    Solves the intersection of the kernel of the superop A and the Hermitian subspace.
    A maps d*d matrices to whatever.

    Singular values <= tol are considered zero.
    """
    # Hermitian and antihermitian (orthogonal) real subspaces: V = H \oplus A, h \in H, a \in A
    # If G = A'*A is either block-diagonal or block-antidiagonal wrt these,
    # x = h+a, Gx = \lambda x implies that Gh = \lambda h and Aa = \lambda a.
    # Hence...
    # Ax := [Q, x] is block-antidiagonal if Q is hermitian

    # Ville Bergholm 2011-2012

    def to_real(C):
        """Represents a complex linear map in a real vector space.
        Corresponding transformation for column vectors: x_R = r_[x.real, x.imag]
        """
        return np.r_[np.c_[C.real, -C.imag], np.c_[C.imag, C.real]]

    D = A.shape[1]
    d = int(np.sqrt(D))  # since A is a superop

    # Hermitian basis.
    # reshape uses row-major order, so technically we would need to transpose each matrix,
    # but since they're Hermitian and orthonormal either way it doesn't really matter.
    V = gellmann(d).reshape((D-1, D))
    # Add identity
    V = np.c_[np.eye(d).flatten() / np.sqrt(d), V.transpose()]
    U = np.r_[V.real, V.imag] # same, except now in a real vector space

    # find A restricted to the Hermitian (real) subspace H
    AH = np.dot(to_real(A), U)

    # solve the kernel
    Z, spectrum = nullspace(AH, tol, spectrum=True) # null space (kernel) of AH

    # columns of C: complex orthogonal vectors spanning the kernel (with real coefficients)
    C = np.dot(V, Z)
    return C, spectrum


def projector(v):
    """Projector corresponding to vector v."""
    return np.outer(v, v.conj())


def eighsort(A):
    """Returns eigenvalues and eigenvectors of a Hermitian matrix, sorted in nonincreasing order."""
    d, v = np.linalg.eigh(A)
    ind = d.argsort()[::-1]  # nonincreasing real part
    return d[ind], v[:, ind]


def comm(A, B):
    """Array commutator.

    Args:
      A, B (array):
    Returns:
      array: [A, B] := A*B - B*A
    """
    return A @ B - B @ A


def acomm(A, B):
    """Array anticommutator.

    Args:
      A, B (array):
    Returns:
      array: {A, B} := A*B + B*A
    """
    return A @ B + B @ A


def expv(t, A, v, tol=1.0e-7, m=30, iteration='arnoldi'):
    r"""Multiply a vector by an exponentiated matrix.

    Approximates :math:`exp(t A) v` using a Krylov subspace technique.
    Efficient for large sparse matrices.
    The basis for the Krylov subspace is constructed using either Arnoldi or Lanczos iteration.

    Args:
        t (array[float]): vector of nondecreasing time instances >= 0, ``len(t) == s``
        A (array[complex]): n*n matrix (usually sparse)
        v (array[complex]): n-dimensional vector
        tol (float): tolerance
        m (int): Krylov subspace dimension, ``m <= n``
        iteration (str): iteration type, in ('arnoldi', 'lanczos'). Lanczos is faster but requires a Hermitian ``A``.

    Returns:
        array[complex], float, float: result vectors as a (s, n) array, total truncation error estimate, hump size

    The result is :math:`W[i,:] \approx \exp(t[i] A) v`.
    The hump size is :math:`\max_{s \in [0, t]}  \| \exp(s A) \|`

    Uses the sparse algorithm from :cite:`EXPOKIT`.
    """
    # Ville Bergholm 2009-2012
    # pylint: disable=too-many-locals,too-many-statements

    # just in case somebody tries to use numpy.matrix instances here
    if isinstance(A, np.matrix) or isinstance(v, np.matrix):
        raise ValueError("A and v must be plain numpy.ndarray instances, not numpy.matrix.")

    n = A.shape[0]
    m = min(n, m)  # Krylov space dimension, m <= n

    if np.isscalar(t):
        tt = np.array([t])
    else:
        tt = t

    a_norm = spl.norm(A, np.inf)
    v_norm = spl.norm(v)

    happy_tol  = 1.0e-7  # "happy breakdown" tolerance
    min_error = a_norm * np.finfo(float).eps # due to roundoff

    # step size control
    max_stepsize_changes = 10
    # safety factors
    gamma = 0.9
    delta = 1.2
    # initial stepsize
    fact = np.sqrt(2 * np.pi * (m + 1)) * ((m + 1) / np.exp(1)) ** (m + 1)

    def ceil_at_nsd(x, n = 2):
        temp = 10 ** (np.floor(np.log10(x))-n+1)
        return np.ceil(x / temp) * temp

    def update_stepsize(step, err_loc, r):
        step *= gamma  * (tol * step / err_loc) ** (1 / r)
        return ceil_at_nsd(step, 2)

    dt = np.dtype(complex)
    # TODO don't use complex matrices unless we have to: dt = result_type(t, A, v)

    # TODO shortcuts for Hessenberg matrix exponentiation?
    # upper Hessenberg matrix for the Arnoldi process + two extra rows/columns for the error estimate trick
    H = np.zeros((m+2, m+2), dt)
    H[m + 1, m] = 1           # never overwritten!
    V = np.zeros((n, m+1), dt)   # orthonormal basis for the Krylov subspace + one extra vector

    W = np.empty((len(tt), len(v)), dt)  # results
    t = 0  # current time
    beta = v_norm
    error = 0  # error estimate
    hump = [[v_norm, t]]
    #v_norm_max = v_norm  # for estimating the hump

    def iterate_lanczos(v, beta):
        """Lanczos iteration, for Hermitian matrices.
        Produces a tridiagonal H, cheaper than Arnoldi.

        Returns the number of basis vectors generated, and a boolean indicating a happy breakdown.
        NOTE that the we _must_not_ change global variables other than V and H here
        """
        # beta_0 and alpha_m are not used in H, beta_m only in a single position for error control
        prev = 0
        for k in range(0, m):
            vk = (1 / beta) * v
            V[:, k] = vk  # store the now orthonormal basis vector
            # construct the next Krylov vector beta_{k+1} v_{k+1}
            v = np.dot(A, vk)
            H[k, k] = alpha = np.vdot(vk, v)
            v += -alpha * vk -beta * prev
            # beta_{k+1}
            beta = spl.norm(v)
            if beta < happy_tol: # "happy breakdown": iteration terminates, Krylov approximation is exact
                return k+1, True
            if k == m-1:
                # v_m and one beta_m for error control (alpha_m not used)
                H[m, m-1] = beta
                V[:, m] = (1 / beta) * v
            else:
                H[k+1, k] = H[k, k+1] = beta
                prev = vk
        return m+1, False

    def iterate_arnoldi(v, beta):
        """Arnoldi iteration, for generic matrices.
        Produces a Hessenberg-form H.
        """
        V[:, 0] = (1 / beta) * v  # the first basis vector v_0 is just v, normalized
        for j in range(1, m+1):
            p = np.dot(A, V[:, j-1])  # construct the Krylov vector v_j
            # orthogonalize it with the previous ones
            for i in range(j):
                H[i, j-1] = np.vdot(V[:, i], p)
                p -= H[i, j-1] * V[:, i]
            temp = spl.norm(p)
            if temp < happy_tol: # "happy breakdown": iteration terminates, Krylov approximation is exact
                return j, True
            # store the now orthonormal basis vector
            H[j, j-1] = temp
            V[:, j] = (1 / temp) * p
        return m+1, False  # one extra vector for error control

    # choose iteration type
    iteration = iteration.lower()
    if iteration == 'lanczos':
        iteration = iterate_lanczos  # only works for Hermitian matrices!
    elif iteration == 'arnoldi':
        iteration = iterate_arnoldi
    else:
        raise ValueError("Only 'arnoldi' and 'lanczos' iterations are supported.")

    # loop over the time instances (which must be in increasing order)
    for kk, t_end in enumerate(tt):
        # initial stepsize
        # TODO we should inherit the stepsize from the previous interval
        r = m
        t_step = (1 / a_norm) * ((fact * tol) / (4 * beta * a_norm)) ** (1 / r)
        t_step = ceil_at_nsd(t_step, 2)

        while t < t_end:
            t_step = min(t_end - t, t_step)  # step at most the remaining distance

            # Arnoldi/Lanczos iteration, (re)builds H and V
            j, happy = iteration(v, beta)
            # now V^\dagger A V = H  (just the first m vectors, or j if we had a happy breakdown!)
            # assert(spl.norm(np.dot(np.dot(V[:, :m].conj().transpose(), A), V[:, :m]) -H[:m,:m]) < tol)

            # error control
            if happy:
                # "happy breakdown", using j Krylov basis vectors
                t_step = t_end - t  # step all the rest of the way
                F = spl.expm(t_step * H[:j, :j])
                err_loc = happy_tol
                r = m
            else:
                # no happy breakdown, we need the error estimate (using all m+1 vectors)
                av_norm = spl.norm(np.dot(A, V[:, m]))
                # find a reasonable step size
                for k in range(max_stepsize_changes + 1):
                    F = spl.expm(t_step * H)
                    err1 = beta * abs(F[m, 0])
                    err2 = beta * abs(F[m+1, 0]) * av_norm
                    if err1 > 10 * err2:  # quick convergence
                        err_loc = err2
                        r = m
                    elif err1 > err2:  # slow convergence
                        err_loc = (err2 * err1) / (err1 - err2)
                        r = m
                    else:  # asymptotic convergence
                        err_loc = err1
                        r = m-1
                    # should we accept the step?
                    if err_loc <= delta * tol * t_step:
                        break
                    if k >= max_stepsize_changes:
                        raise RuntimeError(f'Requested tolerance cannot be achieved in {max_stepsize_changes} stepsize changes.')
                    t_step = update_stepsize(t_step, err_loc, r)

            # step accepted, update v, beta, error, hump
            v = np.dot(V[:, :j], beta * F[:j, 0])
            beta = spl.norm(v)
            error += max(err_loc, min_error)
            #v_norm_max = max(v_norm_max, beta)

            t += t_step
            t_step = update_stepsize(t_step, err_loc, r)
            hump.append([beta, t])

        W[kk, :] = v

    hump = np.array(hump) / v_norm
    return W, error, hump



# random matrices

def rand_hermitian(n):
    """Random Hermitian n*n matrix.

    Args:
        n (int): matrix size

    Returns:
        array[complex]: random Hermitian n*n matrix

    .. note:: The randomness is not defined in any deeply meaningful sense.
    """
    H = (np.random.rand(n, n) - 0.5) +1j * (np.random.rand(n, n) - 0.5)
    return H + H.conj().transpose() # make it Hermitian


def rand_U(n):
    """Random U(n) matrix.

    Args:
        n (int): matrix size

    Returns:
        array[complex]: random unitary n*n matrix

    The matrix is random with respect to the Haar measure.
    Uses the algorithm in :cite:`Mezzadri`.
    """
    # sample the Ginibre ensemble, p(Z(i,j)) == 1/pi * exp(-abs(Z(i,j))^2),
    # p(Z) == 1/pi^(n^2) * exp(-trace(Z'*Z))
    Z = (np.random.randn(n, n) + 1j*np.random.randn(n, n)) / np.sqrt(2)

    # QR factorization
    Q, R = np.linalg.qr(Z)

    # eliminate multivaluedness in Q
    P = np.diag(R).copy()  # TODO remove .copy() once numpy supports read/write views
    P /= abs(P)
    return np.dot(Q, np.diag(P))


def rand_SU(n):
    """Random SU(n) matrix.

    Args:
        n (int): matrix size

    Returns:
        array[complex]: random special unitary n*n matrix

    The matrix is random with respect to the Haar measure.
    """
    U = rand_U(n)
    d = np.linalg.det(U) ** (1/n) # *np.exp(i*2*np.pi*k/n), not unique FIXME
    return U/d


def rand_U1(n):
    """Random diagonal unitary matrix.

    Args:
        n (int): matrix size

    Returns:
        array[complex]: random diagonal unitary n*n matrix

    The matrix is random with respect to the Haar measure.
    """
    return np.diag(np.exp(2j * np.pi * np.random.rand(n)))


def rand_pu(n):
    r"""Random n-partition of unity.

    Args:
        n (int): number of parts

    Returns:
        array[float]: n-partition, i.e. a 1d array of ``n`` nonnegative numbers that sum up to one

    The n-partition is random with respect to
    the order-n Dirichlet distribution :math:`Dir(\alpha)`
    with :math:`\alpha = (1, 1, ..., 1)`.
    """
    # Sample the Dirichlet distribution using n gamma-distributed
    # random vars. The (shared) scale parameter of the gamma pdfs is irrelevant,
    # and the shape parameters correspond to the Dirichlet \alpha params.
    # Furthermore, Gamma(x; 1,1) = exp(-x), so
    p = -np.log(np.random.rand(n))  # Gamma(1,1) -distributed
    return p / sum(p)  # Dir(1, 1, ..., 1) -distributed

    # TODO this would be a simpler choice, but what's the exact distribution?
    #p = np.sort(np.random.rand(n-1))  # n-1 points in [0,1]
    #d = np.sort(np.r_[p, 1] - np.r_[0, p])  # n deltas between points = partition of unity


def rand_positive(n):
    """Random n*n positive semidefinite matrix.

    Args:
        n (int): matrix size

    Returns:
        array[complex]: random positive semidefinite matrix with trace 1

    Since the returned matrix has all-real eigenvalues, it is Hermitian by construction.
    """
    d = rand_pu(n) # random partition of unity
    U = rand_U(n) # random unitary
    A = U.T.conj() @ (np.diag(d) @ U)
    return np.array((A + A.T.conj()) / 2)   # eliminate rounding errors
    # An alternative would be to use an inverse purification, but this would be expensive.


def rand_GL(n):
    r"""Random GL(n, C) matrix.

    Args:
        n (int): matrix size

    Returns:
        array[complex]: random general linear n*n matrix, with normally distributed elements.

    The :math:`\det S \neq 0` condition is for now fulfilled only statistically,
    i.e. you almost never obtain a non-invertible matrix.
    """
    A = np.random.randn(n, n) +1j * np.random.randn(n, n)
    return A


def rand_SL(n):
    """Random SL(n, C) matrix.

    Args:
        n (int): matrix size

    Returns:
        array[complex]: random special linear n*n matrix

    .. note:: The randomness is not defined in any deeply meaningful sense.
    """
    S = rand_GL(n)
    d = np.linalg.det(S) ** (1/n)
    return S/d



# superoperators

def vec(rho):
    """Flattens a matrix into a vector.

    Matrix rho is flattened columnwise into a column vector v.

    Used e.g. to convert state operators to superoperator representation.
    """
    # JDW 2009
    # Ville Bergholm 2009

    return rho.flatten('F')  # copy


def inv_vec(v, dim=None):
    r"""Reshapes a vector into a matrix.

    Reshapes vector v into a matrix :math:`\rho` (shape == dim),
    using column-major ordering. If dim is not given, :math:`\rho` is assumed to be square.

    Used e.g. to convert state operators from superoperator representation
    to standard matrix representation.

    Args:
      v   (array): vector to be reshaped, len == m*n
      dim (Sequence[int]): (m, n) == shape of the matrix
    """
    # JDW 2009
    # Ville Bergholm 2009
    d = v.size
    if dim is None:
        # assume a square matrix
        n = int(np.sqrt(d))
        if n**2 != d:
            raise ValueError('Length of vector v is not a squared integer.')
        dim = (n, n)
    elif np.prod(dim) != d:
        raise ValueError('Dimensions are not compatible with given vector.')
    return v.reshape(dim, order='F').copy()


def lmul(L, q=None):
    r"""Superoperator equivalent for multiplying from the left.

    .. math:: L \rho = \text{inv_vec}(\text{lmul}(L) \text{vec}(\rho))

    Args:
      L (array): multiplying operator, shape == (m, p)
      q   (int): The shape of :math:`\rho` is (p, q). If q is not given, :math:`\rho` is assumed square.
    Returns:
      array: superoperator that implements left multiplication of a vectorized matrix :math:`\rho` by the matrix L
    """
    if q is None:
        q = L.shape[1]  # assume target is a square matrix
    return np.kron(np.eye(q), L)


def rmul(R, p=None):
    r"""Superoperator equivalent for multiplying from the right.

    .. math:: \rho R = \text{inv_vec}(\text{rmul}(R) \text{vec}(\rho))

    Args:
      R (array): multiplying operator, shape == (q, r)
      p   (int): The shape of :math:`\rho` is (p, q). If p is not given, :math:`\rho` is assumed square.
    Returns:
      array: superoperator that implements right multiplication of a vectorized matrix :math:`\rho` by the matrix R
    """
    if p is None:
        p = R.shape[0]  # assume target is a square matrix
    return np.kron(R.transpose(), np.eye(p))


def lrmul(L, R):
    r"""Superoperator equivalent for multiplying both from left and right.

    .. math:: L \rho R = \text{inv_vec}(\text{lrmul}(L, R) \text{vec}(\rho))

    Args:
      L (array): matrix, shape (m, p)
      R (array): matrix, shape (q, r)
    Returns:
      array: superoperator that implements left multiplication by the matrix L and right
        multiplication by the matrix R of a vectorized matrix :math:`\rho`
    """
    # L and R fix the shape of rho completely
    return np.kron(R.transpose(), L)


def superop_lindblad(A, H=None):
    r"""Liouvillian superoperator for a set of Lindblad operators.

    Args:
      A (Sequence[array]): sequence of traceless, orthogonal Lindblad operators
      H (array): optional Hamiltonian operator
    Returns:
      array: Liouvillian superoperator L

    L corresponding to the diagonal-form Lindblad equation

    .. math::

       \dot{\rho} = \text{inv_vec}(L \text{vec}(\rho))
       = -i [H, \rho] +\sum_k \left(A_k \rho A_k^\dagger -\frac{1}{2} \{A_k^\dagger A_k, \rho\}\right)
    """
    # James D. Whitfield 2009
    # Ville Bergholm 2009-2010

    # Hamiltonian
    if H is None:
        sh = A[0].shape
        iH = 0
    else:
        sh = H.shape
        iH = 1j * H

    L = np.zeros(np.array(sh) ** 2, dtype=complex)
    ac = np.zeros(sh, dtype=complex)
    for k in A:
        ac += 0.5 * np.dot(k.conj().transpose(), k)
        L += lrmul(k, k.conj().transpose())

    L += lmul(-ac -iH) +rmul(-ac +iH)
    return L


def superop_fp(L, tol=None):
    r"""Fixed point states of a Liouvillian superoperator.

    Finds the intersection of the kernel of the Liouvillian L
    with the set of valid state operators, thus giving the set of
    fixed point states for the quantum channel represented by the
    master equation

    .. math:: \dot{\rho} = \text{inv_vec}(L \text{vec}(\rho)).

    Let L.shape == (D, D) (and d = sqrt(D) be the dimension of the Hilbert space).

    Returns the D*n array A, which contains as its columns a set
    of n vectorized orthogonal Hermitian matrices (with respect to
    the Hilbert-Schmidt inner product) which "span" the set of FP
    states in the following sense:

    .. math:: vec(\rho) = A c,  \quad \text{where} \quad c \in \text{R}^n \quad \text{and} \quad c_1 = 1.

    A[:,0] is the shortest vector in the Hermitian kernel of L that
    has trace 1, the other columns of A are traceless and normalized.
    Hence, A defines an (n-1)-dimensional hyperplane.

    A valid state operator also has to fulfill :math:`\rho \ge 0`.
    These operators form a convex set in the Hermitian trace-1 hyperplane
    defined by A. Currently this function does nothing to enforce
    positivity, it is up to the user to choose the coefficients :math:`a_k`
    such that this condition is satisfied.

    Args:
      L (array): Liouvillian superoperator
      tol (float): tolerance, singular values of L that are <= tol are treated as zero
    Returns:
      array, : array containing n vectorized Hermitian matrices as its columns, shape == (D, n)
    """
    #If L has Lindblad form, if L(rho) = \lambda * rho,
    #we also have L(rho') = conj(\lambda) * rho'
    #Hence the non-real eigenvalues of L come in conjugate pairs.
    #Especially if rho \in Ker L, also rho' \in Ker L.

    # Ville Bergholm 2011-2012

    # Hilbert space dimension
    d = int(np.sqrt(L.shape[1]))

    # Get the kernel of L, restricted to vec-mapped Hermitian matrices.
    # columns of A: complex orthogonal vectors A_i spanning the kernel (with real coefficients)
    A, spectrum = nullspace_hermitian(L, tol)

    # Extract the trace-1 core of A and orthonormalize the rest.
    # We want A_i to be orthonormal wrt. the H-S inner product
    # <X, Y> := trace(X' * Y) = vec(X)' * vec(Y)

    # With this inner product, trace(A) = <eye(d), A>.
    temp = vec(np.eye(d, d))
    a = np.dot(temp, A)  # a_i = trace(A_i)

    # Construct the shortest linear combination of A_i which has tr = 1.
    # This is simply (1/d) * eye(d) _iff_ it belongs to span(A_i).
    core = np.dot(A, (a / spl.norm(a)**2))  # trace-1

    # Remove the component parallel to core from all A_i.
    temp = core / spl.norm(core) # normalized
    A = A -np.outer(temp, np.dot(temp.conj(), A))

    # Re-orthonormalize the vectors, add the core
    ttt = orth(A, 1e-6)
    A = np.c_[core, ttt]  # TODO tolerance?

    N = A.shape[1]
    for k in range(N):
        temp = inv_vec(A[:, k])
        A[:, k] = vec(0.5 * (temp + temp.conj().transpose())) # re-Hermitize to fix numerical errors

    # TODO intersect with positive ops

    #[u,t] = schur(L)
    #unnormality = spl.norm(t, 'fro')^2 -spl.norm(np.diag(t))^2
    #[us, ts] = ordschur(u, t, 'udi')
    #E = ordeig(t)
    # TODO eigendecomposition, find orthogonal complement to span(v) = ker(v').
    # these are the states which do not belong to eigenspaces and
    # should show transient polynomial behavior

    return A, spectrum


def superop_to_choi(L):
    r"""Convert a Liouvillian superoperator to a Choi matrix.

    Given a Liouvillian superoperator L operating on vectorized state
    operators, :math:`L: A \otimes A \to B \otimes B`, returns the
    corresponding Choi matrix :math:`C: A \otimes B \to A \otimes B`.

    Args:
      L (array): Liouvillian superoperator, shape == (b**2, a**2)
    Returns:
      array: corresponding Choi matrix, shape == (a*b, a*b)
    """
    # input and output Hilbert space dimensions
    dim = np.sqrt(L.shape)
    A = int(round(dim[1]))
    B = int(round(dim[0]))
    D = A * B

    # reshape uses little-endian ordering, whereas we use big-endian
    temp = L.reshape((B, B, A, A))
    # swap the more significant output and less significant input indices
    temp = temp.transpose((3, 1, 2, 0))
    # back to a flat matrix
    return temp.reshape((D, D))



# physical operators

@copy_memoize
def angular_momentum(d):
    r"""Angular momentum matrices.

    Returns the dimensionless angular momentum matrices :math:`\vec{J} / \hbar`
    for the d-dimensional subspace.

    The angular momentum matrices fulfill the commutation relation
    :math:`[J_x, J_y] = i J_z` and all its cyclic permutations.

    :math:`\exp(-1i * J[k] * \theta)` is a rotation by :math:`\theta` radians.

    Args:
      d (int): dimension, the corresponding angular momentum quantum number is j = (d-1)/2
    Returns:
      tuple: 3-tuple of angular momentum matrices (Jx, Jy, Jz)
    """
    # Ville Bergholm 2009-2017

    if d < 1:
        raise ValueError('Dimension must be one or greater.')

    j = (d - 1) / 2 # angular momentum quantum number, d == 2*j + 1
    # raising operator in subspace J^2 = j*(j+1)
    m = j
    Jplus = np.zeros((d, d))
    for k in range(d-1):
        m -= 1
        Jplus[k, k+1] = np.sqrt(j*(j+1) -m*(m+1))

    # lowering operator
    Jminus = Jplus.conj().transpose()
    # Jplus  = Jx + i*Jy
    # Jminus = Jx - i*Jy
    return (0.5*(Jplus + Jminus), 0.5j*(Jminus - Jplus), np.diag(np.arange(j, -j-1, -1)))


@copy_memoize
def boson_ladder(d):
    r"""Bosonic ladder operators.

    Args:
      d (int): truncation dimension
    Returns:
      array: bosonic annihilation operator :math:`b`

    Returns the d-dimensional approximation of the bosonic
    annihilation operator :math:`b` for a single bosonic mode in the
    number basis :math:`\{\ket{0}, \ket{1}, ..., \ket{d-1}\}`.

    The corresponding creation operator is :math:`b^\dagger`.
    The (infinite-dimensional) bosonic annihilation and creation operators fulfill the commutation relation
    :math:`[b, b^\dagger] = \I`. Due to the d-dimensional basis truncation,
    this does not hold for the last dimension of the b matrices returned by this function.
    """
    return np.diag(np.sqrt(range(1, d)), 1)


@copy_memoize
def fermion_ladder(n):
    r"""Fermionic ladder operators.

    Args:
      n (int): number of fermionic modes
    Returns:
      array[object]: fermionic annihilation operators :math:`(f_1, f_2, \ldots, f_n)`

    Returns the fermionic annihilation operators for a
    system of n fermionic modes in the second quantization.

    The annihilation operators are built using the Jordan-Wigner
    transformation for a chain of n qubits, where the state of each
    qubit denotes the occupation number of the corresponding mode.
    First define annihilation and number operators for a lone fermion mode:

    .. math::

      \sigma^- &:= (\sigma_x + i \sigma_y)/2,\\
      n &:= \sigma^+ \sigma^- = (\I-\sigma_z)/2,\\
      &\sigma^-\ket{0} = 0, \quad \sigma^-\ket{1} = \ket{0}, \quad n\ket{k} = k\ket{k}.

    Then define a phase operator to keep track of sign changes when
    permuting the order of the operators: :math:`\phi_k := \sum_{j=1}^{k-1} n_j`.
    Now, the fermionic annihilation operators for the n-mode system are given by

    .. math::

      f_k := (-1)^{\phi_k} \sigma^-_k = \left(\bigotimes_{j=1}^{k-1} {\sigma_z}_j \right) \sigma^-_k.

    These operators fulfill the required anticommutation relations:

    .. math::

       \{f_j, f_k\}  &= 0,\\
       \{f_j, f_k^\dagger\} &= \I \delta_{jk},\\
       f_k^\dagger f_k &= n_k.
    """
    s = np.array([[0, 1], [0, 0]]) # single annihilation op
    temp = 1

    # empty array for the annihilation operators
    f = np.empty(n, dtype='object')

    # Jordan-Wigner transform
    for k in range(n):
        f[k] = mkron(temp, s, np.eye(2 ** (n-k-1)))
        temp = np.kron(temp, sz)

    return f



# SU(2) rotations

def R_nmr(theta, phi):
    r"""SU(2) rotation :math:`\theta_\phi` (NMR notation).

    One-qubit rotation by angle theta about the unit
    vector :math:`[\cos(\phi), \sin(\phi), 0]`, or :math:`\theta_\phi` in the NMR notation.

    Args:
      theta (float): rotation angle
      phi   (float): angle between the rotation axis and the positive x axis
    Returns:
      array[complex]: rotation matrix
    """
    return spl.expm(-1j * theta/2 * (np.cos(phi) * sx + np.sin(phi) * sy))


def Rx(theta):
    r"""SU(2) x-rotation.

    One-qubit rotation about the x axis by the angle theta.

    Args:
      theta (float): rotation angle
    Returns:
      array[complex]: rotation matrix :math:`e^{-i \sigma_x \theta/2}`
    """
    t = theta / 2
    c = np.cos(t)
    _js = -1j * np.sin(t)
    return np.array([[c, _js], [_js, c]])


def Ry(theta):
    r"""SU(2) y-rotation.

    One-qubit rotation about the y axis by the angle theta.

    Args:
      theta (float): rotation angle
    Returns:
      array[complex]: rotation matrix :math:`e^{-i \sigma_y \theta/2}`
    """
    t = theta / 2
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s], [s, c]])


def Rz(theta):
    r"""SU(2) z-rotation.

    One-qubit rotation about the z axis by the angle theta.

    Args:
      theta (float): rotation angle
    Returns:
      array[complex]: rotation matrix :math:`e^{-i \sigma_z \theta/2}`
    """
    return np.array([[np.exp(-1j * theta/2), 0], [0, np.exp(1j * theta/2)]])



# decompositions

def spectral_decomposition(
        A: 'array[complex]',
        tol: float = TOLERANCE
) -> tuple['array[float]', list['array[complex]']]:
    r"""Spectral decomposition of a Hermitian matrix.

    Args:
        A: Hermitian matrix to decompose
        tol: numerical tolerance

    Returns:
        unique eigenvalues ``a`` and the corresponding projectors ``P``
        for ``A``, such that :math:`A = \sum_k  a_k P_k`.
    """
    # Ville Bergholm 2010

    d, v = eighsort(A)
    d = d.real  # A is assumed Hermitian

    # combine projectors for degenerate eigenvalues
    a = [d[0]]
    P = [projector(v[:, 0])]
    for k in range(1, len(d)):
        temp = projector(v[:, k])
        if abs(d[k] - d[k-1]) > tol:
            # new eigenvalue, new projector
            a.append(d[k])
            P.append(temp)
        else:
            # same eigenvalue, extend current P
            P[-1] += temp
    return np.array(a), P



# tensor bases

@copy_memoize
def gellmann(d):
    r"""Gell-Mann matrices.

    Args:
        d (int): dimension

    Returns:
        array[complex]: the ``d**2 - 1`` (traceless, Hermitian) Gell-Mann matrices of dimension d

    The Gell-Mann matrices are normalized such that :math:`\mathrm{Tr}(G_i^\dagger G_j) = \delta_{ij}`.
    They form an orthonormal basis for the real vector space of ``d*d`` traceless Hermitian matrices.

    The matrices are returned in a 3d array ``A``, arranged such that the n:th Gell-Mann matrix is ``A[n,:,:]``.
    """
    # Ville Bergholm 2006-2014

    if d < 1:
        raise ValueError('Dimension must be >= 1.')

    G = np.zeros((d**2 - 1, d, d), dtype=complex)

    # diagonal
    ddd = np.zeros(d)
    ddd[0] = 1
    x = 1 / np.sqrt(2)
    # iterate through the lower triangle
    n = 0
    for k in range(1, d):
        for j in range(0, k):
            # nondiagonal
            G[n, k, j] = x
            G[n, j, k] = x
            n += 1

            G[n, k, j] =  1j * x
            G[n, j, k] = -1j * x
            n += 1
        ddd[k] = -sum(ddd)
        G[n, :, :] = np.diag(ddd) / spl.norm(ddd)
        ddd[k] = 1
        n += 1

    return G


# TODO lazy evaluation/cache purging would be nice here to control memory usage
tensorbasis_cache = {}
def tensorbasis(n, d=None, get_locality=False, only_local=False):
    r"""Hermitian tensor-product basis for End(H).

    Returns a Hermitian basis for linear operators on the Hilbert space H
    which shares H's tensor product structure. The basis elements are tensor products
    of Gell-Mann matrices (which in the case of qubits are equal to Pauli matrices).
    The basis elements are normalized such that :math:`\mathrm{Tr}(b_i^\dagger b_j) = \delta_{ij}`.

    Input is either two scalars, n and d, in which case the system consists of n qu(d)its, :math:`H = C_d^{\otimes n}`,
    or the vector dim, which contains the dimensions of the individual subsystems:
    :math:`H = C_{dim[0]} \otimes ... \otimes C_{dim[n-1]}`.

    In addition to expanding Hermitian operators on H, this basis can be multiplied by
    the imaginary unit to obtain the antihermitian generators of U(prod(dim)).

    Args:
      n (int, Sequence[int]): number of subsystems, or if d is None, a dimension vector
      d (int, None): dimension of the qudits constituting the subsystems
      get_locality (bool): if True, return also the locality of the basis elements
      only_local (bool): if True, only return basis elements whose locality==1
    Returns:
      array[array[complex]], array[int]: basis elements, locality

    The second output variable is an integer array denoting the locality
    of each basis element, i.e. the number of non-identity matrices
    in the tensor product.
    """
    # Ville Bergholm 2005-2016
    # pylint: disable=too-many-locals
    if d is None:
        # dim vector
        dim = tuple(n)
        n = len(dim)
    else:
        # n qu(d)its
        dim = (n,) * d

    # check cache first
    if dim in tensorbasis_cache:
        B   = tensorbasis_cache[dim][0]
        loc = tensorbasis_cache[dim][1]
        if only_local:
            ind = np.where(loc == 1)
            B = B[ind]
            loc = loc[ind]

        if get_locality:
            # tuple: matrices, locality
            return deepcopy((B, loc))
        # just the matrices
        return deepcopy(B)

    n_elements = np.array(dim) ** 2    # number of basis elements for each subsystem, incl. identity
    if only_local:
        n_all = np.sum(n_elements-1)  # just the local tensor basis elements, no identity
    else:
        n_all = np.prod(n_elements) # number of all tensor basis elements, incl. identity

    B = np.empty(n_all, dtype='object')
    locality = np.zeros(n_all, dtype=int)  # number of non-identity terms in the corresponding basis element
    # create the tensor basis
    if only_local:
        # creating the local elements like this is cheaper than creating the entire basis and filtering out just the local ones
        k = 0
        for j in range(n):
            # identities before and after
            d = int(np.prod(dim[:j]))
            I_before = np.eye(d) / np.sqrt(d)
            d = int(np.prod(dim[j+1:]))
            I_after = np.eye(d) / np.sqrt(d)
            for i in range(1, n_elements[j]):
                temp = gellmann(dim[j])[i-1]
                B[k] = mkron(I_before, temp, I_after)
                locality[k] = 1
                k += 1
    else:
        for k in range(n_all):  # loop over all basis elements
            inds = np.unravel_index(k, n_elements)
            temp = 1 # basis element being built
            nonid = 0  # number of non-id. matrices included in this element

            for j in range(n):  # loop over subsystems
                ind = inds[j]   # which local basis element to use
                d = dim[j]
                if ind > 0:
                    nonid += 1 # using a non-identity matrix
                    L = gellmann(d)[ind-1]  # Gell-Mann basis vector for the subsystem
                    # TODO gellmann copying the entire basis for a single matrix is inefficient...
                else:
                    L = np.eye(d) / np.sqrt(d)  # identity (scaled)
                temp = np.kron(temp, L)  # tensor in another matrix
            B[k] = (temp)
            locality[k] = nonid  # at least two non-identities => nonlocal element

        # store into cache
        tensorbasis_cache[dim] = deepcopy((B, locality))

    if get_locality:
        return B, locality
    return B



# misc

def op_list(G, dim):
    r"""Operator consisting of k-local terms, given as a list.

    Args:
        G (list[list[tuple]]): list of k-local operator terms
        dim (Sequence[int]): vector of subsystem dimensions
    Returns:
        array[complex]: matrix defined by G

    G is a list of lists, :math:`G = [c_1, c_2, ..., c_n]`.
    Each list :math:`c_i` corresponds to a term in the operator,
    and consists of k two-tuples: :math:`c_i` = [(A1, s1), (A2, s2), ... , (Ak, sk)].
    Aj are arrays and sj subsystem indices, corresponds to the
    k-local term given by the tensor product

    .. math:: A_1^{(s_1)} A_2^{(s_2)} \cdots A_k^{(s_k)}.

    The dimensions of all operators acting on subsystem sj must match dim[sj].

    Alternatively one can think of G as defining a hypergraph, where
    each subsystem corresponds to a vertex and each array c_i in the list
    describes a hyperedge connecting the vertices {s1, s2, ..., sk}.

    Example: The connection list

    ::
        G = [[(sz,1)], [(sx,1), (sx,3)], [(sy,1), (sy,3)], [(sz,1), (sz,3)], [(sz,2)], [(A,2), (B+C,3)], [(2*sz,3)]]

    corresponds to the operator

    .. math::

       \sigma_{z1} +\sigma_{z2} +2 \sigma_{z3} +\sigma_{x1} \sigma_{x3} +\sigma_{y1} \sigma_{y3} +\sigma_{z1} \sigma_{z3} +A_2 (B+C)_3.
    """
    # Ville Bergholm 2009-2017

    # TODO we could try to infer dim from the operators
    D = np.prod(dim)
    H = np.zeros((D, D), dtype=complex)
    for k, spec in enumerate(G):
        a = -1  # last subsystem taken care of
        term = 1
        for j, op in enumerate(spec):
            if len(op) != 2:
                raise ValueError('Malformed local term {0} in spec {1}.'.format(j, k))

            b = op[1]  # subsystem number
            if b <= a:
                raise ValueError('Spec {0} not in ascending order.'.format(k))
            if b >= len(dim):
                raise ValueError('Operator {0} of spec {1} is applied to a non-existent subsystem.'.format(j, k))
            if op[0].shape[1] != dim[b]:
                raise ValueError('The dimension of operator {0} in spec {1} does not match dim.'.format(j, k))

            term = mkron(term, np.eye(int(np.prod(dim[a+1:b]))), op[0])
            a = b
        # final identity
        term = mkron(term, np.eye(int(np.prod(dim[a+1:]))))
        H += term
    return H


def cdot(v, A):
    r"""Real dot product of a vector and a tuple of operators.

    Args:
      v (array): vector
      A (Iterable[array]): arrays of equal shape
    Returns:
      array: :math:`\sum_k v_k A_k`
    """
    res = 0j
    for vv, AA in zip(v, A):
        res += vv * AA
    return res


def qubits(n):
    """Dimension vector for an all-qubit system.

    Args:
      n (int): number of qubits
    Returns:
      tuple: (2,) * n
    """
    return (2,) * n


def majorize(x: 'array[float]', y: 'array[float]', tol: float = TOLERANCE) -> bool:
    r"""Majorization partial order of real vectors.

    Returns True iff x is majorized by y, denoted by :math:`x \preceq y`. This is equivalent to

    .. math::
       \sum_{k=1}^n x^{\downarrow}_k \le \sum_{k=1}^n y^{\downarrow}_k \quad \text{for all} \quad n \in \{1, 2, \ldots, d\},

    where :math:`x^{\downarrow}` is the vector x with the elements sorted in nonincreasing order.

    :math:`x \preceq y` if and only if x is in the convex hull of all the coordinate permutations of y.

    Args:
        x, y: real vectors, equal length
        tol: numerical tolerance
    Returns:
        :math:`x \preceq y`
    """
    if x.ndim != 1 or y.ndim != 1 or np.iscomplexobj(x) or np.iscomplexobj(y):
        raise ValueError('Inputs must be real vectors.')

    if len(x) != len(y):
        raise ValueError('The vectors must be of equal length.')

    x = np.cumsum(np.sort(x)[::-1])
    y = np.cumsum(np.sort(y)[::-1])

    if abs(x[-1] -y[-1]) <= tol:
        # exact majorization
        return all(x-y <= tol)
    # weak majorization could still be possible, but...
    _warn('Vectors have unequal sums.')
    return False


def mkron(*arg):
    r"""This is how kron should work, dammit.

    Args:
      A, B, C... (array): arbitrary numpy arrays
    Returns:
      array: tensor (Kronecker) product of the input arrays, :math:`A \otimes B \otimes C \otimes \ldots`
    """
    X = 1
    for A in arg:
        X = np.kron(X, A)
    return X


def tensorsum(*arg):
    r"""Like kron but adding instead of multiplying.

    Args:
      A, B, C... (array): arbitrary numpy arrays
    Returns:
      array: tensor sum of the input arrays
    """
    #c = np.log(np.kron(np.exp(a), np.exp(b))) # a perverted way of doing it, the exp overflows...
    A = np.asarray(arg[0])
    for B in arg[1:]:
        B = np.asarray(B)
        # number of dims must match, pad the shorter dim vector with singletons from the left
        nda = A.ndim
        ndb = B.ndim
        a = A.shape
        b = B.shape
        if ndb > nda:
            a = (1,)*(ndb-nda) + a
            nd = ndb
        else:
            b = (1,)*(nda-ndb) + b
            nd = nda
        # outer sum
        A = np.add.outer(A,B).reshape(a + b)
        # collapse the extra dimensions
        for _ in range(nd):
            A = np.concatenate(A, axis=nd-1)
    return A


def trisurf(p: 'np.array', max_dist: float) -> 'np.array':
    """Populates a 2D triangular surface with evenly spaced points for Delaunay triangulation.

    Args:
        p: vertices of the triangle, ``p.shape = (3, 2)``
        max_dist: maximum 2-norm distance between the created points
    """
    def samples(a, b):
        """Find the number of samples between points a and b, such that distance between them is <= max_dist."""
        d = np.linalg.norm(b - a)
        return max(2, int(np.ceil(d / max_dist)) + 1)  # include endpoint, at least 2 points

    def interp(a, b, n, endpoint=True):
        """Linear interpolation from point a to b, using n samples."""
        if np.all(a == b):
            return [a]
        return [(1 - s) * a + s * b for s in np.linspace(0, 1, n, endpoint=endpoint)]

    # find the longest edge
    edges = np.array([[p[0], p[1]], [p[1], p[2]], [p[2], p[0]]])
    m = [samples(*e) for e in edges]
    ind = np.argmax(m)
    # longest edge first, reversed
    edges = np.roll(edges, -ind, 0)
    edges[0] = edges[0][::-1]
    m = np.roll(m, -ind, 0)
    # interpolate lines of points from the longest edge to the two others
    n = m[1] + m[2] - 1  # use the shared point between the two other edges only once
    longest = np.array(interp(*edges[0], n))
    others = np.r_[interp(*edges[1], m[1] - 1, endpoint=False), interp(*edges[2], m[2])]
    ret = []
    for a, b in zip(longest, others):
        ret.extend(interp(a, b, samples(a, b)))
    return np.array(ret)
