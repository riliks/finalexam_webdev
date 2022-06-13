r"""
Harmonic oscillators
====================

.. currentmodule:: qit.ho

This module simulates harmonic oscillators by truncating the state
space dimension to a finite value. Higher truncation limits give more accurate results.
All the functions in this module operate in the truncated number basis
:math:`\{\ket{0}, \ket{1}, ..., \ket{n-1}\}`
of the harmonic oscillator, where n is the truncation dimension.

The corresponding truncated annihilation operator can be obtained with :func:`qit.utils.boson_ladder`.


Contents
--------

.. autosummary::

   coherent_state
   position_state
   momentum_state
   position
   momentum
   displace
   squeeze
   rotate
   beamsplitter
   cx
   husimi
   wigner
"""
# Author: Ville Bergholm 2011-2020

import numpy as np
import scipy.special as sps
import scipy.linalg as spl

from .state import State
from .utils import boson_ladder

__all__ = ['coherent_state', 'displace', 'squeeze', 'rotate',
           'beamsplitter', 'cx',
           'position', 'momentum',
           'position_state', 'momentum_state', 'husimi', 'wigner']


# default truncation dimension for states and operators in the number state basis
default_n = 30

def coherent_state(alpha, n=default_n):
    r"""Coherent states of a harmonic oscillator.

    Args:
        alpha (complex): displacement parameter
        n (int): truncation dimension

    Returns:
        array[complex]: coherent state

    Returns the n-dimensional approximation to the
    coherent state :math:`\ket{\alpha}`,

    .. math::

       \ket{\alpha} := D(\alpha) \ket{0}
       = e^{-\frac{|\alpha|^2}{2}} \sum_{k=0}^\infty \frac{\alpha^k}{\sqrt{k!}} \ket{k},

    in the number basis. The coherent states are eigenstates of the annihilation
    operator, :math:`a\ket{\alpha} = \alpha \ket{\alpha}`.
    """
    # Ville Bergholm 2010

    k = np.arange(n)
    ket = (alpha ** k) / np.sqrt(sps.factorial(k))
    return State(ket, n).normalize()
    #s = State(0, n).u_propagate(spl.expm(alpha * boson_ladder(n).T.conj()))
    #s = State(0, n).u_propagate(displace(alpha, n))
    #s *= exp(-abs(alpha) ** 2 / 2) # normalization


def displace(alpha, n=default_n):
    r"""Bosonic displacement operator.

    Args:
        alpha (complex): displacement parameter
        n (int): truncation dimension

    Returns:
        array[complex]: displacement operator

    Returns the n-dimensional approximation for the bosonic
    phase space displacement operator

    .. math::

       D(\alpha) := \exp\left(\alpha a^\dagger - \alpha^* a\right)
       = \exp\left( i \sqrt{2} \left(Q \: \im{\alpha} -P \: \re{\alpha}\right)\right)

    in the number basis. This yields

    .. math::

       D^\dagger(\alpha) Q D(\alpha) &= Q +\sqrt{2} \: \re{\alpha} \: \mathbb{I},\\
       D^\dagger(\alpha) P D(\alpha) &= P +\sqrt{2} \: \im{\alpha} \: \mathbb{I},

    and thus the displacement operator displaces the state of a harmonic oscillator in phase space.
    """
    if not np.isscalar(alpha):
        raise TypeError('alpha must be a scalar.')

    a = boson_ladder(n)
    return spl.expm(alpha * a.T.conj() -alpha.conjugate() * a)


def squeeze(z, n=default_n):
    r"""Bosonic squeezing operator.

    Args:
        z (complex): squeezing parameter
        n (int): truncation dimension

    Returns:
        array[complex]: squeezing operator

    Returns the n-dimensional approximation for the bosonic
    phase space squeezing operator

    .. math::

       S(z) := \exp\left(\frac{1}{2} (z^* a^2 - z a^{\dagger 2})\right)
       = \exp\left(\frac{i}{2} \left((QP+PQ)\re{z} +(P^2-Q^2)\im{z}\right)\right)

    in the number basis.
    """
    if not np.isscalar(z):
        raise TypeError('z must be a scalar.')

    a = boson_ladder(n)
    ad = a.T.conj()
    return spl.expm(0.5 * (z.conjugate() * (a @ a) - z * (ad @ ad)))


def rotate(phi, n=default_n):
    r"""Bosonic rotation operator.

    Args:
        phi (float): rotation angle
        n (int): truncation dimension

    Returns:
        array[complex]: rotation operator

    Returns the n-dimensional approximation for the bosonic
    phase space rotation operator

    .. math::

       R(\phi) := \exp\left(i \phi a^{\dagger} a \right)
       = \exp\left(i \frac{\phi}{2} \left(Q^2 + P^2 -\I\right)\right)

    in the number basis.
    """
    if not np.isscalar(phi):
        raise TypeError('z must be a scalar.')

    num = np.exp((1j * phi) * np.arange(n))
    return np.diag(num)


def beamsplitter(theta, phi, n=default_n):
    r"""Bosonic beamsplitter operator.

    Args:
        theta (float): rotation angle
        phi (float): phase angle
        n (int): truncation dimension

    Returns:
        array[complex]: beamsplitter operator

    Returns the n-dimensional approximation for the bosonic
    two-mode beamsplitter operator

    .. math::

       B(\theta, \phi) :=& \exp\left(z a_1^\dagger a_2 -z^* a_1 a_2^\dagger\right)\\
       =& \exp\left(\theta (e^{i\phi} a_1^\dagger a_2 -e^{-i\phi} a_1 a_2^\dagger)\right)\\
       =& \exp\left(i \left((Q \otimes P - P \otimes Q) \re{z} + (Q \otimes Q + P \otimes P) \im{z} \right)\right)

    in the number basis, where :math:`z = \theta e^{i \phi}`.
    """
    z = theta * np.exp(1j * phi)
    a = boson_ladder(n)
    ad = a.T.conj()
    return spl.expm(z * np.kron(ad, a) -z.conjugate() * np.kron(a, ad))


def cx(s=1.0, n=default_n):
    r"""Bosonic controlled addition operator.

    Args:
        s (float): scale factor
        n (int): truncation dimension

    Returns:
        array[complex]: controlled addition operator

    Returns the n-dimensional approximation for the bosonic
    two-mode controlled addition operator

    .. math::

       CX(s) := \exp\left(-i s Q \otimes P\right)

    in the number basis. In the position basis it has the effect

    .. math::

       CX(s)\ket{x_1, x_2}_x = \ket{x_1, x_2 + s x_1}_x.
    """
    a = boson_ladder(n)
    ad = a.T.conj()
    temp = -0.5 * s * np.kron(a + ad, a - ad)
    return spl.expm(temp)


def position(n=default_n):
    r"""Position operator.

    Args:
        n (int): truncation dimension

    Returns:
        array[complex]: position operator

    Returns the n-dimensional truncation of the
    dimensionless position operator Q in the number basis.

    .. math::

       Q &= \sqrt{\frac{m \omega}{\hbar}}   q =    (a+a^\dagger) / \sqrt{2},\\
       P &= \sqrt{\frac{1}{m \hbar \omega}} p = -i (a-a^\dagger) / \sqrt{2}.

    (Equivalently, :math:`a = (Q + iP) / \sqrt{2}`).
    These operators fulfill :math:`[q, p] = i \hbar, \quad  [Q, P] = i`.
    The Hamiltonian of the harmonic oscillator is

    .. math::

       H = \frac{p^2}{2m} +\frac{1}{2} m \omega^2 q^2
         = \frac{1}{2} \hbar \omega \left(P^2 +Q^2\right)
         = \hbar \omega \left(a^\dagger a +\frac{1}{2}\right).
    """
    a = boson_ladder(n)
    return np.array(a + a.T.conj()) / np.sqrt(2)


def momentum(n=default_n):
    """Momentum operator.

    Args:
        n (int): truncation dimension

    Returns:
        array[complex]: momentum operator

    Returns the n-dimensional truncation of the
    dimensionless momentum operator P in the number basis.

    See :func:`position`.
    """
    a = boson_ladder(n)
    return -1j * np.array(a - a.T.conj()) / np.sqrt(2)


def position_state(q, n=default_n):
    r"""Position eigenstates of a harmonic oscillator.

    Args:
        q (float): dimensionless position coordinate
        n (int): truncation dimension

    Returns:
        array[complex]: approximate position eigenstate

    Returns the n-dimensional approximation of the eigenstate :math:`\ket{q}`
    of the dimensionless position operator Q in the number basis.

    See :func:`position`, :func:`momentum`.

    Difference equation:

    .. math::

       r_1 &= \sqrt{2} \: q \: r_0,\\
       \sqrt{k+1} \: r_{k+1} &= \sqrt{2} \: q \: r_k -\sqrt{k} \: r_{k-1}, \qquad \text{when} \quad k >= 1.
    """
    # Ville Bergholm 2010

    ket = np.empty(n, dtype=complex)
    temp = np.sqrt(2) * q
    ket[0] = 1  # arbitrary nonzero initial value r_0
    ket[1] = temp * ket[0]
    for k in range(2, n):
        ket[k] = temp / np.sqrt(k) * ket[k - 1] - np.sqrt((k-1) / k) * ket[k - 2]
    ket /= spl.norm(ket)
    return State(ket, n)


def momentum_state(p, n=default_n):
    r"""Momentum eigenstates of a harmonic oscillator.

    Args:
        q (float): dimensionless momentum coordinate
        n (int): truncation dimension

    Returns:
        array[complex]: approximate momentum eigenstate

    Returns the n-dimensional approximation of the eigenstate :math:`\ket{p}`
    of the dimensionless momentum operator P in the number basis.

    See :func:`position`, :func:`momentum`.

    Difference equation:

    .. math::

       r_1 &= i \sqrt{2} \: p \: r_0,\\
       \sqrt{k+1} \: r_{k+1} &= i \sqrt{2} \: p \: r_k +\sqrt{k} \: r_{k-1}, \qquad \text{when} \quad k >= 1.
    """
    # Ville Bergholm 2010

    ket = np.empty(n, dtype=complex)
    temp = 1j * np.sqrt(2) * p
    ket[0] = 1  # arbitrary nonzero initial value r_0
    ket[1] = temp * ket[0]
    for k in range(2, n):
        ket[k] = temp / np.sqrt(k) * ket[k - 1] + np.sqrt((k-1) / k) * ket[k - 2]
    ket /= spl.norm(ket)
    return State(ket, n)


def husimi(rho, alpha=None, z=0, *, res=(40, 40), lim=(-2, 2, -2, 2)):
    r"""Husimi probability distribution.

    Args:
        rho (State): harmonic oscillator state (truncated)
        alpha (array[complex]): displacement parameters of the reference state
        z (complex): squeezing parameter of the reference state
        res (tuple[int]): if ``alpha is None``: number of points in the alpha grid, (nx, ny)
        lim (tuple[float]): if ``alpha is None``: limits of the alpha grid, (xmin, xmax, ymin, ymax)

    Returns:
        array[float] [, array[float], array[float]]: Husimi distribution H_rho(alpha), ``H.shape == alpha.shape``

    Returns the Husimi probability distribution
    :math:`H(\im{\alpha}, \re{\alpha})` corresponding to the harmonic
    oscillator state rho given in the number basis:

    .. math::

       H(\rho, \alpha, z) = \frac{1}{\pi} \bra{\alpha, z} \rho \ket{\alpha, z}

    ``z`` is the optional squeezing parameter for the reference state:
    :math:`\ket{\alpha, z} := D(\alpha) S(z) \ket{0}`.
    The integral of :math:`H(\alpha)` over :math:`\alpha` is normalized to unity.

    If ``alpha`` is ``None`` it is set to a 2d grid of points with the resolution and limits
    set by ``res`` and ``lim``. In addition to ``H``, the 1d x and y coordinate vectors of the grid are returned.
    """
    if alpha is None:
        # return a 2D grid of W values
        a = np.linspace(lim[0], lim[1], res[0])
        b = np.linspace(lim[2], lim[3], res[1])
        #a, b = ogrid[lim[0]:lim[1]:1j*res[0], lim[2]:lim[3]:1j*res[1]]
        alpha = a + 1j * b[:, np.newaxis]
        return_ab = True
    else:
        return_ab = False

    # reference state
    n = np.prod(rho.dims())
    ref = State(0, n).u_propagate(squeeze(z, n))
    ref /= np.sqrt(np.pi) # normalization included for convenience

    H = np.empty(alpha.shape)
    for k, c in enumerate(alpha.flat):
        temp = ref.u_propagate(displace(c, n))
        H.flat[k] = rho.fidelity(temp) ** 2

    if return_ab:
        return H, a, b
    return H


def wigner(rho, alpha=None, *, res=(20, 20), lim=(-2, 2, -2, 2), method=0):
    r"""Wigner quasi-probability distribution.

    Args:
      rho (State): harmonic oscillator state
      alpha (array[complex]): phase space points for which to compute the Wigner function
      res (tuple[int]): if ``alpha is None``: number of points in the alpha grid, (nx, ny)
      lim (tuple[float]): if ``alpha is None``: limits of the alpha grid, (xmin, xmax, ymin, ymax)

    Returns:
      array[float], array[float], array[float]: wigner(alpha), alpha_re, alpha_im

    Returns the Wigner quasi-probability distribution
    :math:`W(\im{\alpha}, \re{\alpha})` corresponding to the harmonic
    oscillator state ``rho`` given in the number basis.

    For a normalized state, the integral of W is normalized to unity.

    NOTE: The truncation of the number state space to a finite dimension
    results in spurious circular ripples in the Wigner function outside
    a given radius. To increase the accuracy, increase the truncation dimension.
    """
    # pylint: disable=too-many-locals,undefined-variable
    if alpha is None:
        # return a grid of W values for a grid of alphas
        a = np.linspace(lim[0], lim[1], res[0])
        b = np.linspace(lim[2], lim[3], res[1])
        #a, b = ogrid[lim[0]:lim[1]:1j*res[0], lim[2]:lim[3]:1j*res[1]]
        alpha = a + 1j*b[:, np.newaxis]
        return_ab = True
    else:
        return_ab = False

    n = np.prod(rho.dims())
    W = np.empty(alpha.shape)
    # parity operator (diagonal)
    P = np.ones(n)
    P[1:n:2] = -1
    P *= 2 / np.pi  # include Wigner normalization here for convenience

    if method == 0:
        for k, c in enumerate(alpha.flat):
            temp = rho.u_propagate(displace(-c, n))
            W.flat[k] = np.sum(P * temp.prob().real) # == ev(temp, P).real
    else:
        # TODO new faster method
        for k, c in enumerate(alpha.flat):
            temp = np.empty((n,), dtype=complex)
            for y in range(n):
                for w in range(n):
                    temp[y] = _displacement(y, w, -c) * rho.data[w, 0]
            temp = State(temp)
            W.flat[k] = np.sum(P * temp.prob().real)

    if return_ab:
        return W, a, b
    return W
