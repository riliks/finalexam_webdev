r"""
Control sequences
=================

.. currentmodule:: qit.seq

Contents
--------

.. autosummary::
   Seq
   nmr
   bb1
   corpse
   scrofulous
   knill
   dd
   propagate
"""
# Ville Bergholm 2009-2020

import numpy as np
import scipy.linalg as spl
import scipy.optimize as spo

from .base import sx, sy


__all__ = ['Seq', 'nmr', 'bb1', 'corpse', 'scrofulous', 'knill', 'dd', 'propagate']



class Seq:
    r"""
    Piecewise constant control sequences for quantum systems.

    Args:
        tau (array[float]): durations of the time slices
        control (array[float]): ``control[i, j]`` is the value of control field j during time slice i. ``shape == (len(tau), len(B))``.
        name (str): name of the sequence

    The total generator for the time slice j is given by

    .. math::

      G_j = A +\sum_k \text{control}_{jk} B_k,

    and the corresponding propagator is

    .. math::

      P_j = \exp(\tau_j G_j).
    """
    def __init__(self, tau=None, control=None, name=''):
        # construct the sequence
        if tau is None:
            tau = []
        if control is None:
            control = np.zeros((0, 2))

        #: str: name of the sequence
        self.name = name
        #: array: drift generator (typically :math:`-i/\hbar` times a Hamiltonian and a time unit of your choice)
        self.A = np.zeros((2, 2), dtype=complex)
        #: list[array]: control generators
        self.B = [-0.5j * sx, -0.5j * sy]
        #: array[float]: durations of the time slices
        self.tau = np.asarray(tau)
        #: array[float]: control field values for each time slice
        self.control = np.asarray(control)

    def __str__(self):
        # prettyprint the object
        out = 'Control sequence {}\n'.format(self.name)
        out += 'tau: ' + repr(self.tau) +'\n'
        out += 'control: ' + repr(self.control)
        return out

    def __len__(self):
        """Sequence length, i.e. the number of time slices in it."""
        return len(self.tau)

    def generator(self, j):
        """Generator for time slice j.

        Args:
            j (int): time slice index

        Returns:
            array[complex]: antihermitian generator for the given time slice
        """
        G = np.asarray(self.A, dtype=complex).copy()
        for k, B in enumerate(self.B):
            G += self.control[j, k] * B
        return G

    def to_prop(self):
        r"""Propagator corresponding to the control sequence.

        Returns the propagator matrix corresponding to the
        action of the control sequence.

        Governing equation: :math:`\dot(X)(t) = (A +\sum_k u_k(t) B_k) X(t) = G(t) X(t)`.
        """
        n = len(self.tau)
        P = np.eye(self.A.shape[0], dtype=complex)
        for j in range(n):
            G = self.generator(j)
            temp = spl.expm(self.tau[j] * G)
            P = temp @ P
        return P


def nmr(a, name=''):
    r"""Convert a sequence of NMR-style rotations into a one-qubit control sequence.

    Args:
        a (array_like): sequence of one-qubit rotations using the NMR notation: :math:`[[\theta_1, \phi_1], [\theta_2, \phi_2], ...]`
        name (str): name of the sequence

    Each :math:`\theta, \phi` pair corresponds to a NMR rotation
    of the form :math:`\theta_\phi`,
    or a rotation of the angle :math:`\theta`
    about the unit vector :math:`[\cos(\phi), \sin(\phi), 0]`.

    .. math::

       R_{\vec{a}}(\theta) = \exp(-i \vec{a} \cdot \vec{\sigma} \theta/2) = \exp(-i H t) \quad \Leftarrow \quad
       H = \vec{a} \cdot \vec{\sigma}/2, \quad t = \theta.

    Returns:
        Seq: one-qubit control sequence corresponding to the array a
    """
    a = np.asarray(a, dtype=float)
    theta = a[:, 0]
    phi   = a[:, 1]

    # find theta angles that are negative, convert them to corresponding positive rotations
    rows = np.nonzero(theta < 0)[0]
    theta[rows] = -theta[rows]
    phi[rows] = phi[rows] + np.pi
    return Seq(theta, np.c_[np.cos(phi), np.sin(phi)], name=name)


def bb1(theta, phi=0, location=0.5):
    r"""Sequence for correcting pulse length errors.

    Args:
        theta (float): NMR rotation angle
        phi (float): NMR rotation phase
        location (float): relative location of the correction block within the sequence, in [0, 1].

    Returns:
        Seq: bb1 sequence corresponding to the given NMR rotation

    The broadband number 1 control sequence corrects
    proportional errors in pulse length (or amplitude) :cite:`Wimperis`.

    The target rotation is :math:`\theta_\phi` in the NMR notation, see :func:`nmr`.
    """
    # Ville Bergholm 2009-2012

    ph1 = np.arccos(-theta / (4*np.pi))
    W1  = [[np.pi, ph1], [2*np.pi, 3*ph1], [np.pi, ph1]]
    return nmr([[location * theta, phi]] + W1 + [[(1-location) * theta, phi]], name='BB1')


def corpse(theta, phi=0):
    r"""Sequence for correcting off-resonance errors.

    Args:
        theta (float): NMR rotation angle
        phi (float): NMR rotation phase

    Returns:
        Seq: CORPSE sequence corresponding to the given NMR rotation

    The CORPSE control sequence corrects off-resonance
    errors, i.e. ones arising from a constant but unknown
    :math:`\sigma_z` bias in the Hamiltonian :cite:`Cummins`.

    The target rotation is :math:`\theta_\phi` in the NMR notation, see :func:`nmr`.

    CORPSE: Compensation for Off-Resonance with a Pulse SEquence
    """
    # Ville Bergholm 2009

    n = [1, 1, 0] # CORPSE
    #n = [0, 1, 0] # short CORPSE

    temp = np.arcsin(np.sin(theta / 2) / 2)

    th1 = 2*np.pi*n[0] +theta/2 -temp
    th2 = 2*np.pi*n[1] -2*temp
    th3 = 2*np.pi*n[2] +theta/2 -temp
    return nmr([[th1, phi], [th2, phi+np.pi], [th3, phi]], name='CORPSE')


def scrofulous(theta, phi=0):
    r"""Sequence for correcting pulse length errors.

    Args:
        theta (float): NMR rotation angle
        phi (float): NMR rotation phase

    Returns:
        Seq: SCROFULOUS sequence corresponding to the given NMR rotation

    The SCROFULOUS control sequence corrects errors
    in pulse duration (or amplitude) :cite:`Cummins`.

    The target rotation is :math:`\theta_\phi` in the NMR notation, see :func:`nmr`.

    SCROFULOUS: Short Composite ROtation For Undoing Length Over- and UnderShoot
    """
    # Ville Bergholm 2006-2016

    th1 = spo.brentq(lambda t: (np.sin(t)/t -(2 / np.pi) * np.cos(theta / 2)), 0.1, 4.6)
    ph1 = np.arccos(-np.pi * np.cos(th1) / (2 * th1 * np.sin(theta / 2))) +phi
    ph2 = ph1 - np.arccos(-np.pi / (2 * th1))

    u1 = [[th1, ph1]]
    u2 = [[np.pi,  ph2]]
    return nmr(u1 + u2 + u1, name='SCROFULOUS')


def knill(phi=0):
    r"""Sequence for robust pi pulses.

    Args:
        phi (float): phase angle

    Returns:
        Seq: Knill pi rotation sequence

    The target rotation in the NMR notation is :math:`\pi_\phi` followed by :math:`R_z(-\pi/3)`.
    In an experimental setting the Z rotation can often be absorbed by a
    reference frame change that does not affect the measurement results :cite:`RHC2010`.

    The Knill sequence is quite robust against off-resonance errors, and somewhat
    robust against pulse strenght errors.
    """
    # Ville Bergholm 2015-2016

    th = np.pi
    return nmr([[th, np.pi/6+phi], [th, phi], [th, np.pi/2+phi], [th, phi], [th, np.pi/6+phi]], name='Knill')


def dd(name, t, *, amp=1.0, n=1):
    r"""Dynamical decoupling and refocusing sequences.

    Args:
        name (str): name of the sequence, in ('wait', 'hahn', 'cpmg', 'uhrig', 'xy4')
        t (float): total waiting time
        amp (float): pi pulse amplitude (higher amplitude yields faster pi pulses
            and better-performing sequences)
        n (int): order (if applicable)

    Returns:
        Seq: decoupling/refocusing sequence

    All these sequences consist of waiting periods interspersed with hard, fast pi pulses
    around an axis in the XY plane of the qubit.
    The target operation for all these sequences is :math:`X^k`, where :math:`k` is the number of pi pulses in the sequence.
    See e.g. :cite:`Uhrig2007`.
    """
    # Ville Bergholm 2007-2016

    # Multiplying the pi pulse strength by factor s is equivalent to A -> A/s, t -> t*s.
    # which sequence?
    if name == 'wait':
        # Do nothing, just wait. For comparison.
        tau = [1]
        phase = []
    elif name == 'hahn':
        # Basic Hahn spin echo
        tau = [0.5, 0.5]
        phase = [0]
    elif name == 'cpmg':
        # Carr-Purcell-Meiboom-Gill
        # The purpose of the CPMG sequence is to facilitate a T_2 measurement
        # under a nonuniform z drift, it is not meant to be a full memory protocol.
        tau = [0.25, 0.5, 0.25]
        phase = [0, 0]
    elif name == 'uhrig':
        # Uhrig's family of sequences
        # n=1: Hahn echo
        # n=2: CPMG
        delta = np.arange(n+2)
        delta = np.sin(np.pi * delta / (2 * (n + 1))) ** 2
        tau   = delta[1:] - delta[:n+1]  # wait durations
        phase = np.zeros(n)
    elif name == 'xy4':
        # uncentered version
        tau = np.ones(4) / 4
        phase = [0, np.pi/2, 0, np.pi/2]
    else:
        raise ValueError('Unknown sequence.')

    # initialize the sequence struct
    s = Seq(name=name)
    # waits and pi pulses with given phases
    for k, p in enumerate(phase):
        # wait
        s.tau = np.r_[s.tau, t * tau[k]]
        s.control = np.r_[s.control, np.zeros((1, 2))]
        # pi pulse
        s.tau = np.r_[s.tau, np.pi / amp]
        s.control = np.r_[s.control, amp * np.array([[np.cos(p), np.sin(p)]])]
    if len(tau) > len(phase):
        # final wait
        s.tau = np.r_[s.tau, t * tau[-1]]
        s.control = np.r_[s.control, np.zeros((1, 2))]
    return s


def propagate(s, seq, out_func=lambda x: x, base_dt=0.1):
    """Propagate a state in time using a control sequence.

    Args:
        s (~qit.state.state): state to propagate
        seq (seq): control sequence
        out_func (callable[[state], Any]): Transformation for the propagated state. Default is identity.
        base_dt (float): maximum timestep size

    Returns:
        list[Any], list[float]: propagated transformed states, corresponding time instances
    """
    # Ville Bergholm 2009-2016

    n = len(seq)
    t = [0]  # initial time
    out = [out_func(s)]  # initial state

    # loop over the sequence
    for j in range(n):
        G = seq.generator(j)
        T = seq.tau[j]  # pulse duration
        n_steps = max(int(np.ceil(T / base_dt)), 1)
        dt = T / n_steps

        P = spl.expm(G * dt)
        for _ in range(n_steps):
            s = s.u_propagate(P)
            out.append(out_func(s))

        temp = t[-1]
        t.extend(list(np.linspace(temp+dt, temp+T, n_steps)))
    return out, t
