"""
Demos and examples
==================

.. currentmodule:: qit.examples

This module contains various examples which demonstrate the features
of QIT. The :func:`tour` function is a "guided tour" to the toolkit
which runs all the demos in succession.


Demos
-----

.. autosummary::

   adiabatic_qc_3sat
   bb84
   bernstein_vazirani
   grover_search
   markov_decoherence
   nmr_sequences
   phase_estimation_precision
   qft_circuit
   quantum_channels
   quantum_walk
   qubit_and_resonator
   shor_factorization
   superdense_coding
   teleportation
   werner_states
   markovian_bath_properties
   tour


General-purpose quantum algorithms
----------------------------------

.. autosummary::

   adiabatic_qc
   phase_estimation
   find_order
"""
# Ville Bergholm 2011-2020
# pylint: disable=too-many-locals,too-many-statements

from __future__ import annotations

from operator import mod
from copy import deepcopy

import numpy as np
from numpy import pi
import numpy.linalg
import numpy.random as npr
import scipy.special as sps
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D   # FIXME makes buggy 3d plotting work for some reason

from qit.base import sx, sy, sz, p0, p1, TOLERANCE
from qit.lmap import Lmap, tensor, numstr_to_array
from qit.utils import gcd, lcm, qubits, Ry, rand_U, rand_SU, mkron, boson_ladder
from qit.state import State, fidelity
from qit import gate, ho, plot, base, markov, seq


def adiabatic_qc_3sat(n=6, n_clauses=None, clauses=None, problem='3sat'):
    """Adiabatic quantum computing demo.

    This example solves random 3-SAT problems by simulating the
    adiabatic quantum algorithm of :cite:`Farhi`.

    NOTE: This simulation is incredibly inefficient because we first essentially
    solve the NP-complete problem classically using an exhaustive
    search when computing the problem Hamiltonian H1, and then
    simulate an adiabatic quantum computer solving the same problem
    using the quantum algorithm.

    Args:
      n (int): number of logical variables (bits, qubits)
      n_clauses (int): number of clauses
      clauses (array[int]): array of shape (n_clauses, 3) defining the clauses
      problem (str): problem type, either '3sat' or '3exact_cover'
    """
    # Ville Bergholm 2009-2018

    print('\n\n=== Solving 3-SAT using adiabatic qc ===\n')

    n = max(n, 3)

    if clauses is None:
        if n_clauses is None:
            if problem == '3sat':
                n_clauses = 5*n
            else:  # exact cover
                n_clauses = n//2

        # generate clauses
        clauses = np.zeros((n_clauses, 3), dtype=int)
        for k in range(n_clauses):
            bits = list(range(n))
            for j in range(3):
                clauses[k, j] = bits.pop(npr.randint(len(bits))) + 1 # zero can't be negated, add one
        clauses = np.sort(clauses, 1)

        if problem == '3sat':
            for k in range(n_clauses):
                for j in range(3):
                    clauses[k, j] *= (-1) ** (npr.rand() < 0.5) # negate if bit should be inverted
    else:
        n_clauses = clauses.shape[0]

    # cache some stuff (all the matrices in this example are diagonal, so)
    zb = np.array([0, 1], dtype=int) # 0.5*(I - sz)
    z_op = []
    for k in range(n):
        z_op.append(mkron(np.ones(2 ** k, dtype=int), zb, np.ones(2 ** (n-k-1), dtype=int)))

    print('{} bits, {} clauses'.format(n, n_clauses))
    # Encode the problem into the final Hamiltonian.
    H1 = 0
    if problem == '3sat':
        b = [0, 0, 0]
        for k in range(n_clauses):
            # h_c = (b1(*) v b2(*) v b3(*))
            for j in range(3):
                temp = clauses[k, j]
                b[j] = z_op[abs(temp) - 1] # into zero-based index
                if temp < 0:
                    b[j] = np.logical_not(b[j])
            H1 += np.logical_not(b[0] + b[1] + b[2]) # the not makes this a proper OR
        print('Find a bit string such that all the following "(b_i or b_j or b_k)" clauses are satisfied.')
        print('Minus sign means the bit is negated.')
    else:
        # exact cover
        for k in range(n_clauses):
            # h_c = (b1 ^ b2* ^ b3*) v (b1* ^ b2 ^ b3*) v (b1* ^ b2* ^ b3)
            b1 = z_op[clauses[k, 0] - 1]
            b2 = z_op[clauses[k, 1] - 1]
            b3 = z_op[clauses[k, 2] - 1]
            s1 = b1 * np.logical_not(b2) * np.logical_not(b3)
            s2 = b2 * np.logical_not(b3) * np.logical_not(b1)
            s3 = b3 * np.logical_not(b1) * np.logical_not(b2)
            H1 += np.logical_not(s1 + s2 + s3 -s1*s2 -s2*s3 -s3*s1 +s1*s2*s3)
        print('Find a bit string such that all the following "exactly one of bits {a, b, c} is 1" clauses are satisfied:')
    print(clauses)

    # build initial Hamiltonian
    xb = 0.5 * (np.eye(2) -sx)
    H0 = 0
    for k in range(n):
        H0 += mkron(np.eye(2 ** k), xb, np.eye(2 ** (n-k-1)))

    # initial state (ground state of H0)
    s0 = State(np.ones(2 ** n) / np.sqrt(2 ** n), qubits(n)) # n qubits, uniform superposition

    # adiabatic simulation
    adiabatic_qc(H0, H1, s0)
    return H1, clauses




def adiabatic_qc(H0, H1, s0, tmax=50):
    """Adiabatic quantum computing.

    This is a helper function for simulating the adiabatic quantum
    algorithm of :cite:`Farhi` and plotting the results.

    Args:
      H0 (array[complex]): initial Hamiltonian
      H1 (array[float]): diagonal of the final Hamiltonian (assumed diagonal)
      s0 (State): initial state
      tmax (float): final time
    """
    # Ville Bergholm 2009-2011

    H1_full = np.diag(H1) # into a full matrix

    # adiabatic simulation
    steps = tmax*10
    t = np.linspace(0, tmax, steps)

    # linear path
    H_func = lambda t: (1-t/tmax)*H0 +(t/tmax)*H1_full
    res = s0.propagate(H_func, t)

    # plots
    # final state probabilities
    plot.adiabatic_evolution(t, res, H_func)

    fig = plt.figure()
    ax = res[-1].plot(fig)
    ax.set_title('Final state')

    print('Final Hamiltonian (diagonal):')
    print(H1)

    print('Measured result:')
    dummy, dummy, res = res[-1].measure(do='C')
    print(res)
    if H1[np.nonzero(res.data)[0]] == 0:
        print('Which is a valid solution!')
    else:
        print('Which is not a solution!')
        if np.min(H1) > 0:
            print("(In this problem instance there aren't any.)")


def bb84(n=50):
    """Bennett-Brassard 1984 quantum key distribution protocol demo.

    Simulate the BB84 protocol with an eavesdropper attempting an intercept-resend attack.

    Args:
      n (int): number of qubits transferred
    """
    # Ville Bergholm 2009-2016
    print('\n\n=== BB84 protocol ===\n')
    print('Using {} transmitted qubits, intercept-resend attack.\n'.format(n))

    H = base.H

    # Alice generates two random bit vectors
    basis_A = npr.rand(n) > 0.5
    bits_A = npr.rand(n) > 0.5

    # Bob generates one random bit vector
    basis_B = npr.rand(n) > 0.5
    bits_B = np.zeros(n, dtype=bool)

    # Eve hasn't yet decided her basis
    basis_E = np.zeros(n, dtype=bool)
    bits_E = np.zeros(n, dtype=bool)

    print("""Alice transmits a sequence of qubits to Bob using a quantum channel.
For every qubit, she randomly chooses a basis (computational or diagonal)
and randomly prepares the qubit in either the |0> or the |1> state in that basis.""")
    print("\nbasis_A = {}\nbits_A  = {}".format(np.asarray(basis_A, dtype=int),
                                                np.asarray(bits_A, dtype=int)))

    temp = State('0')
    for k in range(n):
        # Alice has a source of qubits in the zero state.
        q = temp

        # Should Alice flip the qubit?
        if bits_A[k]:
            q = q.u_propagate(sx)

        # Should Alice apply a Hadamard?
        if basis_A[k]:
            q = q.u_propagate(H)

        # Alice now sends the qubit to Bob...
        # ===============================================
        # ...but Eve intercepts it, and conducts an intercept/resend attack

        # Eve might have different strategies here... TODO
        # Eve's strategy (simplest choice, random)
        basis_E[k] = npr.rand() > 0.5

        # Eve's choice of basis: Hadamard?
        if basis_E[k]:
            q = q.u_propagate(H)

        # Eve measures in the basis she has chosen
        _, res, q = q.measure(do='C')
        bits_E[k] = res

        # Eve tries to reverse the changes she made...
        if basis_E[k]:
            q = q.u_propagate(H)

        # ...and sends the result to Bob.
        # ===============================================

        # Bob's choice of basis
        if basis_B[k]:
            q = q.u_propagate(H)

        # Bob measures in the basis he has chosen, and discards the qubit.
        _, res = q.measure()
        bits_B[k] = res

    #sum(xor(bits_A, bits_E))/n
    #sum(xor(bits_A, bits_B))/n

    print("""\nHowever, there's an eavesdropper, Eve, on the line. She intercepts the qubits,
randomly measures them in either basis (thus destroying the originals!), and then sends
a new batch of qubits corresponding to her measurements and basis choices to Bob.
Since Eve on the average can choose the right basis only 1/2 of the time,
about 1/4 of her bits differ from Alice's.""")
    print("\nbasis_E = {}\nbits_E  = {}".format(np.asarray(basis_E, dtype=int),
                                                np.asarray(bits_E, dtype=int)))
    print('\nMismatch frequency between Alice and Eve: {}'.format(np.sum(np.logical_xor(bits_A, bits_E)) / n))

    print("""\nWhen Bob receives the qubits, he randomly measures them in either basis.
Due to Eve's eavesdropping, Bob's bits differ from Alice's 3/8 of the time.""")
    print("\nbasis_B = {}\nbits_B  = {}".format(np.asarray(basis_B, dtype=int),
                                                np.asarray(bits_B, dtype=int)))
    print('\nMismatch frequency between Alice and Bob: {}'.format(np.sum(np.logical_xor(bits_A, bits_B)) / n))

    print("""\nNow Bob announces on a public classical channel that he has received all the qubits.
Alice and Bob then reveal the bases they used. Whenever the bases happen to match,
(about 1/2 of the time on the average), they both add their corresponding bit to
their personal key. The two keys should be identical unless there's been an eavesdropper.
However, because of Eve each key bit has a 1/4 probability of being wrong.\n""")

    match = np.logical_not(np.logical_xor(basis_A, basis_B))
    key_A = bits_A[match]
    key_B = bits_B[match]
    m = len(key_A)
    print('{} basis matches.\nkey_A = {}\nkey_B = {}'.format(np.sum(match),
                                                             np.asarray(key_A, dtype=int),
                                                             np.asarray(key_B, dtype=int)))
    print("\nMismatch frequency between Alice's and Bob's keys: {}".format(np.sum(np.logical_xor(key_A, key_B)) / m))

    print("""\nAlice and Bob then sacrifice k bits of their shared key to compare them.
If a nonmatching bit is found, the reason is either an eavesdropper or a noisy channel.
With a noiseless channel Alice and Bob will detect Eve's presence with the probability 1-(3/4)^k.""")



def bernstein_vazirani(n=6, linear=True):
    r"""Bernstein-Vazirani algorithm demo.

    Simulates the Bernstein-Vazirani algorithm :cite:`BV`, which, given a black box oracle
    implementing a linear Boolean function :math:`f_a(x) := a \cdot x`, returns the bit
    vector a (and thus identifies the function) with just a single oracle call.
    If the oracle function is not linear, the algorithm will fail.

    Args:
      n (int): number of qubits
      linear (bool): if False, use a random nonlinear function instead, leading to the algorithm failing
    """
    # Ville Bergholm 2011-2012

    print('\n\n=== Bernstein-Vazirani algorithm ===\n')

    print('Using {} qubits.'.format(n))
    dim = qubits(n)
    H = gate.walsh(n) # n-qubit Walsh-Hadamard gate
    N = 2 ** n

    def oracle(f):
        """Returns a unitary oracle for the Boolean function f(x), given as a truth table."""
        return (-1) ** f.reshape((-1, 1))

    def linear_func(a):
        r"""Builds the linear Boolean function f(x) = a \cdot x as a truth table."""
        dim = qubits(len(a))
        N = np.prod(dim)
        U = np.empty(N, dtype=int)
        for k in range(N):
            x = np.unravel_index(k, dim)
            U[k] = mod(a @ x, 2)
        return U

    # black box oracle encoding the Boolean function f (given as the diagonal)
    if linear:
        # linear f
        a = np.asarray(npr.rand(n) > 0.5, dtype=int)
        f = linear_func(a)
        print('\nUsing the linear function f_a(x) := dot(a, x), defined by the binary vector a = {}.'.format(a))
    else:
        # general f
        f = np.asarray(npr.rand(N) > 0.5, dtype=int)
        # special case: not(linear)
        #a = np.asarray(npr.rand(n) > 0.5, dtype=int)
        #f = 1-linear_func(a)
        print('\nNonlinear function f:\n{}.'.format(f))
    U_oracle = oracle(f)

    # start with all-zero state
    s = State(0, dim)
    # initial superposition
    s = s.u_propagate(H)
    # oracle phase flip
    s.data = U_oracle * s.data
    # final Hadamards
    s = s.u_propagate(H)

    fig = plt.figure()
    s.plot(fig)
    title = 'Bernstein-Vazirani algorithm, '
    if linear:
        title += 'linear oracle'
    else:
        title += 'nonlinear oracle (fails)'
    fig.gca().set_title(title)
    fig.show()

    p, res = s.measure()
    # measured binary vector
    b = np.asarray(np.unravel_index(res, dim))
    print('\nMeasured binary vector b = {}.'.format(b))
    if not linear:
        g = linear_func(b)
        print('\nCorresponding linear function g_b:\n{}.'.format(g))
        print('\nNormalized Hamming distance: |f - g_b| = {}.'.format(sum(abs(f-g)) / N))

    return p



def grover_search(n=8):
    """Grover search algorithm demo.

    Simulate the Grover search algorithm :cite:`Grover` formulated using amplitude amplification.

    Args:
      n (int): number of qubits
    """
    # Ville Bergholm 2009-2010

    print('\n\n=== Grover search algorithm ===\n')

    A = gate.walsh(n) # Walsh-Hadamard gate for generating uniform superpositions
    N = 2 ** n # number of states

    sol = npr.randint(N)
    reps = int(pi / (4 * np.arcsin(np.sqrt(1 / N))))

    print('Using {} qubits.'.format(n))
    print('Probability maximized by {} iterations.'.format(reps))
    print('Correct solution: {}'.format(sol))

    # black box oracle capable of recognizing the correct answer (given as the diagonal)
    # TODO an oracle that actually checks the solutions by computing (using ancillas?)
    U_oracle = np.ones((N, 1))
    U_oracle[sol, 0] = -1

    U_zeroflip = np.ones((N, 1))
    U_zeroflip[0, 0] = -1

    s = State(0, qubits(n))

    # initial superposition
    s = s.u_propagate(A)

    # Grover iteration
    for _ in range(reps):
        # oracle phase flip
        s.data = -U_oracle * s.data

        # inversion about the mean
        s = s.u_propagate(A.ctranspose())
        s.data = U_zeroflip * s.data
        s = s.u_propagate(A)

    p, res = s.measure()
    print('\nMeasured {}.'.format(res))
    return p



def markov_decoherence(T1=7e-10, T2=1e-9, B=None):
    """Markovian decoherence demo.

    Given a pair of T1 and T2 decoherence times, creates a system Hamiltonian
    and a system-bath coupling operator which reproduce them on a single-qubit system.

    Args:
      T1 (float): relaxation time T1 (in s)
      T2 (float): dephasing time T2 (in s)
      B (MarkovianBath): optional bath object
    """
    # Ville Bergholm 2009-2016
    print('\n\n=== Markovian decoherence in a qubit ===\n')

    TU = 1e-9  # time unit, s
    T = 1 # K
    delta = 3 + 3*npr.rand() # qubit energy splitting (GHz)

    T1 /= TU
    T2 /= TU

    # setup the bath
    if B is None:
        B = markov.MarkovianBath('ohmic', 'boson', TU, T) # defaults

    # find the correct qubit-bath coupling
    H, D = B.fit(delta, T1, T2)
    L = markov.superop(H, D, B)
    t = np.linspace(0, 10, 400)

    # T1 demo
    eq = 1 / (1 + np.exp(delta * B.scale)) # equilibrium rho_11
    s = State('1') # qubit in the |1> state
    out = s.propagate(L, t, lambda x, h: x.ev(p1))
    fig, axes = plt.subplots(nrows=2, ncols=1)
    axes[0].plot(t, out, 'r-', t, eq +(1-eq)*np.exp(-t/T1), 'b-.', [0, t[-1]], [eq, eq], 'k:', linewidth=2)
    axes[0].set_xlabel('$t$ [TU]')
    axes[0].set_ylabel('probability')
    axes[0].set_xlim([0, t[-1]])
    axes[0].set_ylim([0, 1])
    axes[0].set_title('$T_1$: relaxation')
    axes[0].legend((r'simulated $P_1$', r'$P_{1}^{eq} +(1-P_{1}^{eq}) \exp(-t/T_1)$'))

    # T2 demo
    s = State('0')
    s = s.u_propagate(Ry(pi/2)) # rotate to (|0>+|1>)/np.sqrt(2)
    out = s.propagate(L, t, lambda x, h: x.u_propagate(Ry(-pi/2)).ev(p0))
    axes[1].plot(t, out, 'r-', t, 0.5*(1+np.exp(-t/T2)), 'b-.', linewidth=2)
    axes[1].set_xlabel('$t$ [TU]')
    axes[1].set_ylabel('probability')
    axes[1].set_xlim([0, t[-1]])
    axes[1].set_ylim([0, 1])
    axes[1].set_title('$T_2$: dephasing')
    axes[1].legend((r'simulated $P_0$', r'$\frac{1}{2} (1+\exp(-t/T_2))$'))
    fig.show()
    return H, D, B



def nmr_sequences(seqs: 'Sequence[Seq]' = None, strength_error: bool = True):
    """NMR control sequences demo.

    Compares the performance of different single-qubit NMR control
    sequences in the presence of systematic errors.
    Plots the fidelity of each control sequence as a function
    of both off-resonance error f and fractional pulse length error g.

    Reproduces fidelity plots in :cite:`Cummins`.

    Args:
      seqs: control sequences to compare
      strength_error: if True use a pulse strength error, otherwise use a pulse length error
    """
    # Ville Bergholm 2006-2021
    print('\n\n=== NMR control sequences for correcting systematic errors ===\n')

    if seqs is None:
        seqs = [seq.nmr([[pi, 0]], name=r'Plain $\pi$ pulse'), seq.corpse(pi), seq.scrofulous(pi), seq.bb1(pi), seq.knill()]

    # Pulse length/timing errors also affect the drift term, pulse strength errors don't.
    if strength_error:
        error_type = 'pulse strength error'
    else:
        error_type = 'pulse length error'

    # off-resonance Hamiltonian (a constant \sigma_z-type drift term) times -1j
    offres_A = -0.5j * sz

    psi = State('0') # initial state
    f = np.arange(-1, 1, 0.05)
    g = np.arange(-1, 1, 0.05)
    nf = len(f)
    ng = len(g)

    fid = np.empty((ng, nf, len(seqs)))

    def plot_trajectory(ax, s_error, title):
        """Apply the sequence on the state psi, plot the evolution.
        """
        out, _ = seq.propagate(psi, s_error, out_func=State.bloch_vector)
        plot.state_trajectory(out, ax=ax)
        ax.set_title(title)

    def plot_slice(x, data, titles, slice_name):
        """Plot the fidelities of a slice in the error plane for every sequence into a new figure.
        """
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(x, data)
        ax.set_xlabel(slice_name)
        ax.set_ylabel('fidelity')
        ax.legend(titles)
        ax.grid(True)
        ax.set_xlim([-1, 1])
        ax.set_ylim([0, 1])
        ax.set_title('Fidelity under ' + slice_name)
        fig.show()

    # loop over the different sequences
    for k, ss in enumerate(seqs):
        fig = plt.figure()
        fig.suptitle(ss.name)
        gs = GridSpec(2, 2)
        U = ss.to_prop()  # target propagator
        #psi_target = [psi.u_propagate(U).bloch_vector()]  # target state

        # State evolution plots.
        # The two systematic error types we are interested here can be
        # incorporated into the control sequence.
        #==================================================
        s_error = deepcopy(ss)
        s_error.A = ss.A + 0.1 * offres_A  # off-resonance error (constant \sigma_z drift term)
        ax = fig.add_subplot(gs[0, 0], projection='3d')
        plot_trajectory(ax, s_error, 'evolution, off-resonance error')

        #==================================================
        s_error = deepcopy(ss)
        if strength_error:
            s_error.control = ss.control * 1.1  # pulse strength error
        else:
            s_error.tau = ss.tau * 1.1  # pulse length error
        ax = fig.add_subplot(gs[1, 0], projection='3d')
        plot_trajectory(ax, s_error, 'evolution, ' + error_type)

        #==================================================
        s_error = deepcopy(ss)

        def u_fidelity(a, b):
            """fidelity of two unitary rotations, [0,1]"""
            return 0.5 * abs(np.trace(a.conj().transpose() @ b))

        # prepare the fidelity contour plot
        for u in range(nf):
            s_error.A = ss.A + f[u] * offres_A  # off-resonance error
            for v in range(ng):
                temp = 1 + g[v]
                if strength_error:
                    s_error.control = ss.control * temp  # pulse strength error
                else:
                    s_error.tau = ss.tau * temp  # pulse length error
                fid[v, u, k] = u_fidelity(U, s_error.to_prop())

        ax = fig.add_subplot(gs[:, 1])
        X, Y = np.meshgrid(f, g)
        ax.contour(X, Y, 1 - fid[:, :, k])
        #ax.surf(X, Y, 1 - fid)
        ax.grid(True)
        ax.set_xlabel('off-resonance error')
        ax.set_ylabel(error_type)
        ax.set_title('fidelity')
        fig.show()

    # horizontal and vertical slices through the error plane
    titles = [s.name for s in seqs]
    plot_slice(f, fid[ng // 2, :, :], titles, 'off-resonance error')
    plot_slice(g, fid[:, nf // 2, :], titles, error_type)



def phase_estimation(t, U, s, implicit=False):
    r"""Quantum phase estimation algorithm.

    Args:
      t (int): number of qubits
      U (Lmap): unitary operator
      s (State): initial state
      implicit (bool): if True, use implicit measurement instead of partial trace

    Returns:
        State: state of the index register after the phase estimation circuit, but before final measurement.

    Estimate an eigenvalue of U using t qubits, starting from the state s.

    To get a result accurate to n bits with probability :math:`\ge 1 -\epsilon`,
    choose

    .. math::

       t \ge n + \left\lceil \log_2\left(2 +\frac{1}{2 \epsilon}\right) \right \rceil.

    See :cite:`Cleve`, :cite:`NC` chapter 5.2.
    """
    # Ville Bergholm 2009-2010

    T = 2 ** t
    S = U.data.shape[0]

    # index register in uniform superposition
    #reg_t = u_propagate(State(0, qubits(t)), gate.walsh(t)) # use Hadamards
    reg_t = State(np.ones(T) / np.sqrt(T), qubits(t)) # skip the Hadamards

    # state register (ignore the dimensions)
    reg_s = State(s, S)

    # full register
    reg = reg_t.tensor(reg_s)

    # controlled unitaries
    for k in range(t):
        ctrl = -np.ones(t, dtype=int)
        ctrl[k] = 1
        temp = gate.controlled(U ** (2 ** (t-1-k)), ctrl)
        reg = reg.u_propagate(temp)
    # from this point forward the state register is not used anymore

    if implicit:
        # save memory and CPU: make an implicit measurement of the state reg, discard the results
        _, _, reg = reg.measure(t, do='D')
        #print('Implicit measurement of state register: {}\n', res)
    else:
        # more expensive computationally: trace over the state register
        reg = reg.ptrace((t,))

    # do an inverse quantum Fourier transform on the index reg
    QFT = gate.qft(qubits(t))
    return reg.u_propagate(QFT.ctranspose())



def phase_estimation_precision(t, U, u=None):
    """Quantum phase estimation demo.

    Args:
      t (int): number of qubits
      U (array): unitary operator
      u (State): Initial state. If not given, use a random eigenvector of U.

    Estimate an eigenvalue of U using t qubits, starting from the state u.
    Plots and returns the probability distribution of the resulting t-bit approximations.

    Uses :func:`phase_estimation`.
    """
    # Ville Bergholm 2009-2010

    print('\n\n=== Phase estimation ===\n\n')

    # find eigenstates of the operator
    N = U.shape[0]
    d, v = np.linalg.eig(U)
    if u is None:
        u = State(v[:, 0], N) # exact eigenstate
        title = 'eigenstate'
    else:
        title = 'given state'

    print('Use {} qubits to estimate the phases of the eigenvalues of a U({}) operator.\n'.format(t, N))
    p = phase_estimation(t, Lmap(U), u).prob()
    T = 2 ** t
    x = np.arange(T) / T
    w = 0.8 / T

    # plot probability distribution
    fig = plt.figure()
    ax = fig.gca()
    ax.bar(x, p, width=w) # TODO align = 'center' ???
    ax.set_xlabel(r'phase / $2\pi$')
    ax.set_ylabel('probability')
    ax.set_title('Phase estimation, ' + title)
    #ax.set_xlim([-1/(T*2), 1-1/(T*2)]) # [0, 1]

    # compare to correct answer
    target = np.angle(d) / (2*pi) + 1
    target -= np.floor(target)
    ax.plot(target, 0.5 * max(p) * np.ones(len(target)), 'mo')

    ax.legend(('Target phases', 'Measurement probability distribution'))
    fig.show()



def qft_circuit(dim=(2, 3, 3, 2)):
    r"""Quantum Fourier transform circuit demo.

    Args:
      dim (Sequence[int]): dimension vector of the subsystems

    Simulate the quadratic quantum circuit for QFT.

    NOTE: If dim is not palindromic the resulting circuit also
    reverses the order of the dimensions in the SWAP cascade.

    .. math::

       U \ket{x_1, x_2, \ldots, x_n}
       = \frac{1}{\sqrt{d}} \sum_{k_i} \ket{k_n,\ldots, k_2, k_1}
         \exp\left(i 2 \pi \left(\sum_{r=1}^n k_r 0.x_r x_{r+1}\ldots x_n\right)\right)
       = \frac{1}{\sqrt{d}} \sum_{k_i} \ket{k_n,\ldots, k_2, k_1}
         \exp\left(i 2 \pi 0.x_1 x_2 \ldots x_n \left(\sum_{r=1}^n d_1 d_2 \cdots d_{r-1} k_r \right)\right).
    """
    # Ville Bergholm 2010-2016

    print('\n\n=== Quantum Fourier transform using a quadratic circuit ===\n')
    print('Subsystem dimensions: {}'.format(dim))

    def Rgate(d):
        r"""R = \sum_{xy} exp(i*2*pi * x*y/prod(dim)) |xy><xy|"""
        temp = np.kron(np.arange(d[0]), np.arange(d[-1])) / np.prod(d)
        return gate.phase(2 * pi * temp, (d[0], d[-1]))

    dim = tuple(dim)
    n = len(dim)
    U = gate.id(dim)

    for k in range(n):
        H = gate.qft((dim[k],))
        U = gate.single(H, k, dim) @ U
        for j in range(k+1, n):
            temp = Rgate(dim[k : j+1])
            U = gate.two(temp, (k, j), dim) @ U

    for k in range(n // 2):
        temp = gate.swap(dim[k], dim[n-1-k])
        U = gate.two(temp, (k, n-1-k), U.dim[0]) @ U

    err = (U - gate.qft(dim)).norm()
    print('Error: {}'.format(err))
    return U



def quantum_channels(p=0.3):
    """Visualization of simple one-qubit channels.

    Visualizes the effect of different quantum channels on a qubit using the Bloch sphere representation.

    Args:
      p (float): probability
    """
    # Ville Bergholm 2009-2012

    print('\n\n=== Quantum channels ===\n')

    I = np.eye(2)
    #t = np.arcsin(np.sqrt(gamma))
    t = pi/3
    channels = {
        'Bit flip': [np.sqrt(1-p)*I, np.sqrt(p)*sx],
        'Phase flip': [np.sqrt(1-p)*I, np.sqrt(p)*sz],
        'Bit and phase flip': [np.sqrt(1-p)*I, np.sqrt(p)*sy],
        'Depolarization': [np.sqrt(1-3*p/4)*I, np.sqrt(p)*sx/2, np.sqrt(p)*sy/2, np.sqrt(p)*sz/2],
        'Amplitude damping': [
            np.sqrt(p) * np.diag([1, np.cos(t)]),
            np.sqrt(p) * np.array([[0, np.sin(t)], [0, 0]]),
            np.sqrt(1-p) * np.diag([np.cos(t), 1]),
            np.sqrt(1-p) * np.array([[0, 0], [np.sin(t), 0]])
        ]
    }

    X, Y, Z = plot.sphere()
    S = np.array([X, Y, Z])

    def present(ax, E, T):
        """Helper"""
        s = S.shape
        res = np.empty(s)

        for a in range(s[1]):
            for b in range(s[2]):
                temp = State.bloch_state(np.r_[1, S[:, a, b]])
                res[:, a, b] = temp.kraus_propagate(E).bloch_vector()[1:]  # skip the normalization

        plot.bloch_sphere(ax)
        ax.plot_surface(res[0], res[1], res[2], rstride=1, cstride=1, color='m', alpha=0.2, linewidth=0)
        ax.set_title(T)

    fig = plt.figure()
    fig.suptitle('Bloch sphere evolution under one-qubit quantum channels')
    n = len(channels)
    gs = GridSpec(2, (n+1) // 2)
    for k, (title, ch) in enumerate(channels.items()):
        ax = fig.add_subplot(gs[k], projection='3d')
        present(ax, ch, title)
    fig.show()



def quantum_walk(steps=8, n=21, p=0.05, n_coin=2):
    """Quantum random walk demo.

    Simulates a wrapping 1D quantum walker controlled by a unitary quantum coin.
    On each step the coin is flipped and the walker moves either to the left
    or to the right depending on the result.
    A three-dimensional coin introduces a third option, staying put.

    Args:
      steps (int): number of steps
      n (int): number of possible walker positions
      p (float): probability of measuring the walker after each step
      n_coin (int): coin dimension, 2 or 3

    Returns:
        State: walker state at the end of the walk

    NOTE: p=1 results in a fully classical random walk, whereas
    p=0 corresponds to the "fully quantum" case.
    """
    # Ville Bergholm 2010-2018
    H = base.H

    print('\n\n=== Quantum walk ===\n')
    # initial state: coin shows heads, walker in center node
    coin = State('0', n_coin)
    walker = State(n // 2, n)

    s = coin.tensor(walker).to_op(inplace=True)

    # translation operators (wrapping)
    left = gate.mod_inc(-1, n).data.toarray()
    right = left.conj().T
    stay = np.eye(n)

    if n_coin == 2:
        # coin flip operator
        #C = Rx(pi / 2)
        C = H
        # shift operator: heads, move left; tails, move right
        S = np.kron(p0, left) +np.kron(p1, right)
        #S = np.kron(p0, stay) +np.kron(p1, right)
    elif n_coin == 3:
        C = rand_U(n_coin)
        S = np.kron(np.diag([1, 0, 0]), left) +np.kron(np.diag([0, 1, 0]), right) +np.kron(np.diag([0, 0, 1]), stay)
    else:
        raise ValueError('Coin dimension must be 2 or 3.')

    # propagator for a single step (flip + shift)
    U = S @ np.kron(C, np.eye(n))

    # Kraus ops for position measurement
    M = []
    for k in range(n):
        temp = np.zeros((n, n), dtype=complex)
        temp[k, k] = np.sqrt(p)
        M.append(np.kron(np.eye(n_coin), temp))
    # "no measurement"
    M.append(np.sqrt(1-p) * np.eye(n_coin * n))

    for k in range(steps):
        s = s.u_propagate(U)
        #s = s.kraus_propagate(M)
        s.data = p * np.diag(np.diag(s.data)) + (1 - p) * s.data  # equivalent but faster...
    s = s.ptrace([0]) # ignore the coin

    # plot final state and position
    fig, ax = plt.subplots()
    s.plot(fig)
    temp = ' after {} steps, p = {}'.format(steps, p)
    ax.set_title('Walker state' + temp)
    fig.show()

    fig, ax = plt.subplots()
    ax.bar(np.arange(n) - n // 2, s.prob(), width=0.8)
    ax.set_title('Walker position' + temp)
    ax.set_xlabel('position')
    ax.set_ylabel('probability')
    fig.show()
    return s


def qubit_and_resonator(d_r=30):
    """Qubit coupled to a microwave resonator demo.

    Simulates a qubit coupled to a microwave resonator.
    Reproduces plots from the experiment in :cite:`Hofheinz`.

    Args:
      d_r (int): resonator truncation dimension
    """
    # Ville Bergholm 2010-2014

    print('\n\n=== Qubit coupled to a single-mode microwave resonator ===\n')

    d_r = max(d_r, 10)

    #omega0 = 1e9 # Energy scale/\hbar, Hz
    #T = 0.025 # K

    # omega_r = 2*pi* 6.570 # GHz, resonator angular frequency
    # qubit-resonator detuning Delta(t) = omega_q(t) -omega_r

    Omega     =  2*pi* 19e-3  # GHz, qubit-resonator coupling
    Delta_off = -2*pi* 463e-3 # GHz, detuning at off-resonance point
    Omega_q   =  2*pi* 0.5    # GHz, qubit microwave drive amplitude
    #  Omega << |Delta_off| < Omega_q << omega_r

    # decoherence times
    #T1_q = 650e-9 # s
    #T2_q = 150e-9 # s
    #T1_r = 3.5e-6 # s
    #T2_r = 2*T1_r # s

    # TODO heat bath and couplings
    #bq = markov.MarkovianBath('ohmic', omega0, T)
    #[H, Dq] = fit(bq, Delta_off, T1_q, T2_q)
    #[H, Dr] = fit_ho(bq, ???, T1_r, T2_r???)
    #D = np.kron(np.eye(d_r), D) # +np.kron(Dr, qit.I)

    #=================================
    # Hamiltonians etc.

    # qubit raising and lowering ops
    sp = 0.5*(sx -1j*sy)
    sm = sp.conj().transpose()

    # resonator annihilation op
    a = boson_ladder(d_r)
    # resonator identity op
    I_r = np.eye(d_r)

    # qubit H
    Hq = np.kron(I_r, sp @ sm)
    # resonator H
    #Hr = np.kron(a'*a, I_q)

    # coupling H, rotating wave approximation
    Hint = Omega/2 * (np.kron(a, sp) +np.kron(a.conj().transpose(), sm))
    # microwave control H
    HOq = lambda ampl, phase: np.kron(I_r, ampl * 0.5 * Omega_q * (np.cos(phase) * sx + np.sin(phase) * sy))
    #Q = ho.position(d_r)
    #P = ho.momentum(d_r)
    #HOr = lambda ampl, phase: np.kron(ampl*Omega_r/np.sqrt(2)*(np.cos(phase)*Q +np.sin(phase)*P), qit.I)

    # system Hamiltonian, in rotating frame defined by H0 = omega_r * (Hq + Hr)
    # D = omega_q - omega_r is the detuning between the qubit and the resonator
    H = lambda D, ampl, phase: D * Hq + Hint + HOq(ampl, phase)

    # readout: qubit in excited state?
    readout = lambda s, h: s.ev(np.kron(I_r, p1))

    s0 = State(0, (d_r, 2)) # resonator + qubit, ground state


    #=================================
    # Rabi test

    t = np.linspace(0, 500, 100)
    detunings = np.linspace(0, 2*pi* 40e-3, 100) # GHz
    out = np.empty((len(detunings), len(t)))

    #L = markov.superop(H(0, 1, 0), D, bq)
    L = H(0, 1, 0)  # zero detuning, sx pulse
    for k, d in enumerate(detunings):
        s = s0.propagate(L, (2/Omega_q)*pi/2) # Q (pi pulse for the qubit)
        #LL = markov.superop(H(d, 0, 0), D, bq)
        LL = H(d, 0, 0)  # detuned propagation
        out[k, :] = s.propagate(LL, t, out_func=readout)

    fig, ax = plt.subplots()
    plot.pcolor(ax, out, t, detunings / (2*pi*1e-3))
    #fig.colorbar(orientation = 'horizontal')
    ax.set_xlabel(r'Interaction time $\tau$ (ns)')
    ax.set_ylabel(r'Detuning, $\Delta/(2\pi)$ (MHz)')
    ax.set_title('One photon Rabi-swap oscillations between qubit and resonator, $P_e$')
    fig.show()

    #figure
    #f = fft(out, [], 2)
    #pcolor(abs(fftshift(f, 2)))
    #=================================
    # state preparation

    def demolish_state(targ):
        """Convert a desired (possibly truncated) resonator state ket into a program for constructing that state.
        State preparation in reverse, uses approximate idealized Hamiltonians.
        """
        # Ideal H without interaction
        A = lambda D, ampl, phase: D*Hq +HOq(ampl, phase)

        # resonator ket into a full normalized qubit+resonator state
        n = len(targ)
        targ = State(np.r_[targ, np.zeros(d_r-n)], (d_r,)).normalize().tensor(State(0, (2,)))
        t = deepcopy(targ)

        n = n-1 # highest excited level in resonator
        prog = np.zeros((n, 4))
        for k in reversed(range(n)):
            # |k+1,g> to |k,e>
            dd = targ.data.reshape(d_r, 2)
            prog[k, 3] = (np.angle(dd[k, 1]) -np.angle(dd[k+1, 0]) -pi/2 +2*pi) / -Delta_off
            targ = targ.propagate(A(-Delta_off, 0, 0), prog[k, 3]) # Z

            dd = targ.data.reshape(d_r, 2)
            prog[k, 2] = (2/(np.sqrt(k+1) * Omega)) * np.arctan2(abs(dd[k+1, 0]), abs(dd[k, 1]))
            targ = targ.propagate(-Hint, prog[k, 2]) # S

            # |k,e> to |k,g>
            dd = targ.data.reshape(d_r, 2)
            phi = np.angle(dd[k, 1]) -np.angle(dd[k, 0]) +pi/2
            prog[k, 1] = phi
            prog[k, 0] = (2/Omega_q) * np.arctan2(abs(dd[k, 1]), abs(dd[k, 0]))
            targ = targ.propagate(A(0, -1, phi), prog[k, 0]) # Q
        return prog, t

    def prepare_state(prog):
        """Prepare a state according to the program."""
        s = s0 # start with ground state
        #s = tensor(ho.coherent_state(0.5, d_r), State(0, 2)) # coherent state
        for k in prog:
            # Q, S, Z
            s = s.propagate(H(0, 1, k[1]), k[0]) # Q
            s = s.propagate(H(0, 0, 0), k[2]) # S
            s = s.propagate(H(Delta_off, 0, 0), k[3]) # Z
        return s


    #=================================
    # readout plot (not corrected for limited visibility)

    prog, dummy = demolish_state([0, 1, 0, 1]) # |1> + |3>
    s1 = prepare_state(prog)

    prog, dummy = demolish_state([0, 1, 0, 1j]) # |1> + i|3>
    s2 = prepare_state(prog)

    t = np.linspace(0, 350, 200)
    out1 = s1.propagate(H(0, 0, 0), t, readout)
    out2 = s2.propagate(H(0, 0, 0), t, readout)

    fig, ax = plt.subplots()
    ax.plot(t, out1, 'b-', t, out2, 'r-')
    ax.set_xlabel(r'Interaction time $\tau$ (ns)')
    ax.set_ylabel('$P_e$')
    ax.set_title('Resonator readout through qubit.')
    ax.legend((r'$|1\rangle + |3\rangle$', r'$|1\rangle + i|3\rangle$'))
    fig.show()

    #=================================
    # Wigner spectroscopy

    if False:
        # "voodoo cat"
        targ = np.zeros(d_r)
        for k in range(0, d_r, 3):
            targ[k] = (2 ** k) / np.sqrt(sps.factorial(k))
    else:
        targ = [1, 0, 0, np.exp(1j * pi * 3/8), 0, 0, 1]

    # calculate the pulse sequence for constructing targ
    prog, t = demolish_state(targ)
    s = prepare_state(prog)

    print('Trying to prepare the state')
    print(t)
    print('Fidelity of prepared state with target state: {:g}'.format(fidelity(s, t)))
    print('Time required for state preparation: {:g} ns'.format(np.sum(prog[:, [0, 2, 3]])))
    print('\nComputing the Wigner function...')
    s = s.ptrace((1,))
    W, a, b = ho.wigner(s, res=(80, 80), lim=(-2.5, 2.5, -2.5, 2.5))
    fig, ax = plt.subplots()
    plot.pcolor(ax, W, a, b, (-1, 1))
    ax.set_title(r'Wigner function $W(\alpha)$')
    ax.set_xlabel(r'Re($\alpha$)')
    ax.set_ylabel(r'Im($\alpha$)')
    fig.show()


def shor_factorization(N=9, cheat=False):
    """Shor's factorization algorithm demo.

    Simulates Shor's factorization algorithm.

    Args:
      N (int): integer to factorize
      cheat (bool): If False, simulates the full algorithm. Otherwise avoids the quantum part.

    Returns:
        int: a factor of N

    NOTE: This is a very computationally intensive quantum algorithm
    to simulate classically, and probably will not run for any
    nontrivial value of N (unless you choose to cheat, in which case
    instead of simulating the quantum part (implemented in :func:`find_order`)
    we use a more efficient classical algorithm for the order-finding).

    See :cite:`Shor`, :cite:`NC` chapter 5.3.
    """
    # Ville Bergholm 2010-2011

    def find_order_cheat(a, N):
        """Classical order-finding algorithm."""
        for r in range(1, N+1):
            if mod_pow(a, r, N) == 1:
                return r
        return None

    def mod_pow(a, x, N):
        """Computes mod(a^x, N) using repeated squaring mod N.
        x must be a positive integer.
        """
        X = numstr_to_array(bin(x)[2:]) # exponent in big-endian binary
        res = 1
        for b in reversed(X): # little-endian
            if b:
                res = mod(res*a, N)
            a = mod(a*a, N) # square a
        return res


    print('\n\n=== Shor\'s factorization algorithm ===\n')
    if cheat:
        print('(cheating)\n')

    # number of bits needed to represent mod N arithmetic:
    m = int(np.ceil(np.log2(N)))
    print('Trying to factor N = {} ({} bits).'.format(N, m))

    # maximum allowed failure probability for the quantum order-finding part
    epsilon = 0.25
    # number of index qubits required for the phase estimation
    t = 2*m +1 +int(np.ceil(np.log2(2 + 1 / (2 * epsilon))))
    print('The quantum order-finding subroutine will need {} + {} qubits.\n'.format(t, m))

    # classical reduction of factoring to order-finding
    while True:
        a = npr.randint(2, N) # random integer, 2 <= a < N
        print('Random integer: a = {}'.format(a))

        p = gcd(a, N)
        if p != 1:
            # a and N have a nontrivial common factor p.
            # This becomes extremely unlikely as N grows.
            print('Lucky guess, we found a common factor!')
            break

        print('Trying to find the period of f(x) = a^x mod N')

        if cheat:
            # classical cheating shortcut
            r = find_order_cheat(a, N)
        else:
            while True:
                print('.')
                # ==== quantum part of the algorithm ====
                [s1, r1] = find_order(a, N, epsilon)
                [s2, r2] = find_order(a, N, epsilon)
                # =============  ends here  =============

                if gcd(s1, s2) == 1:
                    # no common factors
                    r = lcm(r1, r2)
                    break

        print('\n  =>  r = {}'.format(r))

        # if r is odd, try again
        if gcd(r, 2) == 2:
            # r is even

            x = mod_pow(a, r // 2, N)
            if mod(x, N) != N-1:
                # factor found
                p = gcd(x-1, N) # ok?
                if p == 1:
                    p = gcd(x+1, N) # no, try this
                break

            print('a^(r/2) = -1 (mod N), try again...\n')
        else:
            print('r is odd, try again...\n')

    print('\nFactor found: {}'.format(p))
    return p


def find_order(a, N, epsilon=0.25):
    """Quantum order-finding subroutine.

    Finds the period of the function f(x) = a^x mod N.

    Args:
        a (int): base
        N (int): modulus
        epsilon (float): maximum allowed failure probability for the subroutine

    N and epsilon together determine the number of qubits, m required to perform the modular
    arthmetic, and t to serve as the index in the phase estimation.

    Returns:
        int, int: (r, s) where ``r/s`` approximates period / 2**t.

    Uses :func:`phase_estimation`.

    See :cite:`Shor`, :cite:`NC` chapter 5.3.
    """
    # number of bits needed to represent mod N arithmetic:
    m = int(np.ceil(np.log2(N)))
    # number of index qubits required for the phase estimation
    t = 2*m +1 +int(np.ceil(np.log2(2 + 1 / (2 * epsilon))))

    T = 2 ** t # index register dimension
    M = 2 ** m # state register dimension

    # applying f(x) is equivalent to the sequence of controlled modular multiplications in phase estimation
    U = gate.mod_mul(a, M, N)

    # state register initialized in the state |1>
    st = State(1, M)

    # run the phase estimation algorithm
    reg = phase_estimation(t, U, st, implicit=True) # use implicit measurement to save memory

    # measure index register
    dummy, num = reg.measure()

    def find_denominator(x, y, max_den):
        r"""
        Finds the denominator s for r/s \approx x/y such that s < max_den
        using the convergents of the continued fraction representation

        .. math::

           \frac{x}{y} = a_0 +\frac{1}{a_1 +\frac{1}{a_2 +\ldots}}

        We use floor and mod here, which could be both efficiently implemented using
        the classical Euclidean algorithm.
        """
        d_2 = 1
        d_1 = 0
        while True:
            a = x // y # integer part == a_n
            temp = a*d_1 +d_2 # n:th convergent denumerator d_n = a_n*d_{n-1} +d_{n-2}
            if temp >= max_den:
                break

            d_2 = d_1
            d_1 = temp
            temp = mod(x, y)  # x - a*y # subtract integer part
            if temp == 0:
            #if (temp/y < 1 / (2*max_den ** 2))
                break  # continued fraction terminates here, result is exact

            # invert the remainder (swap numerator and denominator)
            x = y
            y = temp
        return d_1

    # another classical part
    s = find_denominator(num, T, T+1)
    r = (num * s) // T
    return r, s


def superdense_coding(d=2):
    """Superdense coding demo.

    Simulate Alice sending two d-its of information to Bob using
    a shared EPR qudit pair.

    Args:
      d (int): qudit dimension
    """
    # Ville Bergholm 2010-2011

    print('\n\n=== Superdense coding ===\n')

    H = gate.qft(d)           # qft (generalized Hadamard) gate
    add = gate.mod_add(d, d)  # modular adder (generalized CNOT) gate
    I = gate.id(d)

    dim = (d, d)

    # EPR preparation circuit
    U = add @ tensor(H, I)

    print('Alice and Bob start with a shared EPR pair:')
    reg = State('00', dim).u_propagate(U)
    print(reg)

    # two random d-its
    a = np.floor(d * npr.rand(2)).astype(int)
    print('\nAlice wishes to send two d-its of information (d = {}) to Bob: a = {}.'.format(d, a))

    Z = H @ gate.mod_inc(a[0], d) @ H.ctranspose()
    X = gate.mod_inc(-a[1], d)

    print('Alice encodes the d-its to her half of the EPR pair using local transformations,')
    reg = reg.u_propagate(tensor(Z @ X, I))
    print(reg)

    print('\nand sends it to Bob. He then disentangles the pair,')
    reg = reg.u_propagate(U.ctranspose())
    print(reg)

    _, b = reg.measure()
    b = np.array(np.unravel_index(b, dim))
    print('\nand measures both qudits in the computational basis, obtaining the result  b = {}.'.format(b))

    if all(a == b):
        print('The d-its were transmitted succesfully.')
    else:
        raise RuntimeError('Should not happen.')


def teleportation(d=2):
    """Quantum teleportation demo.

    Simulate the teleportation of a d-dimensional qudit from Alice to Bob.

    Args:
      d (int): qudit dimension
    """
    # Ville Bergholm 2009-2011

    print('\n\n=== Quantum teleportation ===\n')

    H = gate.qft(d)           # qft (generalized Hadamard) gate
    add = gate.mod_add(d, d)  # modular adder (generalized CNOT) gate
    I = gate.id(d)

    dim = (d, d)

    # EPR preparation circuit
    U = add @ tensor(H, I)

    print('Alice and Bob start with a shared EPR pair:')
    epr = State('00', dim).u_propagate(U)
    print(epr)

    print('\nAlice wants to transmit this payload to Bob:')
    payload = State('0', d).u_propagate(rand_SU(d)).fix_phase()
    # choose a nice global phase
    print(payload)

    print('\nThe total |payload> \\otimes |epr> register looks like')
    reg = payload.tensor(epr)
    print(reg)

    print('\nNow Alice entangles the payload with her half of the EPR pair,')
    reg = reg.u_propagate(tensor(U.ctranspose(), I))
    print(reg)

    _, b, reg = reg.measure((0, 1), do='C')
    b = np.array(np.unravel_index(b, dim))
    print('\nand measures her qudits, getting the result {}.'.format(b))
    print('She then transmits the two d-its to Bob using a classical channel.')
    print('Since Alice\'s measurement has unentangled the state,')
    print('Bob can ignore her qudits. His qudit now looks like')
    reg_B = reg.ptrace((0, 1)).to_ket() # pure state
    print(reg_B)

    print('\nUsing the two classical d-its of data Alice sent him,')
    print('Bob performs a local transformation on his half of the EPR pair.')
    Z = H @ gate.mod_inc(b[0], d) @ H.ctranspose()
    X = gate.mod_inc(-b[1], d)
    reg_B = reg_B.u_propagate(Z @ X).fix_phase()
    print(reg_B)

    ov = fidelity(payload, reg_B)
    print('\nThe overlap between the resulting state and the original payload state is |<payload|B>| = {}'.format(ov))
    if abs(ov-1) > TOLERANCE:
        raise RuntimeError('Should not happen.')

    print('The payload state was succesfully teleported from Alice to Bob.')
    return reg_B, payload


def werner_states(d=2):
    """Werner and isotropic states demo.

    Plots some properties of d-dimensional family of Werner states and their
    dual isotropic states as a function of the parameter p.

    Args:
      d (int): qudit dimension
    """
    # Ville Bergholm 2014-2016

    print('\n\n=== Werner and isotropic states ===\n\n')

    # cover both Werner ([0,1]) and the dual isotropic states
    pp = np.linspace(0, (d+1)/2, 200)
    res = np.empty((len(pp), 3))
    for k, p in enumerate(pp):
        w = State.werner(p, d)
        # corresponding isotropic state
        iso = w.ptranspose(1)
        res[k, :] = [w.purity(), w.lognegativity(1), iso.lognegativity(1)]
        #res2[k] = iso.purity()

    fig, ax = plt.subplots()
    leg = ['Werner/isotropic purity', 'Werner lognegativity', 'Isotropic lognegativity',
           'maximally mixed state', 'maximally entangled generalized Bell state']
    ax.plot(pp, res)
    # fully depolarized state
    p_mix = (d+1)/(2*d)
    ax.plot(p_mix, 0, 'ko')
    # generalized Bell state
    p_bell = (d+1)/2
    ax.plot(p_bell, 0, 'rs')
    if d == 2:
        # singlet state
        p_singlet = 0
        ax.plot(p_singlet, 0, 'bs')
        leg.append('singlet state')

    ax.set_title('Werner and isotropic states in d = {}'.format(d))
    ax.set_xlabel('Werner state parameter p')
    ax.legend(leg)
    ax.grid(True)
    fig.show()


def markovian_bath_properties():
    """Markovian bath properties demo.
    """
    bath = markov.MarkovianBath('ohmic', stat='boson', TU=1e-9, T=1e-3)
    #bath.set_cutoff('smooth', 1.0)
    bath.plot_bath_correlation()
    bath.plot_spectral_correlation_vs_cutoff()
    bath.plot_spectral_correlation()



def tour():
    """Guided tour to the quantum information toolkit.
    """
    print("""This is the guided tour for the Quantum Information Toolkit.
It should be run in the interactive mode, 'ipython --pylab'.
Between examples, press enter to proceed to the next one.""")

    def pause():
        input('Press enter.')
        #plt.waitforbuttonpress()

    if 1:
        # simple demos
        teleportation()
        pause()

        superdense_coding()
        pause()

        bb84(40)
        pause()

        qft_circuit((2, 3, 2))
        pause()

        werner_states(2)
        werner_states(3)
        pause()

        # sequences and channels
        quantum_channels()
        pause()

        nmr_sequences()
        pause()

        # algorithms
        bernstein_vazirani(6, linear=True)
        bernstein_vazirani(6, linear=False)
        pause()

        grover_search(6)
        pause()

        phase_estimation_precision(5, rand_U(4))
        phase_estimation_precision(5, rand_U(4), State(0, [4]))
        pause()

        adiabatic_qc_3sat(5, 25)
        pause()

        # shor_factorization(9) # TODO need to use sparse matrices to get even this far(!)
        shor_factorization(311 * 269, cheat=True)
        pause()

        # simulations
        markov_decoherence(7e-10, 1e-9)
        pause()

        quantum_walk()
        pause()

        qubit_and_resonator(20)
        pause()

        markovian_bath_properties()
        pause()


if __name__ == '__main__':
    tour()
