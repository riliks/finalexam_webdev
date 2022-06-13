"""
Quantum states.

In QIT, quantum states are represented by the :class:`State` class,
defined in this module.
"""
# Ville Bergholm 2008-2020
# pylint: disable=too-many-statements,too-many-locals,too-many-public-methods

from __future__ import annotations

import collections
from collections.abc import Sequence
from copy import deepcopy
import itertools
import numbers
from typing import Optional, Union

import numpy as np
import scipy as sp
import scipy.sparse as sparse
import scipy.integrate
import scipy.linalg as spl

import qit.lmap as lmap
import qit.gate as gate
from qit.base import sy, Q_Bell, TOLERANCE
from qit.lmap import Lmap
from qit.utils import (_warn, vec, inv_vec, qubits, expv, tensorsum,
                       eighsort, spectral_decomposition, majorize, tensorbasis)



__all__ = ['equal_dims', 'index_muls', 'State', 'fidelity', 'trace_dist']



def equal_dims(s, t):
    """True if states s and t have equal dimensions."""
    return s.dims() == t.dims()


def index_muls(dim):
    """Index multipliers for C-ordered data.

    Args:
        dim (array[int]): dimension vector
    Returns:
        array[int]: index multiplier vector

    .. code-block:: python

      ravel_multi_index(s, dim) == index_muls(dim) @ s
    """
    if len(dim) == 0:
        return np.array(())
    muls = np.roll(np.cumprod(dim[::-1]), 1)[::-1]
    muls[-1] = 1  # muls == [d_{n-1}*...*d_1, d_{n-1}*...*d_2, ..., d_{n-1}, 1]
    return muls



class State(Lmap):
    """Class for quantum states.

    Describes the state (pure or mixed) of a discrete, possibly composite quantum system.
    The subsystem dimensions can be obtained with the :meth:`dims` method (big-endian ordering).

    State class instances are special cases of :class:`.Lmap`. They have exactly two indices.
    If ``self.dim[1] == (1,)``, it is a ket representing a pure state.
    Otherwise both indices must have equal dimensions and the object represents a state operator.

    Does not require the state to be physical (it does not have to be trace-1, Hermitian, or nonnegative).

    By default, all :class:`State` methods leave the object unchanged and instead return a modified copy.


    .. rubric:: Utilities

    .. autosummary::
       check
       subsystems
       dims
       clean_selection
       invert_selection
       fix_phase
       normalize
       to_ket
       to_op
       trace
       ptrace
       ptranspose
       reorder
       tensor
       plot


    .. rubric:: Physics

    .. autosummary::
       ev
       var
       prob
       projector
       u_propagate
       propagate
       kraus_propagate
       measure


    .. rubric:: Quantum information

    .. autosummary::
       fidelity
       trace_dist
       purity
       schmidt
       entropy
       concurrence
       negativity
       lognegativity
       scott
       locc_convertible


    .. rubric:: Other state representations

    .. autosummary::
       bloch_vector
       bloch_state


    .. rubric:: Named states

    .. autosummary::
       werner
       isotropic
    """
    def __init__(self, s: Union[str, int, 'array_like', State], dim: Optional[Sequence[int]] = None):
        """
        Args:
            s: state description, see below
            dim: Dimensions of the subsystems comprising the state.
                If ``dim`` is None, the dimensions are inferred from ``s``.

        .. code-block:: python

              calling syntax            result
              ==============            ======
              State('00101')            standard basis ket |00101> in a five-qubit system
              State('02', (2, 3))       standard basis ket |02> in a qubit+qutrit system
              State('GHZ', (2, 2, 2))   named states (in this case the three-qubit GHZ state)
              State(k, (2, 3))          linearized standard basis ket |k> in a qubit+qutrit system, k must be a nonnegative integer
              State(rand(4))            ket, infer dim = (4,)
              State(rand(4), (2, 2))    ket, two qubits
              State(rand(4,4))          state operator, infer dim = (4,)
              State(rand(6,6), (3, 2))  state operator, qutrit+qubit

              State(s)                  (s is a State) copy constructor
              State(s, dim)             (s is a State) copy constructor, redefine the dimensions

        The currently supported named states are
            GHZ (Greenberger-Horne-Zeilinger),
            W,
            Bell1, Bell2, Bell3, Bell4
        """
        # we want a tuple for dim
        if isinstance(dim, collections.abc.Iterable):
            dim = tuple(dim)
            if not dim:
                dim = (1,)
        elif np.isscalar(dim):
            dim = (dim,)

        if isinstance(s, Lmap):
            # copy constructor
            # state vector or operator? (also works with dim == None)
            if s.is_ket():
                dim = (dim, (1,))
            else:
                if s.dim[0] != s.dim[1]:
                    raise ValueError('State operator must be square.')
                dim = (dim, dim)
            super().__init__(s, dim)
            return

        if isinstance(s, str):
            if s[0].isalpha():
                # named state
                name = s.lower()
                if dim is None:
                    dim = (2, 2, 2) # default: three-qubit state
                n = len(dim) # subsystems
                s = np.zeros(np.prod(dim)) # ket
                dmin = min(dim)

                if name in ('bell1', 'bell2', 'bell3', 'bell4'):
                    # Bell state
                    dim = (2, 2)
                    s = deepcopy(Q_Bell[:, ord(name[-1]) - ord('1')])
                elif name == 'ghz':
                    # Greenberger-Horne-Zeilinger state
                    for k in range(dmin):
                        s[np.ravel_multi_index(n*(k,), dim)] = 1
                elif name == 'w':
                    # W state
                    ind = 1
                    for k in reversed(range(n)):
                        s[ind * (dim[k] - 1)] = 1
                        ind *= dim[k]
                else:
                    raise ValueError("Unknown named state '{}'.".format(name))

                s /= spl.norm(s) # normalize

            else:
                # number string defining a standard basis ket
                if dim is None:
                    n = len(s)  # number of subsystems
                    dim = qubits(n)  # assume they're qubits

                # calculate the linear index
                s = lmap.numstr_to_array(s)
                if any(s >= dim):
                    raise ValueError('Invalid basis ket.')

                ind = np.ravel_multi_index(s, dim)
                s = np.zeros(np.prod(dim)) # ket
                s[ind] = 1

            dim = (dim, (1,))  # ket

        elif isinstance(s, numbers.Integral):
            # integer defining a standard basis ket
            if dim is None:
                raise ValueError('Need system dimension.')

            ind = s
            temp = np.prod(dim)  # total number of states
            if ind >= temp:
                raise ValueError('Invalid basis ket.')

            s = np.zeros(temp) # ket
            s[ind] = 1
            dim = (dim, (1,))  # ket

        else:
            # valid ndarray initializer representing a state vector or a state op
            s = np.array(s)
            if not 1 <= s.ndim <= 2:
                raise ValueError('State must be given as a state vector or a state operator.')

            # state vector or operator?
            if s.ndim == 2 and s.shape[1] != 1:
                if s.shape[0] != s.shape[1]:
                    raise ValueError('State operator matrix must be square.')
                dim = (dim, dim)  # op
            else:
                dim = (dim, (1,))  # ket

        # now s is an ndarray
        # call the Lmap constructor
        super().__init__(s, dim)

# utility methods
# TODO design issue: for valid states, lots of these funcs should return reals (marked with a commented-out .real).
# Should we just drop the imaginary part? what about if the state is invalid, how will the user know? what about numerical errors?
# TODO same thing except with normalization, should we assume states are normalized?

    def check(self):
        """Checks the validity of the state.

        Makes sure it is normalized, and if an operator, Hermitian and semipositive.
        """
        ok = True
        if abs(self.trace() - 1) > TOLERANCE:
            _warn('State not properly normalized.')
            ok = False

        if not self.is_ket():
            if spl.norm(self.data - self.data.conj().transpose()) > TOLERANCE:
                _warn('State operator not Hermitian.')
                ok = False
            if min(spl.eigvalsh(self.data)) < -TOLERANCE:
                _warn('State operator not semipositive.')
                ok = False

        if not ok:
            raise ValueError('Not a valid state.')


    def subsystems(self):
        """Number of subsystems in the state.

        Returns:
            int: number of subsystems
        """
        return len(self.dim[0])


    def dims(self):
        """Dimensions of the subsystems of the state.

        Returns:
            tuple[int]: dimensions of the subsystems
        """
        return self.dim[0] # dims of the other index must be equal (or 1)


    def clean_selection(self, sys):
        """Make a subsystem set unique and sorted.

        Args:
            sys (Iterable[int]): set of subsystem indices

        Returns:
            array[int]: same set made unique and sorted, invalid indices removed
        """
        if not isinstance(sys, collections.abc.Iterable) and np.isscalar(sys):
            sys = [sys]
        temp = set(range(self.subsystems())).intersection(sys)
        return np.array(list(temp), dtype=int)


    def invert_selection(self, sys):
        """Invert and sort a subsystem index set."""
        return np.array(list(set(range(self.subsystems())).difference(sys)), dtype=int)


    def fix_phase(self, inplace=False):
        """Apply a global phase convention to a ket state.

        Returns:
            State: Copy of the state. If the state is represented with
                a ket, the copy has a global phase such that the first nonzero element in the
                state vector is real and positive.
        """
        s = self._inplacer(inplace)
        if s.is_ket():
            # apply the phase convention: first nonzero element in state vector is real, positive
            v = s.data
            for temp in v.flat:
                if abs(temp) > TOLERANCE:
                    phase = temp / abs(temp)
                    v /= phase
                    break
        return s


    def normalize(self, inplace=False):
        """Normalize the state to unity.

        Returns:
            State: Copy of the state normalized to unity.
        """
        s = self._inplacer(inplace)
        if s.is_ket():
            s.data /= spl.norm(s.data)
        else:
            s.data /= np.trace(s.data)
        return s


    def purity(self):
        r"""Purity of the state.

        Returns:
            float: Purity of a normalized state, :math:`p = \mathrm{Tr}(\rho^2)`.
                Equivalent to linear entropy, :math:`S_l = 1-p`.
        """
        if self.is_ket():
            return 1

        # rho is hermitian so purity should be real
        return np.trace(self.data @ self.data) # .real


    def to_ket(self, inplace=False):
        """Convert the state representation into a ket (if possible).

        Returns:
            State: If the state is pure, returns a copy for which the
                internal representation (self.data) is a ket vector.

        Raises:
            ValueError: state is not pure
        """
        s = self._inplacer(inplace)
        if not s.is_ket():
            # state op
            if abs(s.purity() - 1) > TOLERANCE:
                raise ValueError('The state is not pure, and thus cannot be represented by a ket vector.')

            _, v = eighsort(s.data)
            s.data = v[:, [0]]  # corresponds to the highest eigenvalue, i.e. 1
            s.fix_phase(inplace = True)  # clean up global phase
            s.dim = (s.dim[0], (1,))
        return s


    def to_op(self, inplace=False):
        """Convert state representation into a state operator.

        Returns:
            State: Copy of the state for which the internal representation (self.data) is a state operator.
        """
        # slight inefficiency when self.is_ket() and inplace==False: the ket data is copied for no reason
        s = self._inplacer(inplace)
        if s.is_ket():
            s.data = np.outer(s.data, s.data.conj())
            s.dim = (s.dim[0], s.dim[0])
        return s


    def trace(self):
        """Trace of the state operator.

        Returns:
            float: Trace of the state operator. For a pure state this is equal to the squared norm of the state vector.
        """
        if self.is_ket():
            # squared norm, thus always real
            return np.vdot(self.data, self.data).real

        return np.trace(self.data)  # .real


    def ptrace(self, sys, inplace=False):
        """Partial trace.

        Args:
            sys (Sequence[int]): subsystems over which to take a partial trace
        Returns:
            State: Partial trace of the state over the given subsystems.
        """
        s = self.to_op(inplace)
        dim = s.dims()
        n = s.subsystems()
        sys = s.clean_selection(sys)
        keep = s.invert_selection(sys)

        # big-endian (C) data ordering
        # we trace over the subsystems in order, starting from the first one
        # partial trace over single system j, performed for every j in sys
        d = list(dim)
        for j in sys:
            muls = index_muls(d)  # muls == [d_{n-1}*...*d_1, d_{n-1}*...*d_2, ..., d_{n-1}, 1]

            # build the index "stencil"
            inds = np.array([0])
            for k in range(n):
                if k != j:
                    inds = tensorsum(inds, np.r_[0 : muls[k] * d[k] : muls[k]])
                    # np.arange(d[k]) * muls[k]

            stride = muls[j] # stride for moving the stencil while summing
            temp = len(inds)
            res = np.zeros((temp, temp), complex) # result
            for k in range(d[j]):
                temp = inds + stride * k
                res += s.data[np.ix_(temp, temp)]

            s.data = res # replace data
            d[j] = 1  # remove traced-over dimension.

        dim = tuple(np.array(dim)[keep]) # remove traced-over dimensions for good
        if len(dim) == 0:
            dim = (1,) # full trace gives a scalar

        s.dim = (dim, dim)
        return s



    def ptranspose(self, sys, inplace=False):
        """Partial transpose.

        Args:
            sys (Sequence[int]): subsystems wrt. which to take a partial transpose
        Returns:
            State: Partial transpose of the state wrt. the given subsystems.
        """
        if sparse.isspmatrix(self.data):
            self.data = self.data.toarray()
            #self.data = self.data.tolil()
            # TODO FIXME: this conversion is required as long as some scipy sparse matrix classes have not implemented the reshape method.
            # They would also need to support an arbitrary number of dimensions...

        # TODO what about kets? can we do better?
        s = self.to_op(inplace)
        dim = s.dims()
        n = s.subsystems()
        # total dimension
        orig_d = s.data.shape
        # which systems to transpose
        sys = s.clean_selection(sys)

        # swap the transposed dimensions
        perm = np.arange(2 * n)  # identity permutation
        perm[np.r_[sys, sys + n]] = perm[np.r_[sys + n, sys]]

        # flat matrix into tensor, partial transpose, back into a flat matrix
        s.data = s.data.reshape(dim + dim).transpose(perm).reshape(orig_d)
        return s


    def reorder(self, perm, inplace=False):
        """Change the relative order of subsystems in a state.

        .. code-block:: python

          reorder([2, 1, 0])    reverse the order of three subsystems
          reorder([2, 5])       swap subsystems 2 and 5

        Args:
            perm (array[int]): permutation vector,
                may consist of either exactly two subsystem indices
                (to be swapped), or a full permutation of subsystem indices.

        Returns:
            State: The state with the subsystems in the given order.
        """
        # this is just an adapter for Lmap.reorder
        if self.is_ket():
            perm = (perm, None)
        else:
            perm = (perm, perm)
        return super().reorder(perm, inplace = inplace)


# physics methods

    def ev(self, A):
        """Expectation value of an observable in the state.

        Args:
            A (array): hermitian observable
        Returns:
            float: expectation value of the observable A in the state
        """
        # TODO for diagonal A, self.ev(A) == sum(A * self.prob())
        if self.is_ket():
            # state vector
            x = np.vdot(self.data, A @ self.data)
        else:
            # state operator
            x = np.trace(A @ self.data)
        return x.real # .real for a Hermitian observable and valid state


    def var(self, A):
        """Variance of an observable in the state.

        Args:
            A (array): hermitian observable
        Returns:
            float: variance of the observable A in the state
        """
        return self.ev(A @ A) - self.ev(A) ** 2


    def prob(self):
        """Measurement probabilities of the state in the computational basis.

        Returns:
            array[float]: vector of probabilities of finding a system in each of the different states of the computational basis
        """
        if self.is_ket():
            temp = self.data.ravel() # into 1D array
            return (temp * temp.conj()).real  # == np.absolute(self.data) ** 2

        return np.diag(self.data).real # .real


    def projector(self):
        """Projection operator defined by the state.

        Returns the projection operator P defined by the state.
        """
        if abs(self.purity() - 1) > TOLERANCE:
            raise ValueError('The state is not pure, and thus does not correspond to a projector.')

        s = self.to_op()
        return Lmap(s)


    def u_propagate(self, U):
        """Propagate the state using a unitary.

        Args:
            U (array, Lmap): unitary propagator

        Returns:
            State: propagated state
        """
        if isinstance(U, Lmap):
            if self.is_ket():
                return State(U @ self)
            return State((U @ self) @ U.ctranspose())

        if isinstance(U, np.ndarray):
            # U is a matrix, dims do not change. could also construct an Lmap here...
            if self.is_ket():
                return State(U @ self.data, self.dims())
            return State((U @ self.data) @ U.conj().transpose(), self.dims())

        raise TypeError('States can only be propagated using Lmaps and arrays.')


    def propagate(self, G, t, out_func=lambda x, h: deepcopy(x), **kwargs):
        r"""Propagate the state continuously in time.

        .. code-block:: python

          propagate(H, t)                     # Hamiltonian
          propagate(L, t)                     # Liouvillian
          propagate([H, A_1, A_2, ...], t)    # Hamiltonian and Lindblad ops

        Propagates the state using the generator G for the time t,
        returns the resulting state.

        Args:
            G (array, callable, list[array]): generator, see below
            t (float, array[float]): single time duration, or a vector of increasing time instants
            out_func (callable): if given, for each time instance t return out_func(s(t), G(t)).

        Keyword args are passed on to the ODE solver.

        Returns:
            State, list[State]: propagated state for each time instant given in t

        The generator G can either be a

        * Hamiltonian H: :math:`\text{out} = \exp(-i H t) \ket{s}` (or :math:`\exp(-i H t) \rho_s \exp(i H t)`)
        * Liouvillian superoperator L: :math:`\text{out} = \text{inv_vec}(\exp(L t) \text{vec}(\rho_s))`
        * list consisting of a Hamiltonian followed by Lindblad operators.

        For time-dependent cases, G can be a function G(t) which takes a time instance t
        as input and returns the corresponding generator(s).
        """
        # Ville Bergholm 2008-2011
        # James Whitfield 2009

        s = self._inplacer(False)
        if np.isscalar(t):
            t = [t]
        t = np.asarray(t)
        n = len(t) # number of time instances we are interested in
        out = []
        dim = s.data.shape[0]  # system dimension

        if callable(G):
            # time dependent
            t_dependent = True
            F = G
            H = G(0)
        else:
            # time independent
            t_dependent = False
            H = G

        if isinstance(H, np.ndarray):
            # matrix
            dim_H = H.shape[1]
            if dim_H == dim:
                gen = 'H'  # Hamiltonian
            elif dim_H == dim ** 2:
                gen = 'L'  # Liouvillian
                s.to_op(inplace=True)
            else:
                raise ValueError('Dimension of the generator does not match the dimension of the state.')
        elif isinstance(H, list):
            # list: Hamiltonian and the Lindblad operators
            dim_H = H[0].shape[1]
            if dim_H == dim:
                gen = 'A'
                s.to_op(inplace = True)

                # HACK, in this case we use an ODE solver anyway
                if not t_dependent:
                    t_dependent = True
                    F = lambda t: H  # ops stay constant
            else:
                raise ValueError('Dimension of the Lindblad ops does not match the dimension of the state.')
        else:
            raise ValueError("""The second parameter has to be either a matrix, a list,
                             or a function that returns a matrix or a list.""")

        dim = s.data.shape  # may have been switched to operator representation

        if t_dependent:
            # time dependent case, use ODE solver

            # derivative functions for the solver TODO vectorization?
            # H, ket
            def pure_fun(t, y, F):
                "Derivative of a pure state, Hamiltonian."
                return -1j * (F(t) @ y)
            def pure_jac(t, y, F):
                "Jacobian of a pure state, Hamiltonian."
                # pylint: disable=unused-argument
                return -1j * F(t)

            # H, state op
            def mixed_fun(t, y, F):
                "Derivative of a mixed state, Hamiltonian."
                H = -1j * F(t)
                if y.ndim == 1:
                    rho = inv_vec(y, dim)  # into a matrix
                    return vec(H @ rho - rho @ H) # back into a vector

                # vectorization, rows of y
                d = np.empty(y.shape, complex)
                for k, y_k in enumerate(y):
                    rho = inv_vec(y_k, dim) # into a matrix
                    d[k] = vec(H @ rho - rho @ H) # back into a vector
                return d

            # L, state op, same as the H/ket ones, only without the -1j
            def liouvillian_fun(t, y, F):
                "Derivative of a state, Liouvillian."
                return F(t) @ y
            def liouvillian_jac(t, y, F):
                "Jacobian of a state, Liouvillian."
                # pylint: disable=unused-argument
                return F(t)

            # A, state op
            def lindblad_fun(t, y, F):
                "Derivative of a mixed state, Lindbladian."
                X = F(t)  # X == [H, A_1, A_2, ..., A_n]
                H = -1j * X[0] # -1j * Hamiltonian
                Lind = X[1:]   # Lindblad ops
                if y.ndim == 1:
                    rho = inv_vec(y, dim)  # into a matrix
                    temp = H @ rho - rho @ H
                    for A in Lind:
                        ac = 0.5 * (A.conj().transpose() @ A)
                        temp += (A @ rho) @ A.conj().transpose() -ac @ rho -rho @ ac
                    return vec(temp) # back into a vector

                # vectorization, rows of y
                d = np.empty(y.shape, complex)
                for k, y_k in enumerate(y):
                    rho = inv_vec(y_k, dim)  # into a matrix
                    temp = H @ rho - rho @ H
                    for A in Lind:
                        ac = 0.5 * (A.conj().transpose() @ A)
                        temp += (A @ rho) @ A.conj().transpose() -ac @ rho -rho @ ac
                    d[k] = vec(temp)  # back into a vector
                return d

            # what kind of generator are we using?
            if gen == 'H':  # Hamiltonian
                if dim[1] == 1:
                    func, jac = pure_fun, pure_jac
                else:
                    func, jac = mixed_fun, None
            elif gen == 'L':  # Liouvillian
                func, jac = liouvillian_fun, liouvillian_jac
            else: # 'A'  # Hamiltonian and Lindblad operators in a list
                func, jac = lindblad_fun, None

            # do we want the initial state too? (the integrator can't handle t=0!)
            if t[0] == 0:
                out.append(out_func(s, F(0)))
                t = t[1:]

            # ODE solver default parameters
            odeopts = {'rtol' : 1e-4,
                       'atol' : 1e-6,
                       'method' : 'bdf', # 'adams' for non-stiff cases
                       'with_jacobian' : True}
            odeopts.update(kwargs) # user options

            # run the solver
            r = sp.integrate.ode(func, jac).set_integrator('zvode', **odeopts)
            r.set_initial_value(vec(s.data), 0.0).set_f_params(F).set_jac_params(F)
            for k in t:
                r.integrate(k)  # times must be in increasing order, NOT include zero(!)
                if not r.successful():
                    raise RuntimeError('ODE integrator failed.')
                s.data = inv_vec(r.y, dim)
                out.append(out_func(s, F(k)))

        else:
            # time independent case
            if gen == 'H':
                if dim_H < 500:
                    # eigendecomposition
                    d, v = spl.eigh(H)
                    for k in t:
                        # propagator
                        U = (v @ np.diag(np.exp(-1j * k * d))) @ v.conj().transpose()
                        out.append(out_func(s.u_propagate(U), H))
                else:
                    # Krylov subspace method
                    # FIXME imaginary time doesn't yet work
                    w, err, hump = expv(-1j * t, H, s.data)
                    for k in range(n):
                        s.data = w[k, :]  # TODO state ops
                        out.append(out_func(s, H))
            elif gen == 'L':
                # Krylov subspace method
                w, err, hump = expv(t, H, vec(s.data))
                for k in range(n):
                    s.data = inv_vec(w[k, :])
                    out.append(out_func(s, H))

        if len(out) == 1:
            return out[0] # don't bother to wrap a single output in a list
        return out


    def kraus_propagate(self, E):
        r"""Apply a quantum operation to the state.

        Applies the quantum operation E to the state.
        :math:`E = [E_1, E_2, \ldots]` is a set of Kraus operators.
        """
        # Ville Bergholm 2009

        # TODO allow the user to apply E only to some subsystems of s0
        n = len(E)
        # TODO: If n > prod(dims(s))^2, there is a simpler equivalent
        # operation. Should the user be notified?
        def test_kraus(E):
            "Check if E represents a physical quantum operation."
            temp = 0
            for k in E:
                temp += k.conj().transpose() @ k
            if spl.norm(temp.data - np.eye(temp.shape)) > TOLERANCE:
                _warn('Unphysical quantum operation.')

        if self.is_ket():
            if n == 1:
                return self.u_propagate(E[0]) # remains a pure state

        s = self.to_op()
        q = State(np.zeros(s.data.shape, complex), s.dims())
        for k in E:
            q += s.u_propagate(k)
        return q


    def measure(self, M=None, do='R'):
        r"""Quantum measurement.

        .. code-block:: python

          p, res, c
            = measure()                 # measure the entire system projectively
            = measure((1, 4))           # measure subsystems 1 and 4 projectively
            = measure([M_1, M_2, ...])  # perform a general measurement
            = measure(A)                # measure a Hermitian observable A

        Performs a quantum measurement on the state.

        * If no M is given, a full projective measurement in the
          computational basis is performed.

        * If a list/tuple of subsystems is given as the second parameter, only
          those subsystems are measured, projectively, in the
          computational basis.

        * A general measurement may be performed by giving a complete set
          of measurement operators :math:`[M_1, M_2, \ldots]` as the second parameter.
          A POVM can be emulated using :math:`M_i = \sqrt{P_i}` and discarding the collapsed state.

        * Finally, if the second parameter is a single Hermitian matrix A, the
          corresponding observable is measured. In this case the second
          column of p contains the eigenvalue of A corresponding to each
          measurement result.

        ``p = measure(..., do='P')`` returns the vector ``p``, where p[k] is the probability of
        obtaining result k in the measurement. For a projective measurement
        in the computational basis this corresponds to the ket :math:`\ket{k}`.

        ``p, res = measure(...)`` additionally returns the index of the result of the
        measurement, ``res``, chosen at random from the probability distribution ``p``.

        ``p, res, c = measure(..., do='C')`` additionally gives ``c``, the collapsed state
        corresponding to the measurement result ``res``.
        """
        def rand_measure(p):
            """Result of a random measurement using the prob. distribution p."""
            return np.nonzero(np.random.rand() <= np.cumsum(p))[0][0]

        perform = True
        collapse = False
        do = do.upper()
        if do in ('C', 'D'):
            collapse = True
        elif do == 'P':
            perform = False

        d = self.dims()

        if M is None:
            # full measurement in the computational basis
            p = self.prob()  # probabilities
            if perform:
                res = rand_measure(p)
                if collapse:
                    s = State(res, d) # collapsed state

        elif isinstance(M, np.ndarray):
            # M is a matrix TODO Lmap?
            # measure the given Hermitian observable
            a, P = spectral_decomposition(M)
            m = len(a)  # number of possible results

            p = np.zeros((m, 2))
            for k in range(m):
                p[k, 0] = self.ev(P[k])  # probabilities
            p[:, 1] = a  # corresponding measurement results

            if perform:
                res = rand_measure(p)
                if collapse:
                    # collapsed state
                    ppp = P[res]  # Hermitian projector
                    s = deepcopy(self)
                    if self.is_ket():
                        s.data = (ppp @ s.data) / np.sqrt(p[res, 0])
                    else:
                        s.data = ((ppp @ s.data) @ ppp) / p[res, 0]

        elif isinstance(M, (list, tuple)):
            if isinstance(M[0], numbers.Number):
                # measure a set of subsystems in the computational basis
                sys = self.clean_selection(M)
                d = np.array(d)

                # dimensions of selected subsystems and identity ops between them
                # TODO sequential measured subsystems could be concatenated as well
                q = len(sys)
                pdims = []
                start = 0  # first sys not yet included
                for k in sys:
                    pdims.append(np.prod(d[start:k])) # identity
                    pdims.append(d[k]) # selected subsys
                    start = k+1

                pdims.append(np.prod(d[start:])) # last identity

                # index multipliers
                muls = index_muls(d[sys])
                # now muls == [..., d_s{q-1}*d_s{q}, d_s{q}, 1]

                m = muls[0] * d[sys][0] # number of possible results == np.prod(d[sys])

                def build_stencil(j, q, pdims, muls):
                    """Projector to state j (diagonal because we project into the computational basis)"""
                    stencil = np.ones(pdims[0]) # first identity
                    for k in range(q):
                        # projector for system k
                        temp = np.zeros(pdims[2*k + 1])
                        temp[int(j / muls[k]) % pdims[2*k + 1]] = 1
                        stencil = np.kron(np.kron(stencil, temp), np.ones(pdims[2*k + 2])) # temp + next identity
                    return stencil

                # sum the probabilities
                p = np.zeros(m)
                born = self.prob()
                for j in range(m):
                    p[j] = build_stencil(j, q, pdims, muls) @ born

                if perform:
                    res = rand_measure(p)
                    if collapse:
                        # collapsed state
                        s = deepcopy(self)
                        R = build_stencil(res, q, pdims, muls) # diagonal of a diagonal projector (just zeros and ones)

                        if do == 'D':
                            # discard the measured subsystems from s
                            d = np.delete(d, sys)
                            keep = (R == 1)  # indices of elements to keep

                            if self.is_ket():
                                s.data = s.data[keep] / np.sqrt(p[res])
                            else:
                                s.data = s.data[:, keep][keep, :] / p[res]

                            s = State(s.data, d)
                        else:
                            if self.is_ket():
                                s.data = R.reshape(-1, 1) * s.data / np.sqrt(p[res]) # collapsed state
                            else:
                                s.data = np.outer(R, R) * s.data / p[res] # collapsed state, HACK
            else:
                # otherwise use set M of measurement operators (assumed complete!)
                m = len(M)

                # probabilities
                p = np.zeros(m)
                for k in range(m):
                    p[k] = self.ev(M[k].conj().transpose() @ M[k])  #  M^\dagger M  is Hermitian
                    # TODO for kets, this is slightly faster:
                    #temp = M[k] @ self.data
                    #p[k] = np.vdot(temp, temp)

                if perform:
                    res = rand_measure(p)
                    if collapse:
                        s = deepcopy(self)
                        if self.is_ket():
                            s.data = (M[res] @ s.data) / np.sqrt(p[res])
                        else:
                            s.data = ((M[res] @ s.data) @ M[res].conj().transpose()) / p[res]
        else:
            raise ValueError('Unsupported input type.')
        if collapse:
            return p, res, s
        if perform:
            return p, res
        return p



# quantum information methods

    def fidelity(self, r):
        r"""Fidelity of two states.

        Fidelity of two state operators \rho and \sigma is defined as
        :math:`F(\rho, \sigma) = \mathrm{Tr} \sqrt{\sqrt{\rho} \sigma \sqrt{\rho}}`.
        For state vectors this is equivalent to the overlap, :math:`F = |\braket{a|b}|`.

        Fidelity is symmetric in its arguments and bounded in the interval [0,1].

        .. todo:: Uhlmann's theorem, Bures metric, monotonicity under TP maps

        .. todo:: quantiki defines fidelity as NC-fidelity^2

        See :cite:`NC`, chapter 9.2.2.

        Args:
          r (state): another state
        Returns:
          float: Fidelity of r and the state itself.
        """
        if not isinstance(r, State):
            raise TypeError('Not a state.')

        if self.is_ket():
            if r.is_ket():
                return abs(np.vdot(self.data, r.data))
            return np.sqrt(np.vdot(self.data, r.data @ self.data).real)

        if r.is_ket():
            return np.sqrt(np.vdot(r.data, self.data @ r.data).real)
        temp = spl.sqrtm(self.data)
        return np.trace(spl.sqrtm((temp @ r.data) @ temp)).real


    def trace_dist(self, r):
        r"""Trace distance of two states.

        Trace distance between state operators r and s is defined as
        :math:`D(r, s) = \frac{1}{2} \mathrm{Tr}(\sqrt{A^\dagger A})`, where A = r-s.

        Equivalently :math:`D(r, s) = \frac{1}{2} \sum_k |\lambda_k|`, where :math:`\lambda_k`
        are the eigenvalues of A (since A is Hermitian).

        .. todo:: stuff in NC

        See :cite:`NC`, chapter 9.2.1.

        Args:
            r (state): another state

        Returns:
            float: Trace distance between r and the state itself.
        """
        if not isinstance(r, State):
            raise TypeError('Not a state.')

        # avoid copying state ops since we just do read-only stuff here
        S = self.to_op() if self.is_ket() else self
        R = r.to_op() if r.is_ket() else r

        A = R.data - S.data
        return 0.5 * np.sum(np.abs(spl.eigvalsh(A)))
        #return 0.5*np.trace(spl.sqrtm(A'*A))



    def schmidt(self, sys=None, full=False):
        r"""Schmidt decomposition.

        .. code-block:: python

          lambda       = schmidt(sys)
          lambda, u, v = schmidt(sys, full=True)

        Calculates the Schmidt decomposition of the (pure) state.
        Subsystems listed in sys constitute part A, the rest forming part B.
        Vector lambda will contain the Schmidt coefficients.

        If required, matrices u and v will contain the corresponding orthonormal
        Schmidt bases for A and B, respectively, as column vectors, i.e.
        :math:`\ket{k}_A = u[:, k], \quad \ket{k}_B = v[:, k]`.

        The state is then given by :math:`\sum_k \lambda_k \ket{k}_A \otimes \ket{k}_B`.

        See :cite:`NC`, chapter 2.5.

        Args:
          sys (Sequence[int]): subsystem indices
          full (bool): if True, returns also the Schmidt bases u and v
        Returns:
          array[float]: Schmidt coefficients
        """
        dim = np.array(self.dims())
        n = self.subsystems()

        if sys is None:
            if n == 2:
                # reasonable choice
                sys = (0,)
            else:
                raise ValueError('Requires a vector of subsystems.')

        try:
            s = self.to_ket()
        except ValueError as exc:
            raise ValueError('Schmidt decomposition is only defined for pure states.') from exc

        # complement of sys, dimensions of the partitions
        sys = s.clean_selection(sys)
        compl = s.invert_selection(sys)
        d1 = np.prod(dim[sys])
        d2 = np.prod(dim[compl])
        perm = np.r_[sys, compl]

        if all(perm == range(n)):
            # nothing to do
            pass
        else:
            # reorder the system according to the partitioning
            s.reorder(perm, inplace = True)

        # order the coefficients into a matrix, take an svd
        if not full:
            return spl.svdvals(s.data.reshape(d1, d2))

        u, s, vh = spl.svd(s.data.reshape(d1, d2), full_matrices = False)
        # note the definition of vh in svd
        return s, u, vh.transpose()



    def entropy(self, sys=None, alpha=1):
        r"""Von Neumann or Renyi entropy of the state.

        The Renyi entropy of order :math:`\alpha`, :math:`S_\alpha(\rho) = \frac{1}{1-\alpha} \log_2 \mathrm{Tr}(\rho^\alpha)`.
        When :math:`\alpha = 1`, this coincides with the von Neumann entropy
        :math:`S(\rho) = -\mathrm{Tr}(\rho \log_2(\rho))`.

        Args:
          sys (None, Sequence[int]): If None, returns the entropy of the state.
            If a vector of subsystem indices, returns the
            entropy of entanglement of the state wrt. the partitioning
            defined by sys. Entropy of entanglement is only defined for pure states.
          alpha (float): Renyi entropy order, >= 0.
        Returns:
          float: Renyi entropy
        """
        if sys is not None:
            s = self.to_ket().ptrace(sys) # partial trace over one partition
        else:
            s = self

        if s.is_ket():
            return 0

        p = spl.eigvalsh(s.data)
        if alpha != 1:
            # Rényi entropy
            return np.log2(np.sum(p ** alpha)) / (1 - alpha)

        # Von Neumann entropy
        p[p == 0] = 1   # avoid trouble with the logarithm
        return -(p @ np.log2(p))


    def concurrence(self, sys=None):
        """Concurrence of the state.

        See :cite:`Wootters`, :cite:`Horodecki`.

        Args:
          sys (Sequence[int]): subsystem indices
        Returns:
          float: Concurrence of the state wrt. the given partitioning.
        """
        # TODO rewrite, check
        if abs(self.trace() - 1) > TOLERANCE:
            _warn('State not properly normalized.')

        dim = self.dims()

        if sys is not None:
            # concurrence between a qubit and a larger system
            if not (len(sys) == 1 and dim[sys] == 2):
                raise ValueError('Concurrence only defined between a qubit and another system.')

            if abs(self.purity() - 1) > TOLERANCE:
                raise ValueError('Not a pure state.')

            # pure state
            #n = len(dim)
            rho_A = self.ptrace(self.invert_selection(sys)) # trace over everything but sys
            return 2 * np.sqrt(spl.det(rho_A.data).real) # = np.sqrt(2*(1-real(trace(temp*temp)))), .real

        # concurrence between two qubits
        if self.subsystems() != 2 or any(dim != np.array([2, 2])):
            # not a two-qubit state
            raise ValueError('Not a two-qubit state.')

        W = np.kron(sy, sy)
        p = self.data
        if self.is_ket():
            # ket
            return abs((p.transpose() @ W) @ p)

            # find the coefficients a of the state ket in the magic base
            # phi+-, psi+-,  = triplet,singlet
            #bell = [1 i 0 0 0 0 i 1 0 0 i -1 1 -i 0 0]/np.sqrt(2)
            #a = bell'*p
            #C = abs(sum(a ** 2))

        # state operator
        temp = p @ W  # W.conj() == W so this works
        temp = temp @ temp.conj()  # == p * W * conj(p) * W
        if abs(self.purity() - 1) > TOLERANCE:
            L = np.sqrt(np.sort(np.linalg.eigvals(temp).real)[::-1]).real  # .real?
            return max(0, L[1] -L[2] -L[3] -L[4])

        return np.sqrt(np.trace(temp).real) # same formula as for state vecs, .real?


    def negativity(self, sys):
        """Negativity of the state.

        See :cite:`Peres`, :cite:`Horodecki1`

        Args:
          sys (Sequence[int]): subsystem indices
        Returns:
          float: Negativity of the state wrt. the given partitioning.
        """
        s = self.ptranspose(sys)  # partial transpose the state
        x = spl.svdvals(s.data)  # singular values
        return (np.sum(x) - 1) / 2


    def lognegativity(self, sys):
        r"""Logarithmic negativity of the state.

        Args:
          sys (Sequence[int]): subsystem indices
        Returns:
          float: Logarithmic negativity of the state wrt. the given partitioning, :math:`\log_2(2 N(\rho) +1)`.
        """
        return np.log2(2 * self.negativity(sys) + 1)


    def scott(self, m):
        """Scott's average bipartite entanglement measure.

        See :cite:`Love`, :cite:`Scott`, :cite:`MW`.

        Args:
          m (int): partition size
        Returns:
          array[float]: Terms of the Scott entanglement measure of the system for the given partition size.
          When m = 1 this is coincides with the Meyer-Wallach entanglement measure.
        """
        # Jacob D. Biamonte 2008
        # Ville Bergholm 2008-2014

        dim = self.dims()
        n = self.subsystems()

        if m < 1 or m > n-1:
            raise ValueError('Partition size must be between 1 and n-1.')

        D = min(dim) # FIXME correct for arbitrary combinations of qudits??
        N = sp.special.comb(n, m, exact=True)
        C = (D**m / (D**m - 1)) / N  # normalization

        Q = np.empty((N,))
        # Loop over all m-combinations of n subsystems, trace over everything except them.
        # reversed() fixes the order since we are actually looping over the complements.
        for k, sys in enumerate(reversed(list(itertools.combinations(range(n), n-m)))):
            temp = self.ptrace(sys)  # trace over everything except S_k
            # NOTE: For pure states, tr(\rho_S^2) == tr(\rho_{\bar{S}}^2)
            temp = 1 - np.trace(np.linalg.matrix_power(temp.data, 2))
            Q[k] = C * temp.real
        return Q


    def locc_convertible(self, t, sys):
        """LOCC convertibility of states.

        For bipartite pure states s and t, returns True iff self can be converted to t
        using local operations and classical communication (LOCC).
        See :cite:`NC`, chapter 12.5.1

        Args:
          t (state): another state
          sys (Sequence[int]): vector of subsystem indices defining the partition
        Returns:
          bool: True iff s can be LOCC-converted to t
        """
        if not equal_dims(self, t):
            raise ValueError('States must have equal dimensions.')

        try:
            s = self.to_ket()
            t = t.to_ket()
        except ValueError as exc:
            raise ValueError('Not implemented for nonpure states.') from exc

        s.ptrace(sys, inplace = True)
        t.ptrace(sys, inplace = True)
        return majorize(spl.eigvalsh(s.data), spl.eigvalsh(t.data))


    def plot(self, fig=None, symbols=3):
        """State tomography plot.

        Plots the probabilities of finding a system in this state
        in the different computational basis states upon measurement.
        Relative phases are represented by the colors of the bars.

        If the state is nonpure, also plots the coherences using a 3D bar plot.

        .. todo:: Matplotlib 2.0 handles 2d and 3d axes differently, so just passing an Axes instance won't work, hence Figure

        Args:
          fig (Figure): figure to plot the state into
          symbols (int): how many subsystem labels to show in the tickmarks
        Returns:
          Axes, Axes3D: the plot
        """
        from matplotlib import cm, colors

        dim = self.dims()
        n = self.subsystems()

        # prepare labels
        m = min(n, symbols)  # at most three symbols
        d = dim[:m]
        nd = np.prod(d)
        rest = '0' * (n-m) # the rest is all zeros
        ticklabels = []
        for k in range(nd):
            temp = lmap.array_to_numstr(np.unravel_index(k, d))
            ticklabels.append(temp + rest)

        ntot = np.prod(dim)
        skip = ntot / nd  # only every skip'th state gets a label to avoid clutter
        ticks = np.r_[0 : ntot : skip]
        N = self.data.shape[0]

        # color normalization
        nn = colors.Normalize(vmin=-1, vmax=1, clip=True)
        def phases(A):
            """Phase normalized to (-1,1]"""
            return np.angle(A) / np.pi

        if self.is_ket():
            s = self.fix_phase()
            c = phases(s.data.ravel())  # use phases as colors
            ax = fig.gca()

            width = 0.8
            bars = ax.bar(range(N), s.prob(), width)
            # color bars using phase data
            colormapper = cm.ScalarMappable(norm=nn, cmap=cm.get_cmap('hsv'))
            colormapper.set_array(c)
            for b in range(N):
                bars[b].set_edgecolor('k')
                bars[b].set_facecolor(colormapper.to_rgba(c[b]))

            # add a colorbar
            cb = fig.colorbar(colormapper, ax = ax, ticks = np.linspace(-1, 1, 5))
            cb.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

            # the way it should work (using np.broadcast, ScalarMappable)
            #bars = ax.bar(range(N), s.prob(), color=c, cmap=cm.get_cmap('hsv'), norm=whatever, align='center')
            #cb = fig.colorbar(bars, ax = ax, ticks = np.linspace(-1, 1, 5))

            ax.set_xlabel('Basis state')
            ax.set_ylabel('Probability')
            ax.set_xticks(ticks + width / 2) # shift by half the bar width
            ax.set_xticklabels(ticklabels)
        else:
            import mpl_toolkits.mplot3d

            c = phases(self.data)  # use phases as colors
            ax = fig.add_subplot(111, projection = '3d')
            ax.view_init(40, -115)

            width = 0.6  # bar width
            temp = np.arange(-width/2, N-1) # center the labels
            x, y = np.meshgrid(temp, temp[::-1])
            x = x.ravel()
            y = y.ravel()
            z = np.abs(self.data.ravel())
            pcol = ax.bar3d(x, y, 0, width, width, z, edgecolors='k', norm=nn, cmap=cm.get_cmap('hsv'))
            # now the colors
            pcol.set_array(np.kron(c.ravel(), (1,)*6))  # six faces per bar

            # the way it should work (using np.broadcast, ScalarMappable)
            #x, y = np.meshgrid(temp, temp)
            #pcol = ax.bar3d(x, y, 0, width, width, np.abs(self.data), color=c, cmap=cm.get_cmap('hsv'), norm=whatever, align='center')

            # add a colorbar
            cb = fig.colorbar(pcol, ax = ax, ticks = np.linspace(-1, 1, 5))
            cb.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

            ax.set_xlabel('Col state')
            ax.set_ylabel('Row state')
            ax.set_zlabel('$|\\rho|$')
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabels)
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticklabels[::-1])
            ax.set_alpha(0.8)

        fig.show()
        return ax


# other state representations

    def bloch_vector(self):
        r"""Generalized Bloch vector.

        Returns:
          array[float]: Generalized Bloch vector corresponding to the state.

        For an n-subsystem state the generalized Bloch vector is an order-n correlation
        tensor defined in terms of the standard Hermitian tensor basis B
        corresponding to state dimensions:

        .. math::

           A_{ijk\ldots} = \sqrt{D} \: \mathrm{Tr}(\rho  B_{ijk\ldots}),

        where ``D = prod(self.dims())``. A is always real since :math:`\rho` is Hermitian.
        For valid, normalized states

        * ``self.purity()`` :math:`\le 1 \implies` ``norm(A, 'fro')`` :math:`\le \sqrt{D}`
        * ``self.trace()``  :math:`= 1 \implies`   ``A[0, 0, ..., 0]`` :math:`= 1`

        E.g. for a single-qubit system ``norm(A, 'fro') <= sqrt(2)``.
        """
        dim = self.dims()
        G = tensorbasis(dim)
        a = []
        for g in G:
            a.append(self.ev(g))
        a = np.array(a) * np.sqrt(np.prod(dim)) # to match the usual Bloch vector normalization
        # into an array, one dimension per subsystem
        return a.reshape(np.array(dim) ** 2)


    def tensor(self, *arg):
        """Tensor product of states.

        Returns the tensor product state of states s1, s2, ...
        """
        arg = (self, *arg)
        # if all states are kets, keep the result state a ket
        pure = all(k.is_ket() for k in arg)
        if not pure:
            # otherwise convert all states to state ops before tensoring
            arg = [k.to_op() for k in arg]

        return State(lmap.tensor(*arg))


    @staticmethod
    def bloch_state(A: 'array[float]', dim: Optional[Sequence[int]] = None):
        r"""State corresponding to a generalized Bloch vector.

        Args:
          A: generalized Bloch vector
          dim: dimension vector. If ``None``, we use ``dim = np.sqrt(A.shape).astype(int)``.
        Returns:
          state: State corresponding to the given generalized Bloch vector.

        The inverse operation of :func:`bloch_vector`.
        A is defined in terms of the standard hermitian tensor basis B
        corresponding to the dimension vector dim.

        .. math::

           \rho = \sum_{ijk\ldots} A_{ijk\ldots} B_{ijk\ldots} / \sqrt{D},

        where ``D = prod(dim)``. For valid states ``norm(A, 'fro') <= sqrt(D)``.
        """
        if dim is None:
            dim = tuple(np.sqrt(A.shape).astype(int))  # s == dim ** 2

        G = tensorbasis(dim)
        d = np.prod(dim)
        rho = np.zeros((d, d), dtype=complex)
        for k, a in enumerate(A.flat):
            rho += a * G[k]

        C = 1/np.sqrt(d) # to match the usual Bloch vector normalization
        return State(C * rho, dim)


# named states
    @staticmethod
    def werner(p, d=2):
        r"""Werner states.

        Args:
          p (float): symmetric part weight, :math:`p \in [0,1]`
          d (int): dimension
        Returns:
          State: Werner state

        For every :math:`d \ge 2`, Werner states :cite:`Werner` are a linear family of
        bipartite :math:`d \times d`  dimensional quantum states that are
        invariant under all local unitary rotations of the form
        :math:`U \otimes U`, where :math:`U \in SU(d)`.

        Every Werner state is a linear combination of the identity and
        SWAP operators. Alternatively, they can be understood as convex
        combinations of the appropriately scaled symmetric and
        antisymmetric projectors
        :math:`P_\text{sym} = \frac{1}{2}(\I+\text{SWAP})` and
        :math:`P_\text{asym} = \frac{1}{2}(\I-\text{SWAP})`:

        .. math::

          \rho_\text{Werner} = p \frac{2 P_\text{sym}}{d(d+1)} +(1-p) \frac{2 P_\text{asym}}{d(d-1)}.

        p is the weight of the symmetric part of the state.
        The state is entangled iff p < 1/2, and pure only when d = 2
        and p = 0, at which point it becomes the 2-qubit singlet state.

        For every d, the Werner family of states includes the fully
        depolarized state :math:`\frac{\I}{d^2}`, obtained with :math:`p = \frac{d+1}{2d}`.

        The Werner states are partial-transpose dual to :func:`isotropic states<isotropic>`
        with the parameter :math:`p' = \frac{2p-1}{d}`.
        """
        dim = (d, d)
        S = gate.swap(d, d)
        I = gate.id(dim)
        #temp = 1 -2*p
        #alpha = (d*temp +1) / (temp +d)
        #rho = (I -alpha*S) / (d * (d -alpha))
        rho = p * (I+S)/(d*(d+1)) +(1-p) * (I-S)/(d*(d-1))
        return State(rho)


    @staticmethod
    def isotropic(p, d=2):
        r"""Isotropic states.

        Args:
          p (float): maximally entangled part weight, :math:`p \in [0,1]`
          d (int): dimension
        Returns:
          State: isotropic state

        For every :math:`d \ge 2`, isotropic states :cite:`Werner` are a linear family of
        bipartite :math:`d \times d` dimensional quantum states that are
        invariant under all local unitary rotations of the form
        :math:`U \otimes U^*`, where :math:`U \in SU(d)`.

        Every isotropic state is a linear combination of the identity operator
        and the projector to the maximally entangled state :math:`\ket{\cup} = \sum_{k=0}^{d-1} \ket{k, k}`:

        .. math::

          \rho_\text{Iso} = \frac{p}{d}\ket{\cup}\bra{\cup} +\frac{1-p}{d^2-1} (\I-\frac{1}{d}\ket{\cup}\bra{\cup}).

        p is the weight of the cup state projectorin the mixture.
        The state is entangled iff p > 1/d, and pure and fully entangled iff p = 1.

        For every d, the isotropic family of states includes the fully
        depolarized state :math:`\frac{\I}{d^2}`, obtained with :math:`p = \frac{1}{d^2}`.

        Isotropic states are partial-transpose dual to :func:`Werner states<werner>`.
        """
        cup = gate.copydot(0, 2, d)
        cup_proj = cup @ cup.ctranspose() / d
        I = gate.id([d, d])

        rho = p * cup_proj +(1-p) * (I -cup_proj) / (d**2 -1)
        return State(rho)


# wrappers

def fidelity(s, t):
    """Wrapper for :meth:`State.fidelity`."""
    return s.fidelity(t)


def trace_dist(s, t):
    """Wrapper for :meth:`State.trace_dist`."""
    return s.trace_dist(t)
