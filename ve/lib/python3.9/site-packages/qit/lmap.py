"""
Linear maps.
"""
# Ville Bergholm 2008-2020
from __future__ import annotations

from collections.abc import Sequence
import copy
from typing import Optional, Union

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

from qit.base import TOLERANCE


__all__ = ['numstr_to_array', 'array_to_numstr', 'Lmap', 'tensor']


# TODO sparse matrix support: np.kron -> sparse.kron, norm, probably other funcs as well
# TODO FIXME ndarray data ownership: views vs copies, vs the semantics of Lmap operations. .real(), .conj(), __mul__ etc


def numstr_to_array(s):
    """Utility, converts a numeric string to the corresponding array.

    Args:
        s (str): numeric string, e.g. '16023'

    Returns:
        array[int]: corresponding array, e.g. np.array([1, 6, 0, 2, 3])
    """
    return np.array([ord(x) -ord('0') for x in s])


def array_to_numstr(s):
    """Utility, converts an integer array to the corresponding numeric string.

    Args:
        s (Iterable[int]): numeric array, e.g. np.array([1, 6, 0, 2, 3])

    Returns:
        str: corresponding numeric string, e.g. '16023'
    """
    return "".join([chr(x +ord('0')) for x in s])


def array_to_label(s, symbols):
    """Utility, converts an integer array to the corresponding label.

    Args:
        s (Iterable[int]): numeric array
        symbols (Sequence[str]): symbol alphabet for the label

    Returns:
        str: label, ``label[k] == symbols[s[k]]``
    """
    return "".join([symbols[x] for x in s])


def format_ket(ind, dim, is_ket=True):
    """Ket or bra formatting.

    Args:
        ind (int): flat index
        dim (tuple[int]): subsystem dimensions
        is_ket (bool): iff True return a ket symbol, otherwise return a bra symbol

    Returns:
        str: ket or bra symbol containing the corresponding multi-index
    """
    temp = array_to_numstr(np.unravel_index(ind, dim))
    if is_ket:
        return ' |' + temp + '>'
    return ' <' + temp + '|'



class Lmap:
    """Bounded finite-dimensional linear maps between tensor products of Hilbert spaces.

    Contains both the order-2 tensor representing the map, and the dimension vectors of the
    domain and codomain vector spaces.
    All the usual scalar-map and map-map arithmetic operators are
    provided, including the exponentiation of maps by integers.

    Base class of :class:`.State`.
    """
# TODO: Another possible interpretation of Lmap would be to
# treat each subsystem as an index, with the subsystems within dim[0] and dim[1]
# corresponding to contravariant and covariant indices, respectively?

# TODO def __format__(self, format_spec)
# TODO linalg efficiency: copy vs. view

    def __init__(self, s: Union['array_like', Lmap], dim: Optional[Sequence[int]] = None):
        """
        Args:
          s: Linear map. If an Lmap instance is given, a copy is made.
          dim: 2-tuple containing the output and input subsystem dimensions
              stored in tuples:  ``dim == (out, in)``.
              If ``dim``, ``out`` or ``in`` is ``None``, the corresponding dimensions are inferred from ``s``.

        .. code-block:: python

          calling syntax                      resulting dim
          ==============                      =============
          Lmap(rand(a))                       ((a,), (1,))      1D array default: ket vector
          Lmap(rand(a), ((1,), None))         ((1,), (a,))      bra vector given as a 1D array
          Lmap(rand(a, b))                    ((a,), (b,))      2D array, all dims inferred
          Lmap(rand(4, b), ((2, 2), None))    ((2, 2), (b,))    2D array, output: two qubits
          Lmap(rand(a, 6), (None, (3, 2)))    ((a,), (3, 2))    2D array, input: qutrit+qubit
          Lmap(rand(6, 6), ((3, 2), (2, 3)))  ((3, 2), (2, 3))  2D array, all dims given

          Lmap(A)             (A is an Lmap) copy constructor
          Lmap(A, dim)        (A is an Lmap) copy constructor, redefine the dimensions
        """
        # initialize the ndarray part
        if isinstance(s, Lmap):
            # copy constructor
            #: array: linear map data
            self.data = copy.deepcopy(s.data)
            defdim = s.dim  # copy the dimensions too, unless redefined
        else:
            if sparse.isspmatrix(s):
                # TODO Lmap constructor, mul/add, tensor funcs must be able to handle both dense and sparse arrays.
                self.data = s
            else:
                # valid array initializer
                self.data = np.asarray(s, dtype=complex) # NOTE that if s is an ndarray it is not copied here

            # into a 2d array
            if self.data.ndim == 0:
                # scalar
                self.data.resize((1, 1))
            elif self.data.ndim == 1:
                # vector, ket by default
                self.data.resize((self.data.size, 1))
            elif self.data.ndim > 2:
                raise ValueError('Array dimension must be <= 2.')
            # now self.data.ndim == 2, always

            # is it a bra given as a 1D array?
            if dim and dim[0] == (1,):
                self.data.resize((1, self.data.size))

            # infer default dims from data (wrap them in tuples!)
            defdim = tuple((k,) for k in self.data.shape)

        # set the dimensions
        if dim is None:
            # infer both dimensions from s
            dim = (None, None)

        #: tuple[tuple, tuple]: 2-tuple of output and input dimension tuples, big-endian
        self.dim = []
        for k, d in enumerate(dim):
            if d is None or len(d) == 0:
                # not specified, use default
                self.dim.append(defdim[k])
            else:
                self.dim.append(tuple(d))
        self.dim = tuple(self.dim)

        # check dimensions
        if self.data.shape != tuple(map(np.prod, self.dim)):
            raise ValueError('Dimensions of the array do not match the combined dimensions of the subsystems.')


    def __repr__(self):
        """Display the Lmap in a neat format."""

        def format_scalar(x):
            "Print a complex scalar."
            if abs(x.imag) < TOLERANCE:
                # just the real part
                out = ' {0:+.4g}'.format(x.real)
            elif abs(x.real) < TOLERANCE:
                # just the imaginary part
                out = ' {0:+.4g}j'.format(x.imag)
            else:
                # both
                out = ' +({0:.4g}{1:+.4g}j)'.format(x.real, x.imag) #' +' + str(x)
            return out

        out = ''
        # is it a vector? (a map with a singleton domain or codomain dimension)
        sh = self.data.shape
        if 1 in sh:
            # vector
            # ket or bra?
            if sh[1] == 1:
                # scalar? just print the number
                if sh[0] == 1:
                    out += format_scalar(self.data.flat[0])
                    return out
                dim = self.dim[0]
                is_ket = True
            else:
                dim = self.dim[1]
                is_ket = False

            printed = 0
            if self.is_sparse():
                # with sparse arrays we loop over nonzero elements only
                temp = self.data.tocoo()
                if is_ket:
                    inds = temp.row
                else:
                    inds = temp.col
                for ind, x in zip(inds, temp.data):
                    printed += 1
                    out += format_scalar(x)
                    out += format_ket(ind, dim, is_ket)
                    # sanity check, do not display Lmaps with hundreds of terms
                    if printed >= 20:
                        out += ' ...'
                        break
            else:
                # loop over all vector elements
                for ind in range(np.prod(dim)):
                    x = self.data.flat[ind]
                    # make sure there is something to print
                    if abs(x) < TOLERANCE:
                        continue
                    printed += 1
                    out += format_scalar(x)
                    out += format_ket(ind, dim, is_ket)
                    # sanity check
                    if printed >= 20:  # or ind >= 128:
                        out += ' ...'
                        break
        else:
            # matrix
            out = repr(self.data)

        out += '\ndim: ' + str(self.dim[0]) + ' <- ' + str(self.dim[1])
        return out


# utilities

    def _inplacer(self, inplace):
        """Utility for implementing inplace operations.

        Args:
            inplace (bool): iff True perform the operation in place

        Returns:
            Lmap: iff inplace is True the object itself, otherwise a deep copy

        Functions using this should begin with ``s = self._inplacer(inplace)``
        and end with ``return s``
        """
        if inplace:
            return self
        return copy.deepcopy(self)


    def remove_singletons(self):
        """Eliminate unnecessary singleton dimensions.

        NOTE: changes the object itself!
        """
        dd = []
        for d in self.dim:
            temp = tuple(x for x in d if x > 1)
            if not temp:
                temp = (1,)
            dd.append(temp)
        self.dim = tuple(dd)


    def is_compatible(self, t):
        """True iff the Lmaps have equal dimensions and can thus be added."""
        if not isinstance(t, Lmap):
            raise TypeError('t is not an Lmap.')
        return self.dim == t.dim


    def is_ket(self):
        """True if the Lmap is a ket."""
        return self.data.shape[1] == 1


    def is_sparse(self):
        """True if the Lmap is internally represented as a sparse array."""
        return sparse.isspmatrix(self.data)


# linear algebra

    def real(self):
        """Real part."""
        s = copy.copy(self)
        s.data = self.data.real
        return s

    def imag(self):
        """Imaginary part."""
        s = copy.copy(self)
        s.data = self.data.imag
        return s

    def conj(self):
        """Complex conjugate."""
        s = copy.copy(self)  # preserves the type, important for subclasses
        s.data = np.conj(self.data) # copy
        return s

    def transpose(self):
        """Transpose."""
        s = copy.copy(self)
        s.dim = (s.dim[1], s.dim[0]) # swap dims
        s.data = self.data.transpose().copy()
        return s

    def ctranspose(self):
        """Hermitian conjugate."""
        s = copy.copy(self)
        s.dim = (s.dim[1], s.dim[0]) # swap dims
        s.data = np.conj(self.data).transpose() # view to a copy
        return s

    def __mul__(self, t):
        """Multiplication of Lmaps by scalars."""
        # must be able to handle sparse data
        if not np.isscalar(t):
            raise TypeError('The * operator is for scalar multiplication only.')

        # t is a scalar
        s = copy.copy(self)
        s.data = self.data * t
        return s

    def __rmul__(self, t):
        """Multiplication of Lmaps by scalars, reverse."""
        # scalars commute
        return self.__mul__(t)

    def __matmul__(self, t):
        """Concatenation (matrix multiplication) of Lmaps."""
        # must be able to handle sparse data
        if isinstance(t, Lmap):
            if self.dim[1] != t.dim[0]:
                raise ValueError('The input and output dimensions do not match.')

            s = copy.copy(self)
            s.dim = (self.dim[0], t.dim[1])
            #s.data = self.data.dot(t.data)
            s.data = self.data @ t.data
            return s

        raise TypeError('The @ operator is for Lmap concatenation only.')

    def __truediv__(self, t):
        """Division of Lmaps by scalars from the right."""
        if not np.isscalar(t):
            raise TypeError('The / operator is for scalar division only.')
        s = copy.copy(self)
        s.data = self.data / t
        return s

    def __add__(self, t):
        """Addition of Lmaps."""
        if not self.is_compatible(t):
            raise ValueError('The Lmaps are not compatible.')
        s = copy.copy(self)
        s.data = self.data + t.data
        return s

    def __sub__(self, t):
        """Subtraction of Lmaps."""
        if not self.is_compatible(t):
            raise ValueError('The Lmaps are not compatible.')
        s = copy.copy(self)
        s.data = self.data - t.data
        return s

    def __pow__(self, n):
        """Exponentiation of Lmaps by integer scalars."""
        if self.dim[0] != self.dim[1]:
            raise ValueError('The input and output dimensions do not match.')
        s = copy.copy(self)
        s.data = np.linalg.matrix_power(self.data, n)
        return s


    def __imul__(self, t):
        """In-place multiplication of Lmaps by scalars from the right."""
        if not np.isscalar(t):
            raise TypeError('The * operator is for scalar multiplication only.')

        self.data *= t
        return self


    def __itruediv__(self, t):
        """In-place division of Lmaps by scalars from the right."""
        if not np.isscalar(t):
            raise TypeError('The / operator is for scalar division only.')

        self.data /= t
        return self


    def __iadd__(self, t):
        """In-place addition of Lmaps."""
        if not self.is_compatible(t):
            raise ValueError('The Lmaps are not compatible.')
        self.data += t.data
        return self


    def __isub__(self, t):
        """In-place subtraction of Lmaps."""
        if not self.is_compatible(t):
            raise ValueError('The Lmaps are not compatible.')
        self.data -= t.data
        return self


    def trace(self):
        """Trace of the Lmap.

        The trace is only properly defined if self.dim[0] == self.dim[1].
        """
        if not np.array_equal(self.dim[0], self.dim[1]):
            raise ValueError('Trace not defined for non-endomorphisms.')

        return np.trace(self.data)


    def norm(self):
        """Frobenius matrix norm of the Lmap."""
        if self.is_sparse():
            return sparse.linalg.norm(self.data, 'fro')
        return np.linalg.norm(self.data, 'fro')


    def tensorpow(self, n):
        r"""Tensor power of the Lmap.

        Args:
            n (int): number of copies of the Lmap to tensor together

        Returns:
            Lmap: :math:`U^{\otimes n}`.
        """
        if n < 1:
            raise ValueError('Only positive integer tensor powers are allowed.')
        s = copy.deepcopy(self)
        # repeat the input and output dim vectors n times
        s.dim = (self.dim[0] * n, self.dim[1] * n)
        for _ in range(n-1):
            # kronecker product of the data
            s.data = np.kron(s.data, self.data)
        s.remove_singletons()
        return s


# subsystem ordering

    def reorder_legs(self, perm, inplace=False):
        """Arbitrary reordering of Lmap input/output subsystems (tensor legs).

        Args:
            perm (Iterable[Iterable[int]]): (t_1, t_2, ...) where t_i are sequences of nonnegative integers,
                one for each Lmap index (normally two). Concatenated together the t_i must form a full permutation.

        Returns:
            Lmap: copy of the Lmap with permuted leg order

        Differs from :meth:`Lmap.reorder` in that :meth:`reorder_legs` also allows permuting
        input legs into output legs and vice versa, i.e. there is just one big permutation for all the legs.
        """
        s = self._inplacer(inplace)

        # concatenate all leg dims into a vector
        dims = np.array([x for y in s.dim for x in y])
        # loop over partial permutations (one p per data array index)
        total_perm = []
        newdim = []
        for p in perm:
            p = list(p)
            total_perm.extend(p)
            # reorder the dimensions vectors
            temp = tuple(dims[p])
            newdim.append(temp)

        # check that total_perm is a permutation
        temp = np.arange(len(dims))
        if len(temp) != len(total_perm) or len(set(temp) ^ set(total_perm)) != 0:
            raise ValueError('Invalid permutation.')

        s.dim = tuple(newdim)
        s.remove_singletons()
        # tensor into another tensor which has one index per subsystem, permute dimensions,
        # back into a tensor with the original number of indices
        final_d = map(np.prod, newdim)
        s.data = s.data.reshape(dims).transpose(total_perm).reshape(final_d)
        return s


    def reorder(self, perm, inplace=False):
        """Change the relative order of the input and/or output subsystems.

        Args:
          perm (tuple[Sequence[int]]): 2-tuple of permutations of subsystem indices, (perm_out, perm_in)

        Returns:
          Lmap: Copy of the Lmap with permuted subsystem order.

        A permutation can be either None (identity/do nothing), a pair (a, b) of subsystems to be swapped,
        or a tuple containing a full permutation of the subsystems.
        Two subsystems to be swapped must be in decreasing order so as not
        to mistake the two-element identity permutation (0, 1) for a swap.

        .. code-block:: python

          reorder((None, (2, 1, 0)))   # ignore first index, reverse the order of subsystems in the second
          reorder(((5, 2), None))      # swap the subsystems 2 and 5 in the first index, ignore the second

        NOTE: The full permutations are interpreted in the same sense as
        numpy.transpose() understands them, i.e. the permutation
        tuple is the new ordering of the old subsystem indices.
        This is the inverse of the mathematically more common "one-line" notation.
        """
        s = self._inplacer(inplace)

        # sparse matrices cannot be reshaped to more than 2 dimensions
        if sparse.isspmatrix(s.data):
            s.data = s.data.toarray()

        orig_d = s.data.shape  # original dimensions
        total_d = []
        total_perm = []
        last_used_index = 0
        newdim = list(s.dim)

        # loop over indices
        for k, this_perm in enumerate(perm):
            # avoid a subtle problem with the input syntax, (0, 1) must not be understood as swap!
            if this_perm is not None and tuple(this_perm) == (0, 1):
                this_perm = None

            # requested permutation for this index
            if this_perm is None:
                # no change
                # let the dimensions vector be, lump all subsystems in this index into one
                this_dim = (orig_d[k],)
                this_perm = np.array([0])
                this_n = 1
            else:
                this_dim = np.array(s.dim[k])  # subsystem dims
                this_perm = np.array(this_perm) # requested permutation for this index
                this_n = len(this_dim)  # number of subsystems

                temp = np.arange(this_n) # identity permutation

                if len(this_perm) == 2:
                    # swap two subsystems
                    temp[this_perm] = this_perm[::-1]
                    this_perm = temp
                else:
                    # full permutation
                    if len(set(temp) ^ set(this_perm)) != 0:
                        raise ValueError('Invalid permutation.')

                # reorder the dimensions vector
                newdim[k] = tuple(this_dim[this_perm])

            # big-endian ordering
            total_d.extend(this_dim)
            total_perm.extend(last_used_index + this_perm)
            last_used_index += this_n

        # tensor into another tensor which has one index per subsystem, permute dimensions,
        # back into a tensor with the original number of indices
        s.dim = tuple(newdim)
        s.data = s.data.reshape(total_d).transpose(total_perm).reshape(orig_d)
        return s



def tensor(*arg):
    """Tensor product of Lmaps."""
    data = 1
    dout = []
    din = []
    kron = np.kron

    for k in arg:
        if k.is_sparse():
            # switch to sparse kron, which can also handle dense arrays
            kron = sparse.kron

        # concatenate dimensions
        dout += k.dim[0]
        din += k.dim[1]
        # kronecker product of the data
        data = kron(data, k.data)

    s = Lmap(data, (tuple(dout), tuple(din)))
    return s
