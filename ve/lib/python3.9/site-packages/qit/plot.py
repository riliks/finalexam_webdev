"""
Plotting functions
==================

.. currentmodule:: qit.plot

This module contains functions for producing various plots.


Plots
-----

.. autosummary::
   adiabatic_evolution
   state_trajectory
   pcolor


Plotting utilities
------------------

.. autosummary::
   sphere
   bloch_sphere
   correlation_simplex
   asongoficeandfire

----
"""
# Ville Bergholm 2011-2020

import numpy as np

import mpl_toolkits.mplot3d as mplot3d
from mpl_toolkits.mplot3d import Axes3D  # required to make 3d plotting work for some reason!
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors

from qit.state import State
from qit.utils import copy_memoize, eighsort




def set_plotstyle(ax):
    """Given an axes handle, sets the plotting style -related properties for those axes.

    If everything works properly, all the child elements of the
    axes (titles, labels, legends, tick labels etc.) should inherit
    these settings.

    TODO FIXME
    """
    # TODO matplotlib.style.use()
    #set(ax, 'FontSize',18)  #, 'FontName','Bitstream Vera Sans')
    #set(get(ax, 'Parent'), 'DefaultLineLineWidth',2)  # apparently not an axes property(!)

    #set(ax, 'LineStyleOrder', '-|-.')
    #set(ax, 'Box','on')
    #set(ax, 'XMinorGrid','off', 'YMinorGrid','off')


def sphere(n=15):
    """Coordinate meshes for a unit sphere.

    Args:
      n (int): number of vertices in the theta direction (2*n in phi)
    Returns:
      array[float], array[float], array[float]: X, Y, Z coordinate meshes
    """
    theta = np.linspace(0, np.pi, n)
    phi = np.linspace(0, 2*np.pi, 2*n)
    X = np.outer(np.sin(theta), np.cos(phi))
    Y = np.outer(np.sin(theta), np.sin(phi))
    Z = np.outer(np.cos(theta), np.ones(phi.shape))
    return X, Y, Z


def adiabatic_evolution(t, st, H_func, n=4):
    """Adiabatic evolution plot.

    Plots the energies of the eigenstates of ``H_func(t[k])`` as a function of ``t[k]``,
    and the overlap of ``st[k]`` with the ``n`` lowest final Hamiltonian eigenstates.
    Useful for illustrating adiabatic evolution.

    Args:
      t  (Sequence[float]): time instances
      st (Sequence[state]): states corresponding to the times
      H_func (callable): time-dependant Hamiltonian function, float -> array
      n  (int): how many lowest eigenstates of the final Hamiltonian to include in the overlap plot

    Returns:
      Figure: the plots
    """
    # Jacob D. Biamonte 2008
    # Ville Bergholm 2009-2010
    # pylint: disable=too-many-locals

    T = t[-1]  # final time
    H = H_func(T)

    n = min(n, H.shape[0])
    m = len(t)

    # find the n lowest eigenstates of the final Hamiltonian
    #d, v = scipy.sparse.linalg.eigs(H, n, which = 'SR')
    #ind = d.argsort()  # increasing real part
    _, v = eighsort(H)
    lowest = []
    for j in range(n):
        #j = ind[j]
        lowest.append(State(v[:, -j-1]))
    # TODO with degenerate states these are more or less random linear combinations of the basis states... overlaps are not meaningful

    energies = np.zeros((m, H.shape[0]))
    overlaps = np.zeros((m, n))
    for k in range(m):
        tt = t[k]
        H = H_func(tt)
        energies[k, :] = np.sort(np.linalg.eigvalsh(H).real)
        for j in range(n):
            overlaps[k, j] = lowest[j].fidelity(st[k]) ** 2 # squared overlap with lowest final states

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(t/T, energies)
    ax.grid(True)
    ax.set_title('Energy spectrum')
    ax.set_xlabel('Adiabatic time')
    ax.set_ylabel('Energy')
    ax.set_xlim((0, 1))
    ax.set_ylim((np.min(energies), np.max(energies)))

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(t/T, overlaps) #, 'LineWidth', 1.7)
    ax.grid(True)
    ax.set_title('Squared overlap of current state and final eigenstates')
    ax.set_xlabel('Adiabatic time')
    ax.set_ylabel('Probability')
    temp = []
    for k in range(n):
        temp.append('$|{0}\\rangle$'.format(k))
    ax.legend(temp)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))  # (0, max(overlaps))
    return fig


def bloch_sphere(ax=None, equator=True):
    r"""Bloch sphere plot.

    Plots a Bloch sphere, a geometrical representation of the state space of a single qubit.
    Pure states are on the surface of the sphere, nonpure states inside it.
    The states :math:`\ket{0}` and :math:`\ket{1}` lie on the north and south poles of the sphere, respectively.

    Args:
      ax (Axes): axes in which to plot the sphere. If None, a new figure is created.
      equator (bool): if True, plot the equator (intersection with the XY plane) on the sphere
    Returns:
      Axes: axes containing the plot
    """
    # Ville Bergholm  2005-2012
    # James Whitfield 2010
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # surface
    X, Y, Z = sphere()
    ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, color = 'g', alpha = 0.2, linewidth = 0) #cmap = xxx
    ax.axis('tight')
    # poles
    coords = np.array([[0, 0, 1], [0, 0, -1]]).T  # easier to read this way
    ax.scatter(*coords, c = 'r', marker = 'o')
    ax.text(0, 0,  1.1, '$|0\\rangle$')
    ax.text(0, 0, -1.2, '$|1\\rangle$')
    # equator
    if equator:
        phi = np.linspace(0, 2*np.pi, 40)
        ax.plot(np.cos(phi), np.sin(phi), np.zeros(phi.shape), 'k-')
    # labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return ax


def correlation_simplex(ax=None, labels='diagonal'):
    """Plots the correlations simplexes for two-qubit states.

    Plots the geometrical representation of the set of allowed
    correlations in a two-qubit state. For each group of three
    correlation variables the set is a tetrahedron.
    The groups are called 'diagonal', 'pos', 'neg'.
    For diagonal correlations the vertices correspond to the four Bell states.

    .. note: The strange logic in the ordering of the 'pos' and 'neg' correlations follows the logic of the Bell state labeling convention.

    Args:
      ax    (Axes): axes in which to plot the simplex. If None, a new figure is created.
      labels (str): name of the correlation group whose vertex and axis labels are plotted, in {'diagonal', 'pos', 'neg'}
    Returns:
      Axes, list[int]: axes containing the plot, list of the three linear indices denoting the correlations corresponding to the three axes
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    ax.grid(True)
    ax.view_init(20, -105)

    # tetrahedron
    # vertices and faces
    v = np.array([[-1, -1, -1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]])
    f = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [3, 2, 1]]
    polys = [v[k, :] for k in f]
    polyc = mplot3d.art3d.Poly3DCollection(polys, alpha=0.2, facecolor='g', edgecolor='k')
    ax.add_collection(polyc)
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    # mark vertices
    ax.scatter([0], [0], [0], c = 'r', marker = '.')  # center
    ax.scatter(v[:, 0], v[:, 1], v[:, 2], c = 'r', marker = '.')  # vertices

    # label axes and vertices
    if labels == 'diagonal':
        ax.set_title('diagonal correlations')
        ax.set_xlabel('XX')
        ax.set_ylabel('YY')
        ax.set_zlabel('ZZ')
        ax.text(1.1, 1.1, -1.1, r'$|\Psi^+\rangle$')
        ax.text(1.1, -1.1, 1.1, r'$|\Phi^+\rangle$')
        ax.text(-1.1, 1.1, 1.1, r'$|\Phi^-\rangle$')
        ax.text(-1.2, -1.2, -1.2, r'$|\Psi^-\rangle$')
        ind = [5, 10, 15]

    elif labels == 'pos':
        ax.set_title('pos correlations')
        ax.set_xlabel('ZX')
        ax.set_ylabel('XY')
        ax.set_zlabel('YZ')
        ax.text(1.1, -1.1, 1.1, r'$|y+,0\rangle +|y-,1\rangle$')
        ax.text(-1.1, 1.1, 1.1, r'$|y+,0\rangle -|y-,1\rangle$')
        ax.text(1.1, 1.1, -1.1, r'$|y-,0\rangle +|y+,1\rangle$')
        ax.text(-1.2, -1.2, -1.2, r'$|y-,0\rangle -|y+,1\rangle$')
        ind = [7, 9, 14]

    elif labels == 'neg':
        ax.set_title('neg correlations')
        ax.set_xlabel('XZ')
        ax.set_ylabel('YX')
        ax.set_zlabel('ZY')
        ax.text(1.1, 1.1, -1.1, r'$|0,y-\rangle +|1,y+\rangle$')
        ax.text(-1.1, 1.1, 1.1, r'$|0,y+\rangle -|1,y-\rangle$')
        ax.text(1.1, -1.1, 1.1, r'$|0,y+\rangle +|1,y-\rangle$')
        ax.text(-1.2, -1.2, -1.2, r'$|0,y-\rangle -|1,y+\rangle$')
        ind = [13, 6, 11]

    elif labels == 'none':
        ind = []

    else:
        raise ValueError('Unknown set of correlations.')

    return ax, ind


def state_trajectory(traj, reset=True, ax=None, color='b', marker=''):
    """Plot a state trajectory in the correlation representation.

    * For a single-qubit system, plots the trajectory in the Bloch sphere.
    * For a two-qubit system, plots the reduced single-qubit states (in
      Bloch spheres), as well as the interqubit correlations.

    `traj` can be obtained e.g. by using one of the continuous-time
    state propagation functions and feeding the results to
    :meth:`state.bloch_vector`.

    Args:
      traj (list[array], array): generalized Bloch vector of the quantum state, or a list of them representing a trajectory
      reset (bool):  if False, adds another trajectory to the axes without erasing .
      ax (Axes): axes to plot in, or None in which case a new figure is created

    Returns:
      Axes: axes containing the plot

    Example 1: Trajectory of the state `s` under the Hamiltonian `H`::

      out = s.propagate(H, t, lambda s, H: s.bloch_vector())
      state_trajectory(out)

    Example 2: Just a single state `s`::

      state_trajectory(s.bloch_vector())
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    def plot_traj(ax, A, ind):
        """Plots the trajectory formed by the correlations given in ind."""
        # if we only have a single point, give it the end marker if none is specified
        if len(A) > 1:
            ax.scatter(A[0,  ind[0]],  A[0,  ind[1]],  A[0,  ind[2]], c=color, marker='x')
            ax.plot(A[:,  ind[0]],  A[:,  ind[1]],  A[:,  ind[2]], c=color, marker=marker)
            ax.scatter(A[-1, ind[0]],  A[-1, ind[1]],  A[-1, ind[2]], c=color, marker='o')
        else:
            m = 'o' if marker == '' else marker
            ax.scatter(A[:,  ind[0]],  A[:,  ind[1]],  A[:,  ind[2]], c=color, marker=m)

    if isinstance(traj, list):
        d = traj[0].size
    else:
        d = traj.size
    A = np.array(traj).reshape((-1, d), order='F')  # list index becomes the first dimension

    if A.shape[1] == 4:
        # single qubit
        if reset:
            ax.clear()
            bloch_sphere(ax)
        plot_traj(ax, A, [1, 2, 3])

    elif A.shape[1] == 16:
        # two qubits (or a single ququat...)

        # TODO split ax into subplots...
        #fig.delaxes(ax)
        if reset:
            gs = mpl.gridspec.GridSpec(2, 3)

            ax = fig.add_subplot(gs[0, 0], projection = '3d')
            bloch_sphere(ax)
            ax.set_title('qubit A')

            ax = fig.add_subplot(gs[0, 1], projection = '3d')
            bloch_sphere(ax)
            ax.set_title('qubit B')

            ax = fig.add_subplot(gs[1, 0], projection = '3d')
            correlation_simplex(ax, labels = 'diagonal')

            ax = fig.add_subplot(gs[1, 1], projection = '3d')
            correlation_simplex(ax, labels = 'pos')

            ax = fig.add_subplot(gs[1, 2], projection = '3d')
            correlation_simplex(ax, labels = 'neg')

        # update existing axes instances
        qqq = fig.get_axes()
        plot_traj(qqq[0], A, [1, 2, 3])
        plot_traj(qqq[1], A, [4, 8, 12])
        plot_traj(qqq[2], A, [5, 10, 15])
        plot_traj(qqq[3], A, [7, 9, 14])
        plot_traj(qqq[4], A, [13, 6, 11])

    else:
        raise ValueError('At the moment only plots one- and two-qubit trajectories.')
    return ax


def pcolor(ax, W, x, y, clim=(0, 1), cmap=None):
    """Easy pseudocolor plot.

    Plots the 2D function given in the matrix W.

    Args:
      ax (Axes): axes where to plot the stuff
      W (array[float]): discretized 2D function to be plotted
      x, y (array[float]): quad midpoint coordinate vectors that define the coordinate grid
      clim (tuple[float]): 2-tuple defining the color limits

    Returns:
      Axes: the plot
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    # x and y are quad midpoint coordinates but pcolor wants quad vertices, so
    def to_quad(q):
        "Quad midpoints to vertices."
        return (np.r_[q, q[-1]] + np.r_[q[0], q]) / 2

    if cmap is None:
        cmap = asongoficeandfire()
    #q = ax.pcolor(to_quad(x), to_quad(y), W, norm=mpl.colors.Normalize(clim[0], clim[1]), cmap=cmap)
    q = ax.pcolormesh(to_quad(x), to_quad(y), W, norm=mpl.colors.Normalize(clim[0], clim[1]), cmap=cmap)
    ax.axis('equal')
    ax.axis('tight')
    #ax.shading('interp')
    ax.get_figure().colorbar(q)
    return ax



@copy_memoize
def asongoficeandfire(n=127):
    """Colormap with blues and reds. Wraps.

    Args:
      n (int): number of colors in the map
    Returns:
      ~matplotlib.colors.Colormap: colormap object
    """
    # exponent
    d = 3.1
    p = np.linspace(-1, 1, n)
    # negative values: reds
    x = p[p < 0]
    c = np.c_[1 - ((1 + x) ** d), 0.5 * (np.tanh(4 * (-x -0.5)) + 1), (-x) ** d]
    # positive values: blues
    x = p[p >= 0]
    c = np.r_[c, np.c_[x ** d, 0.5 * (np.tanh(4 * (x -0.5)) + 1), 1 - ((1 - x) ** d)]]
    return colors.ListedColormap(c, name='asongoficeandfire')
    # TODO colors.LinearSegmentedColormap(name, segmentdata, N=256, gamma=1.0)
