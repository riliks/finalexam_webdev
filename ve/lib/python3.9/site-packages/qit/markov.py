"""
Born-Markov noise
=================

This module simulates the effects of a heat bath coupled to a quantum
system, using the Born-Markov approximation in the weak coupling limit.
This results in a Lindblad-form master equation for the system.

The treatment in this module mostly follows :cite:`BP`.


:class:`MarkovianBath` methods
------------------------------

.. currentmodule:: qit.markov.MarkovianBath

.. autosummary::

   desc
   set_cutoff
   setup
   build_LUT
   compute_gs
   corr
   fit
   plot_spectral_correlation
   plot_spectral_correlation_vs_cutoff
   plot_bath_correlation

.. currentmodule:: qit.markov
"""
import collections.abc
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import norm
from scipy.integrate import quad
import scipy.special as sps
import scipy.constants as const
import matplotlib.pyplot as plt

from .base import sx, sz, TOLERANCE
from .utils import lmul, rmul, lrmul, spectral_decomposition


__all__ = ['MarkovianBath', 'ops', 'lindblad_ops', 'superop']


class MarkovianBath:
    r"""Markovian heat bath.

    Supports bosonic and fermionic canonical ensembles at
    absolute temperature T, with an ohmic spectral density.
    The bath couples to the system via a single-term coupling
    :math:`H_\text{int} = A \otimes \sum_k \lambda_k (a_k +a_k')`.

    Args:
        bath_type: bath type, determines the spectral density, in {'ohmic', 'photon'}.
        stat: bath statistics, in ('boson', 'fermion')
        TU: bath time unit (in s)
        T: bath temperature (in K)

    The bath spectral density is either ohmic or photonic, with a cutoff:

    .. math::
       J_\text{ohmic}(\omega) = \omega \: \mathrm{cut}(\omega) \Theta(\omega),\\
       J_\text{photon}(\omega) = \omega^3 \: \mathrm{cut}(\omega) \Theta(\omega).

    Three types of cutoffs are supported:

    .. math::
       \mathrm{cut}_\text{exp}(x \: \omega_c) &= \exp(-x),\\
       \mathrm{cut}_\text{smooth}(x \: \omega_c) &= \frac{1}{1+x^2},\\
       \mathrm{cut}_\text{sharp}(x \: \omega_c)  &= \Theta(1-x).

    The effects of the bath on the system are contained in the complex spectral correlation tensor :math:`\Gamma`.
    Computing values of this tensor is the main purpose of this class.
    It depends on three parameters:
    the inverse temperature of the bath :math:`s = \beta \hbar`,
    the spectral cutoff frequency of the bath :math:`\omega_c`,
    and the system frequency :math:`\omega`. It has the following scaling property:

    .. math::
       \Gamma_{s,\omega_c}(\omega) = \Gamma_{s/a,\omega_c a}(\omega a) / a.

    Hence we may eliminate the dimensions by choosing a = TU:

    .. math::
       \Gamma_{s,\omega_c}(\omega) = \frac{1}{\text{TU}} \Gamma_{\hat{s}, \hat{\omega_c}}(\hat{\omega}),

    where the hat denotes a dimensionless quantity.
    Since we only have one coupling term, :math:`\Gamma` is a scalar.
    We split it into its hermitian and antihermitian parts:

    .. math::
       \Gamma_{s, \omega_c}(\omega) = \frac{1}{2} \gamma(\omega) +i S(\omega),

    where :math:`\gamma` and :math:`S` are real, and

    .. math::
      \gamma(\omega) &= 2 \pi (1 \pm n(\omega)) \mathrm{cut}(|\omega|)
      \begin{cases}
      \omega   \quad \text{(bosons)}\\
      |\omega| \quad \text{(fermions)}
      \end{cases}\\
      S(\omega) &= P \int_0^\infty \mathrm{d}\nu \frac{J(\nu)}{(\omega-\nu)(\omega+\nu)}
      \begin{cases}
      \omega \coth(\beta \hbar \nu/2) +\nu \quad \text{(bosons)}\\
      \omega +\nu \tanh(\beta \hbar \nu/2) \quad \text{(fermions)}
      \end{cases}

    where :math:`n(\omega) := 1/(e^{\beta \hbar \omega} \mp 1)` is the Planc/Fermi function
    and :math:`\beta = 1/(k_B T)`.
    Since :math:`\Gamma` is pretty expensive to compute, we store the computed results
    into a lookup table which is used to interpolate nearby values.

    Private attributes (set automatically):

    ===========  ===========
    Data member  Description
    ===========  ===========
    cut_func     Spectral density cutoff function.
    pf           Planck or Fermi function depending on statistics.
    corr_int     Integral transform for the dimensionless bath correlation function.
    g_func       Spectral correlation tensor, real part. :math:`\gamma(\omega/\text{TU}) \: \text{TU} = \mathrm{g\_func}(\omega)`.
    s_func       Spectral correlation tensor, imaginary part.
    g0           :math:`\lim_{\omega \to 0} \gamma(\omega)`.
    s0           :math:`\lim_{\omega \to 0} S(\omega)`.
    omega        Lookup table.
    gs_table     Lookup table. :math:`\gamma(\text{omega[k]}/\text{TU}) \: \text{TU}` = gs_table[k, 0].
    ===========  ===========
    """
    # Ville Bergholm 2009-2016
    # pylint: disable=attribute-defined-outside-init,too-many-instance-attributes
    def __init__(self, bath_type: str, stat: str, TU: float, T: float):
        # Set up a descriptor for a heat bath coupled to a quantum system.
        self.type = bath_type
        """Bath type. 'ohmic' or 'photon'."""
        self.stat = stat
        """Bath statistics. Either 'boson' or 'fermion'."""
        self.TU   = TU
        r"""Time unit (in s). All Hamiltonians have been made dimensionless by multiplying with :math:`\text{TU}/\hbar`."""
        self.T    = T
        """Absolute temperature of the bath (in K)."""

        self.scale = 0
        r"""Dimensionless temperature scaling parameter :math:`\hbar / (k_B T \: \text{TU})`."""
        self.cut_type = ''
        """Spectral density cutoff type, in ``{'sharp', 'smooth', 'exp'}``."""
        self.cut_omega = 0
        r"""Spectral density cutoff angular frequency :math:`\omega_c` (in :math:`1/\text{TU}`)."""

        self.set_cutoff('exp', 1.0)


    def __repr__(self) -> str:
        """String representation."""
        return f'<Markovian heat bath. Spectral density: {self.type}, {self.stat}ic statistics, T = {self.T:g} K, TU = {self.TU:g} s>'


    def desc(self, long: bool=True) -> str:
        """Bath description string for plots."""
        temp = rf'{self.type}, {self.stat}, $\beta \hbar \omega_c$: {self.scale*self.cut_omega:.4g},'\
            + rf'cutoff: {self.cut_type}, {self.cut_omega:.4g}'
        if long:
            return r'Bath correlation tensor $\Gamma(\omega) = \frac{1}{2} \gamma(\omega) +i S(\omega)$: ' +temp
        return temp


    def set_cutoff(self, cutoff_type: Optional[str], cut_omega: Optional[float]) -> None:
        """Set the spectral density cutoff.

        Passing ``None`` leaves the corresponding property unchanged.

        Args:
            cutoff_type: cutoff type, in ``{'sharp', 'smooth', 'exp'}``
            cut_omega: omega cutoff value
        """
        # We assume that cut_func(0) == 1.
        if cutoff_type is not None:
            self.cut_type = cutoff_type
        if cut_omega is not None:
            self.cut_omega = cut_omega  # == omega_c*TU

        # update cutoff function
        if self.cut_type == 'sharp':
            self.cut_func = lambda x: x <= self.cut_omega  # Heaviside theta cutoff
        elif self.cut_type == 'smooth':
            self.cut_func = lambda x: 1 / (1 + (x / self.cut_omega)**2)  # rational Lorentz cutoff
        elif self.cut_type == 'exp':
            self.cut_func = lambda x: np.exp(-x / self.cut_omega)  # exponential cutoff
        else:
            raise ValueError(f"Unknown cutoff type '{self.cut_type}'.")
        self.setup()


    def setup(self) -> None:
        """Initializes the g and s functions, and the LUT.

        Must be called after parameters change.
        """
        # shorthand
        self.scale = const.hbar / (const.k * self.T * self.TU)

        self.g0 = 0
        # spectral density (without cutoff)
        if self.type == 'ohmic':
            self.J = lambda nu: nu
            if self.stat == 'boson':
                self.g0 = 2 * np.pi / self.scale
        elif self.type == 'photon':
            self.J = lambda nu: nu**3
        else:
            raise ValueError('Unknown bath type.')

        # s_func has simple poles at \nu = \pm x.
        if self.stat == 'boson':
            self.pf = lambda x: 1/(np.exp(self.scale * x) - 1)
            self.corr_int_real = lambda s, nu: self.J(nu) * self.cut_func(nu) * np.cos(nu * s) / np.tanh(self.scale * nu / 2)
            self.corr_int_imag = lambda s, nu: self.J(nu) * self.cut_func(nu) * -np.sin(nu * s)
            self.g_func = lambda x: 2 * np.pi * self.J(x) * self.cut_func(abs(x)) * (1 + self.pf(x))
            #self.s_func = lambda x,nu: self.J(nu) * self.cut_func(nu) * ((1 + self.pf(nu)) / (x - nu) + self.pf(nu) / (x + nu))
            self.s_func = lambda x,nu: self.J(nu) * self.cut_func(nu) * (x / np.tanh(self.scale * nu / 2) + nu) / (x**2 - nu**2)
        elif self.stat == 'fermion':
            self.pf = lambda x: 1/(np.exp(self.scale * x) + 1)
            self.corr_int_real = lambda s, nu: self.J(nu) * self.cut_func(nu) * np.cos(nu * s)
            self.corr_int_imag = lambda s, nu: self.J(nu) * self.cut_func(nu) * np.sin(nu * s) * -np.tanh(self.scale * nu / 2)
            self.g_func = lambda x: 2 * np.pi * self.J(abs(x)) * self.cut_func(abs(x)) * (1 - self.pf(x))
            #self.s_func = lambda x,nu: self.J(nu) * self.cut_func(nu) * ((1 - self.pf(nu)) / (x - nu) + self.pf(nu) / (x + nu))
            self.s_func = lambda x,nu: self.J(nu) * self.cut_func(nu) * (x + nu * np.tanh(self.scale * nu / 2)) / (x**2 - nu**2)
        else:
            raise ValueError('Unknown bath statistics.')
        self.s0, abserr = quad(lambda nu: self.s_func(0, nu), 0, np.inf)

        # clear lookup tables, since changing the cutoff requires recalc of S
        self.omega = np.empty((0,), dtype=float)
        self.gs_table = np.empty((0,2), dtype=float)
        self._pad_LUT()


    def _pad_LUT(self) -> None:
        """Add limits at infinity to the lookup tables."""
        self.omega    = np.r_[-np.inf, self.omega, np.inf]
        self.gs_table = np.r_[[[0, 0]], self.gs_table, [[0, 0]]]


    def build_LUT(self, om: Optional['np.ndarray[float]'] = None) -> None:
        """Build a lookup table for the spectral correlation tensor Gamma.

        Args:
            om: vector of omegas denoting the points to compute
        """
        # TODO justify limits for S lookup
        if om is None:
            # Default sampling for the lookup table.
            lim = self.cut_omega
            #lim = np.log10(10) / 5 / self.scale  # up to boltzmann factor == 10
            om = np.logspace(np.log10(1.1 * lim), np.log10(5 * lim), 20) # logarithmic sampling
            om = np.r_[np.linspace(0.05 * lim, 1 * lim, 20), om] # sampling is denser near zero, where S changes more rapidly
            om = np.r_[-om[::-1], 0, om]  # symmetric around zero

        self.omega = om
        self.gs_table = np.empty((len(om), 2))
        for k in range(len(om)):
            print(k)
            self.gs_table[k] = self.compute_gs(self.omega[k])
        self._pad_LUT()


    def _plot(self, x, om, q, gs, odd_s, even_s, f=plt.plot):
        """Plotting utility.

        Args:
            x (array[float]): x coordinates of the points
            om (array[float]):
            q (array[float]):
            gs (array[float]): gamma and S, the real and imaginary components of the
                spectral correlation tensor
            odd_s (array[float]): odd part of S
            even_s (array[float]): even part of S
            f (callable): plotting function

        Returns:
            Axes: the plot
        """
        _, ax = plt.subplots()
        ax.grid(True)
        ax.plot(x, gs, '-x')
        ax.plot(x, odd_s, 'k-')
        ax.plot(x, even_s, 'm-')

        # analytical expressions for even and odd s funcs
        if self.cut_type == 'sharp':
            odd_s = np.log(abs(q**2 / (q**2-1)))
            even_s = np.log(abs((1+q) / (1-q))) -2/q
        elif self.cut_type == 'smooth':
            odd_s = 2*np.log(abs(q)) / (1+q**2)
            even_s = -np.pi/q  / (1+q**2)
        elif self.cut_type == 'exp':
            odd_s = sps.expi(-q) * np.exp(q) + sps.expi(q) * np.exp(-q)
            even_s = -(sps.expi(-q) * np.exp(q) - sps.expi(q) * np.exp(-q)) - 2 / q
        else:
            raise ValueError('Unknown cut type.')

        f(x, om * odd_s, 'ko', x, om * even_s, 'mo')
        ax.set_ylabel('[1/TU]')
        ax.legend([
            r'$\gamma(\omega)$',
            r'$S(\omega)$',
            r'$S(\omega)-S(-\omega)$',
            r'$S(\omega)+S(-\omega)$',
            r'$S(\omega)-S(-\omega)$ (fermion)',
            r'$S(\omega)+S(-\omega)$ (boson)',
        ])
        return ax


    def plot_spectral_correlation(self) -> None:
        r"""Plot the spectral correlation tensor components :math:`\gamma` and :math:`S` as a function of omega.

        Additionally plots the even and odd parts of :math:`S`, and compares them to analytical
        expressions. Uses the spectral correlation tensor LUT.
        """
        if len(self.omega) <= 2:
            self.build_LUT()

        om = self.omega
        # computed values
        s = self.gs_table[:,1]
        temp = s[::-1]  # s(-x)
        odd_s  = s -temp  # s(x)-s(-x)
        even_s = s +temp  # s(x)+s(-x)

        # ratio of Lamb shift to dephasing rate
        #g = self.gs_table[:,0]
        #boltz = np.exp(-self.scale * om)
        #ratio = odd_s / (g * (boltz+1))

        c = self.cut_omega
        ax = self._plot(om, om, om/c, self.gs_table, odd_s, even_s)
        # cut limits
        a = plt.axis()[2:]  # y limits
        ax.plot([c,c], a, 'k-')
        ax.plot([-c,-c], a, 'k-')

        ax.set_xlabel(r'$\omega$ [1/TU]')
        ax.set_title(self.desc())
        ax.get_figure().show()


    def plot_spectral_correlation_vs_cutoff(self, boltz: float = 0.5) -> None:
        r"""Plot spectral correlation tensor components as a function of cutoff frequency.

        The angular frequency :math:`\omega` at which the spectral correlation tensor is evaluated
        is fixed by giving the Boltzmann factor :math:`e^{-\beta \hbar \omega}`.

        Args:
            boltz: Boltzmann factor
        """
        orig_scale  = self.scale
        orig_cutoff = self.cut_omega

        omega = -np.log(boltz) / self.scale  # \omega * TU
        # scale with |omega|
        temp = abs(omega)
        omega_scaled = omega / temp  # sign of omega
        self.scale *= temp     # scale the temperature parameter

        # try different cutoffs relative to omega
        cutoff = np.logspace(-1.5, 1.5, 50)
        gs  = np.zeros((len(cutoff), 2))
        gsm = np.zeros((len(cutoff), 2))
        for k, cut_k in enumerate(cutoff):
            print(k)
            self.set_cutoff(None, cut_k)  # scaled cutoff frequency
            gs[k,:]  = self.compute_gs(omega_scaled)
            gsm[k,:] = self.compute_gs(-omega_scaled)
        gs  *= temp
        gsm *= temp
        odd_s  = gs[:,1] -gsm[:,1]
        even_s = gs[:,1] +gsm[:,1]

        # restore original bath parameters
        self.scale = orig_scale
        self.set_cutoff(None, orig_cutoff)

        ax = self._plot(cutoff, omega, omega_scaled/cutoff, gs, odd_s, even_s, plt.semilogx)
        ax.set_xlabel(r'$\omega_c/\omega$')
        ax.set_title(self.desc() + f', boltzmann = {boltz:.4g} => omega = {omega:.4g}')
        ax.get_figure().show()


    def plot_bath_correlation(self) -> None:
        r"""Plot the bath correlation function
        :math:`C_{s,\omega_c}(t) = \frac{1}{\hbar^2}\langle B(t) B(0)\rangle`.

        It scales as

        .. math::
           C_{s,\omega_c}(t) = \frac{1}{a^2} C_{s/a,\omega_c a}(t/a).

        Choosing a = TU, we obtain

        .. math::
           C_{s,\omega_c}(t) = \frac{1}{\text{TU}^2} C_{\hat{s}, \hat{\omega_c}}(\hat{t}).
        """
        # pylint: disable=too-many-locals,too-many-statements

        tol_nu = 1e-5  # approaching the singularity at nu=0 this closely
        c = self.cut_omega
        fig = plt.figure()
        fig.suptitle('Bath correlation function C(t): ' + self.desc(False))

        # plot the functions to be transformed
        ax = plt.subplot(1, 3, 1)
        nu = np.linspace(tol_nu, 5*c, 500)
        ax.plot(nu, self.J(nu) * self.cut_func(nu) * self.pf(nu), 'r')
        if self.stat == 'boson':
            ax.plot(nu, self.J(nu) * self.cut_func(nu) * (1 + self.pf(nu)), 'b')
        else:
            ax.plot(nu, self.J(nu) * self.cut_func(nu) * (1 - self.pf(nu)), 'g')
        ax.grid(True)
        ax.legend([r'$J(\nu) n(\nu)$', r'$J(\nu) (1 \pm n(\nu))$'])
        ax.set_xlabel(r'$\nu$ [1/TU]')
        ax.set_title('Integrand without exponentials')

        # plot the full integrand
        ax = plt.subplot(1, 3, 2)
        t = np.linspace(0, 4/c, 5)
        nu = np.linspace(tol_nu, 5*c, 100)
        res = np.empty((len(nu), len(t)), dtype=complex)
        for k, t_k in enumerate(t):
            print(k)
            res[:,k] = self.corr_int_real(t_k, nu) +1j * self.corr_int_imag(t_k, nu)
        ax.plot(nu, res.real, '-', nu, res.imag, '--')
        ax.grid(True)
        ax.set_xlabel(r'$\nu$ [1/TU]')
        ax.set_title('Integrand for various values of $t$')

        # plot the correlation function C(t)
        ax = plt.subplot(1, 3, 3)
        t = np.linspace(0, 10/c, 100)  # real part of C(t) is even, imaginary part odd
        res = np.empty(len(t), dtype=complex)
        # upper limit for integration
        if self.cut_type == 'smooth':
            int_max = 100*c  # HACK, the integrator does not do well otherwise
        else:
            int_max = np.inf
        for k in range(len(t)):
            print(k)
            #fun = lambda x: x * self.cut_func(x) * (np.exp(-1j*x*t[k])*(1+self.pf(x))+np.exp(1j*x*t[k])*self.pf(x))
            fun1 = lambda x: self.corr_int_real(t[k], x)
            fun2 = lambda x: self.corr_int_imag(t[k], x)
            res[k], abserr = quad(fun1, tol_nu, int_max)
            temp, abserr = quad(fun2, tol_nu, int_max)
            res[k] += 1j*temp
        ax.plot(t, res.real, 'k-', label='Re $C(t)$')
        ax.plot(t, res.imag, 'k--', label='Im $C(t)$')
        ax.plot(t, abs(res), 'k-.', label='$|C(t)|$')
        ax.grid(True)
        ax.set_xlabel('t [TU]')
        ax.set_ylabel('[TU$^{-2}$]')
        ax.set_title('Bath correlation function $C(t)$')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # plot analytic high- and low-temp limits
        x = c * t
        if self.cut_type == 'sharp':
            temp = ((1 +1j * x) * np.exp(-1j * x) - 1) / x**2
            boson_hot_real = 2 / self.scale * np.sin(x) / x
        elif self.cut_type == 'smooth':
            shi, chi = sps.shichi(x)
            temp = np.sinh(x) * shi -np.cosh(x) * chi -1j * np.pi / 2 * np.exp(-abs(x))
            boson_hot_real = np.pi / self.scale * np.exp(-abs(x))
        elif self.cut_type == 'exp':
            temp = 1 / (1 +1j * x)**2
            boson_hot_real = 2 / self.scale / (1 + x**2)
        else:
            raise ValueError(f"Unknown cutoff type '{self.cut_type}'.")
        temp *= c**2
        boson_hot_real *= c
        if self.stat == 'boson':
            ax.plot(t, temp.real, 'b.', label='Re $C(t)$ (analytic) (cold)')  # real part, cold
            ax.plot(t, boson_hot_real, 'r.', label='Re $C(t)$ (analytic) (hot)')  # real part, hot
            ax.plot(t, temp.imag, 'ko', label='Im $C(t)$ (analytic)')  # imag part, every T
        else:
            ax.plot(t, temp.real, 'k.', label='Re $C(t)$ (analytic)')  # real part, every T
            ax.plot(t, temp.imag, 'bo', label='Im $C(t)$ (analytic) (cold)')  # imag part, cold
            ax.plot(t, 0*t, 'ro', label='Im $C(t)$ (analytic) (hot)')  # imag part, hot
        ax.legend()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.get_figure().show()
        return ax


    def compute_gs(self, x: float) -> Tuple[float, float]:
        r"""Computes the spectral correlation tensor.
        See :func:`corr` for usage.

        Args:
            x: angular frequency [1/TU]

        Returns:
            Real and imaginary parts of the spectral correlation tensor at ``x`` [1/TU]
        """
        ep = 1e-5 # epsilon for Cauchy principal value
        tol_omega0 = 1e-8

        if abs(x) <= tol_omega0:
            return self.g0, self.s0

        g = self.g_func(x)

        # Cauchy principal value, integrand has simple poles at \nu = \pm x.
        # TODO scipy quad can do these types of simple pole PVs directly...
        f = lambda nu: self.s_func(x, nu)
        a, abserr = quad(f, tol_omega0, abs(x) -ep)
        b, abserr = quad(f, abs(x) +ep, np.inf)
        return g, a + b


    def corr(self, x: float) -> Tuple[float, float]:
        r"""Bath spectral correlation tensor, computed or interpolated.

        .. math::

           \Gamma(x/\text{TU}) \: \text{TU} = \frac{1}{2} g +i s

        Args:
            x: angular frequency [1/TU]

        Returns:
            Real and imaginary parts of the spectral correlation tensor at ``x`` [1/TU]
        """
        # Ville Bergholm 2009-2011

        tol_omega = 1e-8
        max_ip_omega = 0.1 # maximum interpolation distance, TODO justify

        # assume parameters are set and lookup table computed
        #s = interp1(self.omega, self.gs_table, x, 'linear', 0)

        # TODO omega and gs_table into a single dictionary?
        # binary search for the interval [omega_a, omega_b) in which x falls
        b = np.searchsorted(self.omega, x, side = 'right')
        a = b - 1
        ee = self.omega[[a, b]]
        tt = self.gs_table[[a, b], :]
        # now x is in [ee[0], ee[1])

        gap = ee[1] - ee[0]
        d1 = abs(x - ee[0])
        d2 = abs(x - ee[1])

        def interpolate(ee, tt, x):
            "Quick interpolation."
            # interp1 does way too many checks
            return tt[0] + ((x - ee[0]) / (ee[1] - ee[0])) * (tt[1] - tt[0])

        # x close enough to either endpoint?
        if d1 <= tol_omega:
            return self.gs_table[a, :]
        if d2 <= tol_omega:
            return self.gs_table[b, :]
        if gap <= max_ip_omega + tol_omega:  # short enough gap to interpolate?
            return interpolate(ee, tt, x)

        # compute a new point p, then interpolate
        if gap <= 2 * max_ip_omega:
            p = ee[0] + gap / 2 # gap midpoint
            if x < p:
                idx = 1 # which ee p will replace
            else:
                idx = 0
        elif d1 <= max_ip_omega: # x within interpolation distance from one of the gap endpoints?
            p = ee[0] + max_ip_omega
            idx = 1
        elif d2 <= max_ip_omega:
            p = ee[1] - max_ip_omega
            idx = 0
        else: # x not near anything, don't interpolate
            p = x
            idx = 0

        # compute new g, s values at p and insert them into the table
        temp = self.compute_gs(p)

        self.omega = np.r_[self.omega[:b], p, self.omega[b:]]
        self.gs_table = np.r_[self.gs_table[:b], [temp], self.gs_table[b:]]

        # now interpolate the required value
        ee[idx] = p
        tt[idx, :] = temp
        return interpolate(ee, tt, x)


    def fit(self, delta: float, T1: float, T2: float):
        r"""Qubit-bath coupling that reproduces given decoherence times.

        Args:
            delta: qubit energy splitting (in units of :math:`\hbar/TU`)
            T1, T2: qubit decoherence times T1 and T2 (in units of :math:`TU`)

        Returns:
            array[complex], array[complex]: qubit Hamiltonian, qubit-bath coupling operator

        Returns the qubit Hamiltonian H and the qubit-bath coupling operator D
        that reproduce the decoherence times T1 and T2
        for a single-qubit system coupled to the bath.

        The bath object is not modified.
        """
        # Ville Bergholm 2009-2016

        if self.type == 'ohmic':
            # Fitting an ohmic bath to a given set of decoherence times

            iTd = 1/T2 -0.5/T1 # inverse pure dephasing time
            if iTd < 0:
                raise ValueError('Unphysical decoherence times!')

            # match bath couplings to T1, T2
            x = self.scale * delta / 2

            if self.stat == 'boson':
                temp = x / np.tanh(x) * self.cut_func(abs(delta))
                # coupling, ZX angle
                alpha = np.arctan2(1, np.sqrt(T1 * iTd *temp))
                # dimensionless system-bath coupling factor squared
                c = iTd * self.scale / (4 * np.pi * np.cos(alpha)**2)
                # noise coupling operator
                D = np.sqrt(c) * (np.cos(alpha) * sz + np.sin(alpha) * sx)

                # decoherence times in scaled time units
                #T1 = 1/(c * np.sin(alpha)**2 * 2 * np.pi * delta * coth(x) * self.cut_func(delta))
                #T_dephase = self.scale/(c *4*np.pi*np.cos(alpha)**2)
                #T2 = 1/(0.5/T1 +1/T_dephase)

            elif self.stat == 'fermion':
                if abs(iTd) >= TOLERANCE:
                    raise ValueError('For a fermionic bath we must have T2 = 2*T1')
                # dimensionless system-bath coupling factor squared
                c = 1 / (T1 * 2 * np.pi * abs(delta) * self.cut_func(abs(delta)))
                D = np.sqrt(c) * sx
            else:
                raise NotImplementedError('Unknown bath statistics.')
        else:
            raise NotImplementedError('Unknown bath type.')

        # qubit Hamiltonian
        H = -delta/2 * sz
        return H, D



def ops(H, D):
    r"""Jump operators for a Born-Markov master equation.

    Args:
        H (array[complex]): system Hamiltonian
        D (Sequence[array[complex]): Hermitian system-bath interaction operators

    Returns:
        array[float], array[array[complex]]: dH, A

    dH is a list of the sorted unique nonnegative differences between
    eigenvalues of H, and A is an object array of the corresponding jump operators:
    :math:`A_k(dH_i) = A[k][i]`, where the jump ops A[k] correspond to D[k].

    Since :math:`A_k(-dH) = A_k^\dagger(dH)`, only the nonnegative dH:s and corresponding A:s are returned.
    """
    # Ville Bergholm 2009-2016
    # pylint: disable=too-many-locals

    E, P = spectral_decomposition(H)
    m = len(E) # unique eigenvalues
    # energy difference matrix is antisymmetric, so we really only need the lower triangle
    deltaE = E[:, np.newaxis] - E  # deltaE[i,j] == E[i] - E[j]

    # mergesort is a stable sorting algorithm
    ind = np.argsort(deltaE, axis = None, kind = 'mergesort')
    # index of first lower triangle element
    s = m * (m - 1) // 2
    #assert(ind[s], 0)
    ind = ind[s:] # lower triangle indices only

    if not isinstance(D, collections.abc.Sequence):
        D = [D] # D needs to be a sequence, even if it has just one element
    n_D = len(D) # number of bath coupling ops

    # combine degenerate deltaE, build jump ops
    dH = []
    A = [[] for k in range(n_D)]
    current_dE = np.inf
    # loop over lower triangle indices
    for k in ind:
        dE = deltaE.flat[k]
        if abs(dE -current_dE) > TOLERANCE:
            # new omega value, new jump op
            current_dE = dE
            dH.append(dE)
            for op in range(n_D):
                A[op].append(0)
        # extend current jump op
        r, c = np.unravel_index(k, (m, m))
        for op in range(n_D):
            A[op][-1] += np.dot(np.dot(P[c], D[op]), P[r])

    A  = np.array(A)
    dH = np.array(dH)
    # find columns in which every A vanishes
    temp = np.zeros(A.shape[0:2])
    for k in range(len(dH)):
        for op in range(n_D):
            temp[op, k] = norm(A[op, k]) > TOLERANCE
    temp = temp.any(0)
    # eliminate zero As and corresponding dHs
    A = A[:,temp]
    dH = dH[temp]

    # Are some of the remaining dH differences too low for RWA to hold properly?
    # TODO justify the numerical tolerance used
    for k in range(1, len(dH)):
        if abs(dH[k] -dH[k-1]) < 1e-3:
            print(f'Warning: Small difference between dH({k-1}) and dH({k}) may break the RWA.\n')
    return dH, A


def _check_baths(B):
    """Raises an error if the baths do not all have the same time unit.

    Args:
        B (Sequence[MarkovianBath]): baths to check
    """
    if not isinstance(B, collections.abc.Sequence):
        B = [B] # needs to be a sequence, even if it has just one element

    # make sure the baths have the same TU!
    temp = B[0].TU
    for k in B:
        if k.TU != temp:
            raise ValueError('All the baths must have the same time unit!')
    return B


def lindblad_ops(H, D, B):
    r"""Lindblad operators for a Born-Markov master equation.

    Args:
        H (array[complex]): system Hamiltonian
        D (array[complex]): Hermitian interaction operator
        B (MarkovianBath): bath instance

    Returns:
        list[array[complex]], array[complex]: Lindblad operators, Lamb shift Hamiltonian

    Builds the Lindblad operators corresponding to a
    base Hamiltonian H and a (Hermitian) interaction operator D
    coupling the system to bath B.

    Returns a list L of Lindblad operators :math:`A_i` (in units of :math:`1/\sqrt{\text{TU}}`)
    and the Lamb shift Hamiltonian :math:`H_{\text{LS}}` (in units of :math:`\hbar/\text{TU}`).

    B can also be a sequence of baths, in which case D has to be
    a sequence of the corresponding interaction operators.
    """
    # Ville Bergholm 2009-2016
    B = _check_baths(B)

    # jump ops
    dH, X = ops(H, D)
    H_LS = 0
    L = []
    for n, b in enumerate(B):
        A = X[n] # jump ops for bath/interaction op n

        for k, dH_k in enumerate(dH):
            #NA[k] = norm(A[k], 'fro'); # how significant is this op?

            # first the positive energy shift
            g, s = b.corr(dH_k)
            # is the dissipation significant?
            if abs(g) >= TOLERANCE:
                L.append(np.sqrt(g) * A[k])
            H_LS += s * np.dot(A[k].conj().transpose(), A[k])

            if dH_k == 0:
                # no negative shift
                continue

            # now the corresponding negative energy shift
            g, s = b.corr(-dH_k)
            if abs(g) >= TOLERANCE:
                L.append(np.sqrt(g) * A[k].conj().transpose()) # note the difference here, A(-omega) = A'(omega)
            H_LS += s * np.dot(A[k], A[k].conj().transpose())  # here too
    return L, H_LS
    # TODO ops for different baths can be combined into a single basis,
    # N^2-1 ops max in total


def superop(H, D, B):
    r"""Liouvillian superoperator for a Born-Markov master equation.

    Args:
        H (array[complex]): system Hamiltonian
        D (array[complex], Sequence[array[complex]]): Hermitian interaction operator
        B (MarkovianBath, Sequence[MarkovianBath]): bath instance

    Returns:
        array[complex]: Liouvillian superoperator

    Builds the Liouvillian superoperator L corresponding to a
    base Hamiltonian H and a (Hermitian) interaction operator D
    coupling the system to the bath B.

    L includes the system Hamiltonian, the Lamb shift,
    and the Lindblad dissipator, and is in units of 1/TU.

    B can also be a sequence of baths, in which case D has to be
    a sequence of the corresponding interaction operators.
    """
    # Ville Bergholm 2009-2016
    B = _check_baths(B)

    # jump ops
    dH, X = ops(H, D)
    iH_LS = 1j * H  # i * (system Hamiltonian + Lamb-Stark shift)
    acomm = 0  # anticommutator
    diss = 0   # the rest of the dissipator
    for B_n, A in zip(B, X):
        # A contains the jump ops for bath/interaction op n

        # we build the Liouvillian in a funny order to be a bit more efficient
        for k, dH_k in enumerate(dH):
            # first the positive energy shift
            g, s = B_n.corr(dH_k)
            temp = np.dot(A[k].conj().transpose(), A[k])
            iH_LS += (1j * s) * temp
            acomm += (-0.5 * g) * temp
            diss  += lrmul(g * A[k], A[k].conj().transpose())

            if dH_k == 0:
                # no negative shift
                continue

            # now the corresponding negative energy shift
            g, s = B_n.corr(-dH_k)
            temp = np.dot(A[k], A[k].conj().transpose()) # note the difference here, A(-omega) = A'(omega)
            iH_LS += (1j * s) * temp
            acomm += (-0.5 * g) * temp
            diss  += lrmul(g * A[k].conj().transpose(), A[k]) # here too

    return lmul(acomm -iH_LS) +rmul(acomm +iH_LS) +diss
