import numpy as np
import sympy as sp
from scipy.integrate import ode, complex_ode

from .qtools import Timer
from .qstate import asQuantumState, QuantumStateError
from .qoperator import asOperator, OperatorError, _t

class QuantumSystem:
    """n-level quantum system"""
    def __init__(self, ham):
        self._H = asOperator(ham)
        if isinstance(ham, sp.MatrixBase):
            self._Hdot = asOperator(ham.diff(_t))
        else:
            self._Hdot = None

        self._state = None
        self._mixed = None

    @property
    def N(self):
        """number of dimensions"""
        return self._H.N

    @property
    def H(self):
        """time-dependent Hamiltonian"""
        return self._H

    @property
    def H_dot(self):
        """temporal derivative of Hamiltonian"""
        return self._Hdot

    @H_dot.setter
    def H_dot(self, ham_dot):
        self._Hdot = asOperator(ham_dot)

    @property
    def state(self):
        """current state"""
        return self._state

    @state.setter
    def state(self, state):
        qstate = asQuantumState(state)
        if qstate.N != self.N:
            raise QuantumStateError
        self._state = qstate

    @property
    def is_ensemble(self):
        if self._state:
            return self._state.is_ensemble

    def eigenstates(self, time):
        """numerical eigenstates"""
        w, v = np.linalg.eigh(self.H(time))
        indices = np.argsort(w)
        return v[::, indices]

    def eigenenergies(self, time):
        """numeric eigenvalues"""
        w, v = np.linalg.eigh(self.H(time))
        indices = np.argsort(w)
        return w[indices]

    def propagate(self, t0, t1, N, integrator='zvode', integrator_params={}):
        """Propagate current state.
        --> ndarray times, ndarray states, float runtime

        Keyword arguments:
        integrator -- string in ['zvode', 'lsode', 'dop853']
        integrator_params -- dict
        """
        if self.state is None:
            raise QuantumStateError('Initial state not set')

        # setup
        I = self._setup_integrator(t0, integrator, **integrator_params)
        is_endtime = _wrap_term_crit(t0, t1)
        dt = (t1 - t0)/N        #negative for backward propagation

        # integrate
        print_stats(t0, self.state, 2, 'initial')
        states, times = [], []
        with Timer() as timer:
            while I.successful() and not is_endtime(I.t):
                states.append(I.y)
                times.append(I.t)
                I.integrate(I.t + dt)
                #print_stats(I.t, I.y, 1)

        # evaluate
        states, times = map(np.array, [states, times])
        if self.is_ensemble:
            states = states.reshape(states.shape[0], self.N, self.N)
        #states = list(map(asQuantumState, states))

        self.state = states[-1]
        print_stats(I.t, self.state, 2, 'final')
        print('runtime:\t{:.5f}'.format(timer.interval))
        if not is_endtime(I.t):
            raise RuntimeError('error time: {}'.format(I.t))

        return times, states, timer.interval

    def spectrum(self, t0=-10, t1=10, N=1e3):
        """compute adiabatic and diabatic spectrum"""
        times = np.linspace(t0, t1, int(N))
        diabates = np.ndarray((times.shape[0], self.N))
        adiabates = np.ndarray((times.shape[0], self.N))

        for i in range(times.shape[0]):
            diabates[i] = self.H(times[i]).diagonal().real
            adiabates[i] = self.eigenenergies(times[i]).real
        return times, diabates, adiabates

    def make_transitionless(self):
        """superadiabatic control -- Berry theory"""
        return QuantumSystem(self.H + self.get_CD())

    def get_CD(self):
        """compute counter diabatic Hamiltonian"""
        if (self.H_dot is None):
            raise OperatorError('H_dot not set')

        def H_control(t):
            H0 = self.H(t)
            H0_dot = self.H_dot(t)

            w, v = np.linalg.eigh(H0)   # v -- unitary transformation matrix
            v_inv = v.conj().T

            H1 = np.dot(v_inv, np.dot(H0_dot, v))
            n = H1.shape[0]
            for i in range(n):
                for j in range(n):
                    if (i == j):
                        H1[i, j] = 0
                    else:
                        H1[i, j] /= (w[j] - w[i])

            return 1j*np.dot(v, np.dot(H1, v_inv))
        return asOperator(H_control)

    def _setup_integrator(self, t0, name, **params):
        if (self.is_ensemble):
            func = vonNeumann
        else:
            func = schroedinger

        if (name == 'zvode'):
            I = ode(_wrap(func, self.H))
        else:
            I = complex_ode(_wrap(func, self.H))

        I.set_integrator(name, **params)
        I.set_initial_value(self.state.as_vector(), t0)
        return I


def print_stats(t, state, verbosity=1, prefix='current'):
    """monitor internal processes"""
    np.set_printoptions(precision=5)

    text = ''
    if (verbosity > 0):
        text += prefix + ' time:'
        text += '\t{:.5f}'.format(t)
    if (verbosity > 1):
        text += '\n'
        text += prefix + ' state:'
        text += '\t{}\n'.format(state)
        text += prefix + ' probability:'
        text += '\t{}\n'.format(state.probabilities)
        text += prefix + ' probability conservation -- numerical error:'
        text += '\t{}'.format(abs(1 - state.fidelity))

    if (text):
        print(text)

def _wrap_term_crit(t_begin, t_end):
    """termination criterion for integrator"""
    SAFETY_MARGIN = 1e-10
    def is_endtime_forward(t):
        return (t > t_end + SAFETY_MARGIN)
    def is_endtime_backward(t):
        return (t < t_end - SAFETY_MARGIN)

    if (t_begin < t_end):
        return is_endtime_forward
    else:
        return is_endtime_backward

# https://github.com/scipy/scipy/issues/4781
def _wrap(f, *f_params):
    """wrapper function -- bug workaround"""
    def f_wrapped(t, y):
        return f(t, y, *f_params)
    return f_wrapped

def schroedinger(t, psi, H):
    """Schroedinger equation"""
    return -1j*np.dot(H(t), psi)

def vonNeumann(t, rho, H):
    """(quantum Liouville-)von Neumann equation"""
    H = H(t)
    rho = rho.reshape(H.shape)
    rho_dot = -1j*(np.dot(H, rho) - np.dot(rho, H))
    return rho_dot.flatten()

