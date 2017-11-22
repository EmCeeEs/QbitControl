import numpy as np
import sympy as sp
import time
import itertools

from .qoperator import asOperator
_t = sp.Symbol('t', real=True)

class Timer:
    """Self-made Timer"""
    def __enter__(self):
        self.start = time.clock()
        return self
 
    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start
        #time.sleep(1)

def probability(states):
    """probability from expansion coefficients"""
    if len(states.shape) == 3:          #DensityMatrix
        return np.diagonal(states, axis1=1, axis2=2)
    else:                               #StateVector
        return np.power(np.abs(states), 2)

def fidelity(states):
    """numerical fidelity -- derivation from unity"""
    prop = probability(states)
    return np.sum(prop, axis=1)

def phases(coeffs, offset=1e-10):
    """complex phase of expansion coefficients"""
    return ((np.angle(coeffs) + offset) % (2*np.pi))
    #return ((np.angle(coeffs) + offset) % (2*np.pi)) - np.pi

def relative_phases(coeffs, offset=0):
    """pairwise phase difference between expansion coefficients"""
    phases = np.angle(coeffs)

    diffs = []
    pairs = list(itertools.combinations(range(phases.shape[1]), 2))
    for (j,k) in pairs:
        diffs.append(phases[::,k] - phases[::,j])

    #offset to avoid jumps in plots
    #return (np.stack(diffs, axis=1) + offset), pairs
    return (np.stack(diffs, axis=1) + offset) % (2*np.pi), pairs

def expectation_values(times, states, operator):
    """expectation values of operator at times wrt states"""

    def exp_value(state, operator, time):
        if len(state.shape) == 2:           #DensityMatrix
            return np.trace(np.dot(state, operator(time)))
        else:                               #StateVector
            return np.vdot(state, np.dot(operator(time), state))

    evs = np.ndarray(times.shape, dtype=complex)
    for i in range(times.shape[0]):
        evs[i] = exp_value(states[i], operator, times[i])
    return evs

def shannon_entropy(rho):
    w, v = np.linalg.eigh(rho)
    w = abs(w)  #ensure positivity
    #log of an operator is not defined by numpy
    #return -np.trace(rho*np.log(rho))
    return -np.sum(w*np.log(w))
    
def is_negligible(array):
    """Check whether an array is almost zero"""
    return np.allclose(np.abs(array), 0)

def to_ad_basis(times, coeffs, ham):
    """transform solution to adiabatic basis"""
    def get_coeffs_ad(time, coeffs, ham):
        ww, vv = np.linalg.eigh(ham(time))
        indices = np.argsort(ww)
        vv = vv[::, indices]
        phases = np.vstack(vv.shape[0]*[np.angle(vv[0,::])])
        vv = vv*np.exp(-1j*phases)
        return np.dot(vv.conj().T, coeffs)

    coeffs_ad = []
    for time, coeff in zip(times, coeffs):
        coeffs_ad.append(get_coeffs_ad(time, coeff, ham))
    return np.array(coeffs_ad)

