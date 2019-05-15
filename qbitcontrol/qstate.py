import numpy as np

from .qoperator import asOperator

class QuantumStateError(Exception):
    pass

class QuantumState:
    """base class"""
    _state = None
    _is_ensemble = None
    _is_pure = None

    @property
    def N(self):
        return len(self._state)

    @property
    def is_ensemble(self):
        return self._is_ensemble

    @property
    def is_pure(self):
        return self._is_pure

    def as_vector(self):
        return self._state.flatten()

    def __repr__(self):
        return str(self._state)

    @property
    def probabilities(self):
        raise NotImplementedError

    @property
    def fidelity(self):
        raise NotImplementedError

class StateVector(QuantumState):
    """vector dynamics -- quantum state as normalized vector"""
    def __init__(self, vector):
        self._state = vector
        self._is_pure = True
        self._is_ensemble = False 

    def asDensityMatrix(self):
        """ensemble description"""
        return DensityMatrix(np.outer(self.coeffs, self.coeffs.conj()))

    @property
    def coeffs(self):
        return self._state

    @property
    def probabilities(self):
        return np.power(np.abs(self.coeffs), 2)

    @property
    def fidelity(self):
        return np.sum(self.probabilities)
        return np.linalg.norm(self._state)

    @property
    def is_normalized(self):
        return np.isclose(self.fidelity, 1)

    def ev(self, operator):
        op = asOperator(operator)
        if not op.N == self.N:
            raise QuantumStateError
        return np.vdot(state, np.dot(operator, state))

class DensityMatrix(QuantumState):
    """matrix dynamics -- quantum state as statistical ensemble"""
    def __init__(self, matrix):
        if not np.isclose(matrix.trace(), 1):
            raise QuantumStateError('No statistical probability conservation')

        if not np.linalg.norm(matrix) <= 1:
            if not is_idempotent(matrix):
                raise QuantumStateError('Neither pure nor mixed state')

        self._state = matrix
        self._is_pure = np.isclose(self.fidelity, 1)
        self._is_ensemble = True

    @property
    def entropy(self):
        """shannon entropy"""
        w, v = np.linalg.eigh(self._state)
        return -np.sum(w*np.log(w))

    @property
    def probabilities(self):
        return np.diagonal(self._state)

    @property
    def fidelity(self):
        return sum(self.probabilities)

def asQuantumState(qty):
    """factory pattern function"""
    if isinstance(qty, QuantumState):
        return qty

    state = np.array(qty, dtype=complex).squeeze()
    if state.ndim == 1:
        return StateVector(state)
    elif state.ndim == 2 and is_square(state):
        return DensityMatrix(state)
    else:
        raise QuantumStateError('Wrong shape/dimension')
    
def is_square(mat):
    return mat.shape[0] == mat.shape[1]

def is_idempotent(mat):
    return np.allclose(np.dot(mat, mat), mat)

def normalize(vec):
    return vec/np.linalg.norm(vec)


##################
#### testsuit ####
##################

import unittest
class TestQuantumState(unittest.TestCase):
    def test_constructor(self):
        with self.assertRaises(QuantumStateError):
            asQuantumState([])
        with self.assertRaises(QuantumStateError):
            asQuantumState([1])
        with self.assertRaises(QuantumStateError):
            asQuantumState(np.empty((2,3)))
        with self.assertRaises(QuantumStateError):
            asQuantumState(np.empty((2,2,2)))

        self.assertIsInstance(asQuantumState([1,0]), StateVector)
        self.assertIsInstance(asQuantumState([[0.5,0],[0,0.5]]), DensityMatrix)

    def test_state_vector(self):
        with self.assertRaises(QuantumStateError):
            asQuantumState([0.5, -0.5])
        
    def test_density_matrix(self):
        with self.assertRaises(QuantumStateError):
            asQuantumState([[2,0],[0,0]])
        with self.assertRaises(QuantumStateError):
            asQuantumState([[0.5,1],[-1,0.5]])

    def test_pure_states(self):
        qs = asQuantumState(normalize([0.5,0.5,0.5]))
        self.assertIsInstance(qs, StateVector)

        qs = asQuantumState([[1,0,0],[0,0,0],[0,0,0]])
        self.assertIsInstance(qs, DensityMatrix)
        self.assertTrue(qs.is_pure)

        a, b, c = 0.2, 0.8, np.sqrt(0.2*0.8)
        qs = asQuantumState([[a,c],[c,b]])
        self.assertIsInstance(qs, DensityMatrix)
        self.assertTrue(qs.is_pure)

if __name__ == '__main__':
    unittest.main()

