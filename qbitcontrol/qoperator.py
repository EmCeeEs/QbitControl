import numpy as np
import sympy as sp
import itertools
import random

_t = sp.Symbol('t', real=True)

class OperatorError(Exception):
    pass

class Operator(object):
    """finite dimensional time-dependent operator"""
    def __init__(self, op):
        self._opfunc = as_valid_callable(op)
        self._dimension = self._opfunc(1).shape[0]

    @property
    def N(self):
        return self._dimension

    def __call__(self, t):
        # makes it slower but maybe is worth it
        if hasattr(t, '__iter__'):
            #TODO: use list comprehension -- functional programming
            retval = np.ndarray((len(t), self.N, self.N), dtype=complex)
            for i in range(len(t)):
                retval[i] = self._opfunc(t[i])
            #vfunc = np.vectorize(lambda t: self._opfunc(t))
            #print(vfunc(1))
            return retval
        else:
            return self._opfunc(t)

    #NOTE:  Problems with __radd__ and np.ndarrays:
    #       It wants to add elementwise.
    def __add__(self, other):
        other = asOperator(other)
        if self.N != other.N:
            raise OperatorError('cant add operators of different dimensions')
        def op_sum(t):
            return self(t) + other(t)
        return Operator(op_sum)

    def __mul__(self, other):
        other = asOperator(other)
        if self.N != other.N:
            raise OperatorError('cant mul operators of different dimensions')
        def op_mul(t):
            return np.dot(self(t), other(t))
        return Operator(op_mul)

    def __getitem__(self, indices):
        """operator entry"""
        i, j = indices
        def get_entry(t):
            if isinstance(t, (int, float)):
                return self(t)[i,j]
            else:
                return self(t)[::,i,j]
        return get_entry

def asOperator(qty):
    """factory pattern function"""
    if isinstance(qty, Operator):
        return qty
    else:
        return Operator(qty)

def as_valid_callable(op):
    """return verified callable operator"""
    if callable(op) and is_valid(op):
        return op

    if isinstance(op, sp.MatrixBase):
        opfunc = sp.lambdify(_t, sp.simplify(op), ['numpy'])
        return as_valid_callable(opfunc)

    if isinstance(op, (list, np.ndarray)):
        op = np.array(op).squeeze()
        return as_valid_callable(lambda t: op)

    raise TypeError('cannot convert {} to Operator'.format(type(op)))

def is_valid(op):
    """numerical validation"""
    valid_types = np.sctypes['complex']
    valid_types += np.sctypes['float']
    valid_types += np.sctypes['int']

    TMIN, TMAX, N = -100, 100, 100
    for _ in itertools.repeat(None, N):
        t = random.uniform(TMIN, TMAX)
        if not isinstance(op(t), np.ndarray):
            raise OperatorError('wrong return type')
        if not op(t).ndim == 2:
            raise OperatorError('wrong dimension')
        if not op(t).shape[0] == op(t).shape[1]:     #is square
            raise OperatorError('wrong shape')
        if not op(t).dtype.type in valid_types:
            raise OperatorError('wrong data type')
    return True

def is_hermitian(op):
    """numerical test for hermiticity"""
    TMIN, TMAX, N = -50, 50, 100
    for _ in itertools.repeat(None, N):
        t = random.uniform(TMIN, TMAX)
        if not np.allclose(op(t), op(t).conj().T):
            return False
    return True

def are_close(a, b, **kwargs):
    """check whether two operators are numerical close"""
    pass

# Pauli matrices
from sympy.physics.matrices import msigma
def gen_sigmas():
    """generate Pauli matrices.
    
    Usage:\n
    sp.Matrix(np.dot(vec, sigmas).reshape(2,2))
    """
    sigmas = []
    for i in range(4):
        if (i == 0):
            sigma = sp.Identity(2).as_explicit()
        else:
            sigma = sp.ImmutableMatrix(msigma(i))
        sigmas.append(sigma)
    return sigmas

sigmas = gen_sigmas()

##################
#### testsuit ####
##################

import unittest
class TestOperator(unittest.TestCase):
    def setUp(self):
        self.N_ITER = 100
        self.o1 = np.array([[1,0],[0,-1]])
        self.o2 = np.array([[0,1],[1,0]])
        self.o3 = np.array([[0,-1j],[1j,0]])

    def test_validity(self):
        # types
        with self.assertRaises(TypeError):
            as_valid_callable(2.)
        with self.assertRaises(OperatorError):
            a, b, c, d = ['a', 'b', 'c', 'd']
            as_valid_callable(np.array([[a,b],[c,d]]))
        with self.assertRaises(OperatorError):
            a, b, c, d = map(sp.S, ['a', 'b', 'c', 'd'])
            as_valid_callable(np.array([[a,b],[c,d]]))

        # shape
        with self.assertRaises(OperatorError):
            as_valid_callable(np.ndarray((1,2)))
        with self.assertRaises(OperatorError):
            as_valid_callable(sp.Matrix([1,2]))

        # dimension
        with self.assertRaises(OperatorError):      #squeeze
            as_valid_callable([0])
        with self.assertRaises(OperatorError):
            as_valid_callable(np.arange(8))
        with self.assertRaises(OperatorError):
            as_valid_callable(np.arange(8).reshape(2,2,2))

    def test_correctness(self):
        a1 = lambda t: np.array([[t, 2j],[-2j,-t]])
        a2 = as_valid_callable(a1)
        a3 = as_valid_callable(sp.Matrix([[_t, 2j],[-2j,-_t]]))
        for _ in itertools.repeat(None, self.N_ITER):
            t = random.uniform(-10, 10)
            self.assertTrue(np.allclose(a1(t), a2(t)))
            self.assertTrue(np.allclose(a2(t), a3(t)))

    def test_addition(self):
        op = Operator(self.o1 + self.o2)
        variants = []

        variants.append(Operator(self.o1) + Operator(self.o2))
        variants.append(Operator(self.o1) + self.o2)
        variants.append(Operator(self.o2) + Operator(self.o1))
        variants.append(Operator(self.o2) + self.o1)

        #variants.append(self.o1 + Operator(self.o2))
        #variants.append(self.o2 + Operator(self.o1))

        for _ in itertools.repeat(None, self.N_ITER):
            t = random.uniform(-10, 10)
            for other in variants:
                self.assertTrue(np.allclose(op(t), other(t)))

    def test_multiplication(self):
        op = Operator(np.dot(self.o1, self.o2))

        variantsTrue = []
        variantsTrue.append(Operator(self.o1) * Operator(self.o2))
        variantsTrue.append(Operator(self.o1) * self.o2)
        #variantsTrue.append(self.o1 * Operator(self.o2))

        variantsFalse = []
        variantsFalse.append(Operator(self.o2) * Operator(self.o1))
        variantsFalse.append(Operator(self.o2) * self.o1)
        #variantsFalse.append(self.o2 * Operator(self.o1))

        for _ in itertools.repeat(None, self.N_ITER):
            t = random.uniform(-10, 10)
            for other in variantsTrue:
                self.assertTrue(np.allclose(op(t), other(t)))
            for other in variantsFalse:
                self.assertFalse(np.allclose(op(t), other(t)))

if __name__ == '__main__':
    unittest.main()

