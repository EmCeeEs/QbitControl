import sympy as sp
import numpy as np

from .qsystem import QuantumSystem, _t

class LZ(QuantumSystem):
    """Landau-Zener-Majorana-Stueckelberg"""
    def __init__(self, alpha, delta):
        H = sp.ImmutableMatrix(
            [[1/2*alpha*_t          , 1/2*delta     ],
             [1/2*sp.conjugate(delta) , -1/2*alpha*_t ]])

        if not sp.S(delta).has(_t):
            self._lz_value = self.lz_formula(alpha, delta)
        else:
            self._lz_value = None
        QuantumSystem.__init__(self, H)

    @property
    def lz_value(self):
        """asymptotic survival probability"""
        return self._lz_value

    @staticmethod
    def lz_formula(alpha, delta):
        """Landau-Zener formula"""
        gamma = np.power(np.abs(delta), 2)/(2*np.abs(alpha))
        return np.exp(-2*np.pi*gamma)

class LZNoisy(QuantumSystem):
    """Landau-Zener-Majorana-Stueckelberg + Noise"""
    def __init__(self, alpha, delta, noise=1e-10):
        sys_LZ = LZ(alpha, delta)
        N = sys_LZ.N
        def H_noisy(t):
            return sys_LZ._H(t) + noise*(2*np.random.rand(N, N) - 1)
        def H_noisy(t):
            H_random = np.random.normal(0, noise, (N,N))
            return sys_LZ._H(t) + H_random
        QuantumSystem.__init__(self, H_noisy)

    @property
    def lz_value(self):
        """asymptotic survival probability"""
        return self._lz_value

    @staticmethod
    def lz_formula(alpha, delta):
        """Landau-Zener formula"""
        gamma = np.power(np.abs(delta), 2)/(2*np.abs(alpha))
        return np.exp(-2*np.pi*gamma)

class CH(QuantumSystem):
    """Generalized LZ -- Carroll-Hioe"""
    def __init__(self, alpha, delta):
        H = sp.ImmutableMatrix(
            [[alpha*_t          , delta             , 0         ],
             [delta.conjugate() , 0                 , delta     ],
             [0                 , delta.conjugate() , -alpha*_t ]])

        QuantumSystem.__init__(self, H)

class CH2(QuantumSystem):
    """Generalized LZ (asymmetric) -- Carroll-Hioe"""
    def __init__(self, alpha, delta, epsilon):
        H = sp.ImmutableMatrix(
            [[epsilon+alpha*_t  , delta             , 0                 ],
             [delta.conjugate() , 0                 , delta             ],
             [0                 , delta.conjugate() , epsilon-alpha*_t  ]])

        QuantumSystem.__init__(self, H)

class SequentialCrossings(QuantumSystem):
    """Sequential Crossings"""
    def __init__(self, alpha, delta, epsilon):
        H = sp.ImmutableMatrix(
            [[alpha*_t + epsilon, delta             , 0                 ],
             [delta.conjugate() , 0                 , delta             ],
             [0                 , delta.conjugate() , alpha*_t - epsilon]])

        QuantumSystem.__init__(self, H)

class TW(QuantumSystem):
    """Generalized LZ (asymmetric) -- Carroll-Hioe"""
    def __init__(self, alpha, delta, epsilon):
        H = sp.ImmutableMatrix(
            [[epsilon+alpha*_t  , delta             , 0                 ],
             [delta.conjugate() , -2*epsilon        , delta             ],
             [0                 , delta.conjugate() , epsilon-alpha*_t  ]])

        QuantumSystem.__init__(self, H)

class Square(QuantumSystem):
    """a new idea -- quadliteral, rhombus, parallelogramm"""
    def __init__(self, alpha, delta, epsilon):
        H = sp.zeros(4)
        H[0,0] = 2*epsilon + alpha*_t 
        H[1,1] = epsilon - alpha*_t 
        H[2,2] = epsilon + alpha*_t 
        H[3,3] = 2*epsilon - alpha*_t 
        H[0,1] = H[1,2] = H[2,3] = delta
        H[1,0] = H[2,1] = H[3,2] = delta.conjugate()
        #print(H)
        QuantumSystem.__init__(self, H)

class AsymQS(QuantumSystem):
    """two linear crossings with different slope and coupling strength"""
    def __init__(self, alpha1, alpha2, delta1, delta2, epsilon):
        H = sp.ImmutableMatrix(
            [[epsilon+alpha1*_t          , delta1             , 0         ],
             [sp.conjugate(delta1) , 0                 , delta2     ],
             [0                 , sp.conjugate(delta2) , epsilon-alpha2*_t ]])
        QuantumSystem.__init__(self, H)

class General(QuantumSystem):
    """general"""
    def __init__(self, n):
        H = sp.zeros(n)
        for i in range(n):
            for j in range(n):
                if i == j:
                    epsilon = 100*(np.random.random() - 0.5)
                    alpha = 20*(np.random.random() - 0.5)
                    if 2*(i) >= n:
                        epsilon = 5*(2*i % n) -30
                        alpha = -1*(n - i)
                        H[i,j] = epsilon + alpha*(_t)
                    if 2*(i) < n:
                        epsilon = 10*(i + 1) -30
                        alpha = (i + 1)
                        H[i,j] = epsilon + alpha*_t
                #if i == j-1:
                if (2*i >= n) and (2*j < n):
                    delta = 20*(np.random.random() - 0.5)
                    delta = 0.5
                    H[i,j] = H[j,i] = delta
        sp.pprint(H)
        QuantumSystem.__init__(self, H)

