QbitControl
===========
A simple framework to simulate and control few-level quantum systems.

Description
-----------
This package allows the propagation of pure and mixed quantum states according to Schrödinger and von-Neumann equations, respectively. The embedding system is represented by a time-dependent Hamiltonian and the state by a complex vector or density matrix. The aim is to study (super-)adiabatic control protocols that may allow for efficient state manipulation at the quantum speed limit. It comes with a set of templates implementing common few-level quantum systems:

- Landau-Zener (LZ)
- Carroll-Hioe (CH)
- Sequential Crossings

Installation
------------
.. code:: sh

    pip install git+https://github.com/EmCeeEs/QbitControl.git@master

Example
-------
Let us examine the `LZ problem <https://en.wikipedia.org/wiki/Landau%E2%80%93Zener_formula>`_. It regards a time-dependent system, where two states change their characteristics in the course of time. In the case of non-zero interaction the states do not cross (avoided crossing) resulting in a non-zero transition probability. The LZ formula gives this transition probability in the asymptotic limit. We can simulate this behaviour as follows:

.. code-block:: python

    from qbitcontrol.templates import LZ
    
    # model parameters
    alpha = 1       # sweep velocity
    delta = 0.5     # interaction strength
    
    # numeric parameters
    tstart = -100
    tend = 100
    Nsteps = 1000
    
    # init model
    qsys = LZ(alpha, delta)
    qsys.state = qsys.eigenstates(tstart)[0]   # start in eigenstate
    
    # init simulation
    times, states, _ = qsys.propagate(tstart, tend, Nsteps)
    
    # final state is superposition of eigenstates
    print(states[-1])
    
    # make transitionless (super-adiabatic)
    qsys2 = qsys.make_transitionless()
    qsys2.state = qsys.eigenstates(tstart)[0]   # start in original eigenstate
        
    # init simulation
    times, states, _ = qsys2.propagate(tstart, tend, Nsteps)
    
    # final state is eigenstate
    print(states[-1])

API
---
The main component is the *QuantumSystem* object. It's important attributes are:

- N
    - dimensionality of the problem
- H
    - the system's Hamiltonian
- state
    - the current state of the system

- eigenstates(time)
    - compute eigenstates at specified time
- eigenenergies(time)
    - compute eigenenergies at specified time
- makes_transitionless()
    - compute super-adiabatic version of system
- propagate(tstart, tend, Nsteps)
    - propagate system in time; from tstart to tend with Nsteps
    - return times, states and errors
- spectrum()
    - compute energy spectrum
