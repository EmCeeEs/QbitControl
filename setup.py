from setuptools import setup

setup(
    name='qbitcontrol',
    version='0.1',
    description='Quantum Control of Few-Level Systems',
    url='http://github.com/EmCeeEs/QbitControl',
    author='Marcus Theisen',
    author_email='marcus.theisen@posteo.net',
    license='MIT',
    packages=['qbitcontrol'],
    install_requires=[
        'scipy',
        'sympy',
    ],
    zip_safe=False,
)
