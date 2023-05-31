import unittest

from pauliopt.pauli.pauli_gadget import PPhase
from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from pauliopt.pauli.utils import *
from pauliopt.utils import Angle, pi



def main():
    pp = PauliPolynomial(5)

    pp >>= PPhase(Angle(pi)) @ [I, I, X, Z, Y]
    pp >>= PPhase(Angle(pi)) @ [I, I, X, Z, Y]
    pp >>= PPhase(Angle(pi/2)) @ [I, I, X, Z, Y]

    print(pp.to_latex())


if __name__ == '__main__':
    main()