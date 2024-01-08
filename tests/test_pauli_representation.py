import unittest

from pauliopt.pauli.pauli_gadget import PPhase
from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from pauliopt.pauli.utils import *
from pauliopt.utils import Angle, pi
import os

_PAULI_REPR = "(π/2) @ { I, X, Y, Z }\n(π/4) @ { X, X, Y, X }"

SVG_CODE_PAULI = '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="610" height="210"><linearGradient id="ycolor" x1="0%" x2="100%" y1="100%" y2="0%"><stop offset="0%"  stop-color="#FF8888"/><stop offset="50%"  stop-color="#FF8888"/><stop offset="50%"  stop-color="#CCFFCC"/><stop offset="100%"  stop-color="#CCFFCC"/></linearGradient>\n<path fill="none" stroke="black" d="M 0, 130 L 10, 130"/>\n<path d="M 80 30 Q 72 84 30 120" fill="none" stroke="black"/>\n<rect fill="#FF8888" stroke="black" x="10" y="120" width="20" height="20"/>\n<path fill="none" stroke="black" d="M 0, 160 L 10, 160"/>\n<path d="M 80 30 Q 82 101 30 150" fill="none" stroke="black"/>\n<rect fill="#CCFFCC" stroke="black" x="10" y="150" width="20" height="20"/>\n<path fill="none" stroke="black" d="M 0, 190 L 10, 190"/>\n<path d="M 80 30 Q 92 117 30 180" fill="none" stroke="black"/>\n<rect fill="url(#ycolor)" stroke="black" x="10" y="180" width="20" height="20"/>\n<svg x="60" y="10" height="20" width="50"><rect x="0" y="0" width="100%" height="100%" stroke="black" fill="white" stroke-width="5 %"/><text x="50%" y="50%" width="100%" height="100%" font-size="100%" dominant-baseline="middle" text-anchor="middle" >π</text></svg>\n<path fill="none" stroke="black" d="M 0, 70 L 110, 70"/>\n<path d="M 180 30 Q 155 45 130 60" fill="none" stroke="black"/>\n<rect fill="#FF8888" stroke="black" x="110" y="60" width="20" height="20"/>\n<path fill="none" stroke="black" d="M 0, 100 L 110, 100"/>\n<path d="M 180 30 Q 162 66 130 90" fill="none" stroke="black"/>\n<rect fill="#FF8888" stroke="black" x="110" y="90" width="20" height="20"/>\n<path fill="none" stroke="black" d="M 30, 190 L 110, 190"/>\n<path d="M 180 30 Q 192 117 130 180" fill="none" stroke="black"/>\n<rect fill="url(#ycolor)" stroke="black" x="110" y="180" width="20" height="20"/>\n<svg x="160" y="10" height="20" width="50"><rect x="0" y="0" width="100%" height="100%" stroke="black" fill="white" stroke-width="5 %"/><text x="50%" y="50%" width="100%" height="100%" font-size="100%" dominant-baseline="middle" text-anchor="middle" >π/2</text></svg>\n<path fill="none" stroke="black" d="M 130, 70 L 210, 70"/>\n<path d="M 280 30 Q 255 45 230 60" fill="none" stroke="black"/>\n<rect fill="#FF8888" stroke="black" x="210" y="60" width="20" height="20"/>\n<path fill="none" stroke="black" d="M 30, 160 L 210, 160"/>\n<path d="M 280 30 Q 282 101 230 150" fill="none" stroke="black"/>\n<rect fill="#CCFFCC" stroke="black" x="210" y="150" width="20" height="20"/>\n<path fill="none" stroke="black" d="M 130, 190 L 210, 190"/>\n<path d="M 280 30 Q 292 117 230 180" fill="none" stroke="black"/>\n<rect fill="url(#ycolor)" stroke="black" x="210" y="180" width="20" height="20"/>\n<svg x="260" y="10" height="20" width="50"><rect x="0" y="0" width="100%" height="100%" stroke="black" fill="white" stroke-width="5 %"/><text x="50%" y="50%" width="100%" height="100%" font-size="100%" dominant-baseline="middle" text-anchor="middle" >π/256</text></svg>\n<path fill="none" stroke="black" d="M 230, 70 L 310, 70"/>\n<path d="M 380 30 Q 355 45 330 60" fill="none" stroke="black"/>\n<rect fill="#FF8888" stroke="black" x="310" y="60" width="20" height="20"/>\n<path fill="none" stroke="black" d="M 130, 100 L 310, 100"/>\n<path d="M 380 30 Q 362 66 330 90" fill="none" stroke="black"/>\n<rect fill="#FF8888" stroke="black" x="310" y="90" width="20" height="20"/>\n<path fill="none" stroke="black" d="M 30, 130 L 310, 130"/>\n<path d="M 380 30 Q 372 84 330 120" fill="none" stroke="black"/>\n<rect fill="#FF8888" stroke="black" x="310" y="120" width="20" height="20"/>\n<path fill="none" stroke="black" d="M 230, 160 L 310, 160"/>\n<path d="M 380 30 Q 382 101 330 150" fill="none" stroke="black"/>\n<rect fill="#CCFFCC" stroke="black" x="310" y="150" width="20" height="20"/>\n<path fill="none" stroke="black" d="M 230, 190 L 310, 190"/>\n<path d="M 380 30 Q 392 117 330 180" fill="none" stroke="black"/>\n<rect fill="url(#ycolor)" stroke="black" x="310" y="180" width="20" height="20"/>\n<svg x="360" y="10" height="20" width="50"><rect x="0" y="0" width="100%" height="100%" stroke="black" fill="white" stroke-width="5 %"/><text x="50%" y="50%" width="100%" height="100%" font-size="100%" dominant-baseline="middle" text-anchor="middle" >π/8</text></svg>\n<path fill="none" stroke="black" d="M 330, 70 L 410, 70"/>\n<path d="M 480 30 Q 455 45 430 60" fill="none" stroke="black"/>\n<rect fill="#FF8888" stroke="black" x="410" y="60" width="20" height="20"/>\n<path fill="none" stroke="black" d="M 330, 100 L 410, 100"/>\n<path d="M 480 30 Q 462 66 430 90" fill="none" stroke="black"/>\n<rect fill="#CCFFCC" stroke="black" x="410" y="90" width="20" height="20"/>\n<path fill="none" stroke="black" d="M 330, 190 L 410, 190"/>\n<path d="M 480 30 Q 492 117 430 180" fill="none" stroke="black"/>\n<rect fill="url(#ycolor)" stroke="black" x="410" y="180" width="20" height="20"/>\n<svg x="460" y="10" height="20" width="50"><rect x="0" y="0" width="100%" height="100%" stroke="black" fill="white" stroke-width="5 %"/><text x="50%" y="50%" width="100%" height="100%" font-size="100%" dominant-baseline="middle" text-anchor="middle" >π/4</text></svg>\n<path fill="none" stroke="black" d="M 430, 70 L 510, 70"/>\n<path d="M 580 30 Q 555 45 530 60" fill="none" stroke="black"/>\n<rect fill="#FF8888" stroke="black" x="510" y="60" width="20" height="20"/>\n<path fill="none" stroke="black" d="M 330, 160 L 510, 160"/>\n<path d="M 580 30 Q 582 101 530 150" fill="none" stroke="black"/>\n<rect fill="url(#ycolor)" stroke="black" x="510" y="150" width="20" height="20"/>\n<path fill="none" stroke="black" d="M 430, 190 L 510, 190"/>\n<path d="M 580 30 Q 592 117 530 180" fill="none" stroke="black"/>\n<rect fill="url(#ycolor)" stroke="black" x="510" y="180" width="20" height="20"/>\n<svg x="560" y="10" height="20" width="50"><rect x="0" y="0" width="100%" height="100%" stroke="black" fill="white" stroke-width="5 %"/><text x="50%" y="50%" width="100%" height="100%" font-size="100%" dominant-baseline="middle" text-anchor="middle" >π/2</text></svg>\n<path fill="none" stroke="black" d="M 530, 70 L 610, 70"/>\n<path fill="none" stroke="black" d="M 430, 100 L 610, 100"/>\n<path fill="none" stroke="black" d="M 330, 130 L 610, 130"/>\n<path fill="none" stroke="black" d="M 530, 160 L 610, 160"/>\n<path fill="none" stroke="black" d="M 530, 190 L 610, 190"/>\n</svg>'


class TestPauliConversion(unittest.TestCase):
    def test_circuit_construction(self):
        pp = PauliPolynomial(4)

        pp >>= PPhase(Angle(pi / 2)) @ [I, X, Y, Z]

        self.assertEqual(pp.num_qubits, 4)
        self.assertEqual(len(pp), 1)

        pp >>= PPhase(Angle(pi / 4)) @ [X, X, Y, X]

        self.assertEqual(pp.__repr__(), _PAULI_REPR)
        self.assertEqual(len(pp), 2)

        pp_ = PauliPolynomial(num_qubits=4)
        pp_ >> pp

        self.assertEqual(
            pp.__repr__(),
            pp_.__repr__(),
            "Right shift resulted in different pauli Polynomials.",
        )

    def test_circuit_visualisation_svg(self):
        pp = PauliPolynomial(5)

        pp >>= PPhase(Angle(pi)) @ [I, I, X, Z, Y]
        pp >>= PPhase(Angle(pi / 2)) @ [X, X, I, I, Y]
        pp >>= PPhase(Angle(pi / 256)) @ [X, I, I, Z, Y]
        pp >>= PPhase(Angle(pi / 8)) @ [X, X, X, Z, Y]
        pp >>= PPhase(Angle(pi / 4)) @ [X, Z, I, I, Y]
        pp >>= PPhase(Angle(pi / 2)) @ [X, I, I, Y, Y]

        self.assertEqual(pp.to_svg(svg_code_only=True), SVG_CODE_PAULI)
