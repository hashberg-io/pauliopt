import unittest

from pauliopt.pauli.pauli_gadget import PPhase
from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from pauliopt.pauli.utils import *
from pauliopt.utils import Angle, pi
import os

_PAULI_REPR = "(π/2) @ { I, X, Y, Z }\n(π/4) @ { X, X, Y, X }"

SVG_CODE_PAULI = '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="610" height="200"><linearGradient id="ycolor" x1="0%" x2="100%" y1="100%" y2="0%"><stop offset="0%"  stop-color="#CCFFCC"/><stop offset="50%"  stop-color="#CCFFCC"/><stop offset="50%"  stop-color="#FF8888"/><stop offset="100%"  stop-color="#FF8888"/></linearGradient>\n<path fill="none" stroke="black" d="M 0, 130 L 10, 130"/>\n<path d="M 60 30 Q 63 81 30 120" fill="none" stroke="black"/>\n<rect fill="#CCFFCC" stroke="black" x="10" y="120" width="20" height="20"/>\n<path fill="none" stroke="black" d="M 0, 160 L 10, 160"/>\n<path d="M 60 30 Q 74 97 30 150" fill="none" stroke="black"/>\n<rect fill="#FF8888" stroke="black" x="10" y="150" width="20" height="20"/>\n<path fill="none" stroke="black" d="M 0, 190 L 10, 190"/>\n<path d="M 60 30 Q 84 112 30 180" fill="none" stroke="black"/>\n<rect fill="url(#ycolor)" stroke="black" x="10" y="180" width="20" height="20"/>\n<svg x="60" y="10" height="20" width="50"><rect x="0" y="0" width="100%" height="100%" stroke="black" fill="white" stroke-width="5 %"/><text x="50%" y="50%" width="100%" height="100%" font-size="100%" dominant-baseline="middle" text-anchor="middle" >π</text></svg>\n<path fill="none" stroke="black" d="M 0, 70 L 110, 70"/>\n<path d="M 160 30 Q 145 45 130 60" fill="none" stroke="black"/>\n<rect fill="#CCFFCC" stroke="black" x="110" y="60" width="20" height="20"/>\n<path fill="none" stroke="black" d="M 0, 100 L 110, 100"/>\n<path d="M 160 30 Q 153 64 130 90" fill="none" stroke="black"/>\n<rect fill="#CCFFCC" stroke="black" x="110" y="90" width="20" height="20"/>\n<path fill="none" stroke="black" d="M 30, 190 L 110, 190"/>\n<path d="M 160 30 Q 184 112 130 180" fill="none" stroke="black"/>\n<rect fill="url(#ycolor)" stroke="black" x="110" y="180" width="20" height="20"/>\n<svg x="160" y="10" height="20" width="50"><rect x="0" y="0" width="100%" height="100%" stroke="black" fill="white" stroke-width="5 %"/><text x="50%" y="50%" width="100%" height="100%" font-size="100%" dominant-baseline="middle" text-anchor="middle" >π/2</text></svg>\n<path fill="none" stroke="black" d="M 130, 70 L 210, 70"/>\n<path d="M 260 30 Q 245 45 230 60" fill="none" stroke="black"/>\n<rect fill="#CCFFCC" stroke="black" x="210" y="60" width="20" height="20"/>\n<path fill="none" stroke="black" d="M 30, 160 L 210, 160"/>\n<path d="M 260 30 Q 274 97 230 150" fill="none" stroke="black"/>\n<rect fill="#FF8888" stroke="black" x="210" y="150" width="20" height="20"/>\n<path fill="none" stroke="black" d="M 130, 190 L 210, 190"/>\n<path d="M 260 30 Q 284 112 230 180" fill="none" stroke="black"/>\n<rect fill="url(#ycolor)" stroke="black" x="210" y="180" width="20" height="20"/>\n<svg x="260" y="10" height="20" width="50"><rect x="0" y="0" width="100%" height="100%" stroke="black" fill="white" stroke-width="5 %"/><text x="50%" y="50%" width="100%" height="100%" font-size="100%" dominant-baseline="middle" text-anchor="middle" >π/256</text></svg>\n<path fill="none" stroke="black" d="M 230, 70 L 310, 70"/>\n<path d="M 360 30 Q 345 45 330 60" fill="none" stroke="black"/>\n<rect fill="#CCFFCC" stroke="black" x="310" y="60" width="20" height="20"/>\n<path fill="none" stroke="black" d="M 130, 100 L 310, 100"/>\n<path d="M 360 30 Q 353 64 330 90" fill="none" stroke="black"/>\n<rect fill="#CCFFCC" stroke="black" x="310" y="90" width="20" height="20"/>\n<path fill="none" stroke="black" d="M 30, 130 L 310, 130"/>\n<path d="M 360 30 Q 363 81 330 120" fill="none" stroke="black"/>\n<rect fill="#CCFFCC" stroke="black" x="310" y="120" width="20" height="20"/>\n<path fill="none" stroke="black" d="M 230, 160 L 310, 160"/>\n<path d="M 360 30 Q 374 97 330 150" fill="none" stroke="black"/>\n<rect fill="#FF8888" stroke="black" x="310" y="150" width="20" height="20"/>\n<path fill="none" stroke="black" d="M 230, 190 L 310, 190"/>\n<path d="M 360 30 Q 384 112 330 180" fill="none" stroke="black"/>\n<rect fill="url(#ycolor)" stroke="black" x="310" y="180" width="20" height="20"/>\n<svg x="360" y="10" height="20" width="50"><rect x="0" y="0" width="100%" height="100%" stroke="black" fill="white" stroke-width="5 %"/><text x="50%" y="50%" width="100%" height="100%" font-size="100%" dominant-baseline="middle" text-anchor="middle" >π/8</text></svg>\n<path fill="none" stroke="black" d="M 330, 70 L 410, 70"/>\n<path d="M 460 30 Q 445 45 430 60" fill="none" stroke="black"/>\n<rect fill="#CCFFCC" stroke="black" x="410" y="60" width="20" height="20"/>\n<path fill="none" stroke="black" d="M 330, 100 L 410, 100"/>\n<path d="M 460 30 Q 453 64 430 90" fill="none" stroke="black"/>\n<rect fill="#FF8888" stroke="black" x="410" y="90" width="20" height="20"/>\n<path fill="none" stroke="black" d="M 330, 190 L 410, 190"/>\n<path d="M 460 30 Q 484 112 430 180" fill="none" stroke="black"/>\n<rect fill="url(#ycolor)" stroke="black" x="410" y="180" width="20" height="20"/>\n<svg x="460" y="10" height="20" width="50"><rect x="0" y="0" width="100%" height="100%" stroke="black" fill="white" stroke-width="5 %"/><text x="50%" y="50%" width="100%" height="100%" font-size="100%" dominant-baseline="middle" text-anchor="middle" >π/4</text></svg>\n<path fill="none" stroke="black" d="M 430, 70 L 510, 70"/>\n<path d="M 560 30 Q 545 45 530 60" fill="none" stroke="black"/>\n<rect fill="#CCFFCC" stroke="black" x="510" y="60" width="20" height="20"/>\n<path fill="none" stroke="black" d="M 330, 160 L 510, 160"/>\n<path d="M 560 30 Q 574 97 530 150" fill="none" stroke="black"/>\n<rect fill="url(#ycolor)" stroke="black" x="510" y="150" width="20" height="20"/>\n<path fill="none" stroke="black" d="M 430, 190 L 510, 190"/>\n<path d="M 560 30 Q 584 112 530 180" fill="none" stroke="black"/>\n<rect fill="url(#ycolor)" stroke="black" x="510" y="180" width="20" height="20"/>\n<svg x="560" y="10" height="20" width="50"><rect x="0" y="0" width="100%" height="100%" stroke="black" fill="white" stroke-width="5 %"/><text x="50%" y="50%" width="100%" height="100%" font-size="100%" dominant-baseline="middle" text-anchor="middle" >π/2</text></svg>\n<path fill="none" stroke="black" d="M 530, 70 L 610, 70"/>\n<path fill="none" stroke="black" d="M 430, 100 L 610, 100"/>\n<path fill="none" stroke="black" d="M 330, 130 L 610, 130"/>\n<path fill="none" stroke="black" d="M 530, 160 L 610, 160"/>\n<path fill="none" stroke="black" d="M 530, 190 L 610, 190"/>\n</svg>'

LATEX_CODE_PAULI = """\documentclass[preview]{standalone}

\\usepackage{tikz}
\\usetikzlibrary{zx-calculus}
\\usetikzlibrary{quantikz}
\\usepackage{graphicx}

\\tikzset{
diagonal fill/.style 2 args={fill=#2, path picture={
\\fill[#1, sharp corners] (path picture bounding box.south west) -|
                         (path picture bounding box.north east) -- cycle;}},
reversed diagonal fill/.style 2 args={fill=#2, path picture={
\\fill[#1, sharp corners] (path picture bounding box.north west) |- 
                         (path picture bounding box.south east) -- cycle;}}
}

\\tikzset{
diagonal fill/.style 2 args={fill=#2, path picture={
\\fill[#1, sharp corners] (path picture bounding box.south west) -|
                         (path picture bounding box.north east) -- cycle;}}
}

\\tikzset{
pauliY/.style={
zxAllNodes,
zxSpiders,
inner sep=0mm,
minimum size=2mm,
shape=rectangle,
%fill=colorZxX
diagonal fill={colorZxX}{colorZxZ}
}
}

\\tikzset{
pauliX/.style={
zxAllNodes,
zxSpiders,
inner sep=0mm,
minimum size=2mm,
shape=rectangle,
fill=colorZxX
}
}

\\tikzset{
pauliZ/.style={
zxAllNodes,
zxSpiders,
inner sep=0mm,
minimum size=2mm,
shape=rectangle,
fill=colorZxZ
}
}

\\tikzset{
pauliPhase/.style={
zxAllNodes,
zxSpiders,
inner sep=0.5mm,
minimum size=2mm,
shape=rectangle,
fill=white
}
}
\\begin{document}
\\begin{ZX}
\\zxNone{} 		& \\zxNone{}                                 & |[pauliPhase]| \\pi           & \\zxNone{}      & \\zxNone{}                                 & |[pauliPhase]| \\pi           & \\zxNone{}      & \\zxNone{}                                 & |[pauliPhase]| \\frac{\\pi}{2} & \\zxNone{}      &\\\\ 
\\\\ 
\\zxNone{} \\rar 	& \\zxNone{} \\rar                            & \\zxNone{} \\rar               & \\zxNone{} \\rar & \\zxNone{} \\rar                            & \\zxNone{} \\rar               & \\zxNone{} \\rar & \\zxNone{} \\rar                            & \\zxNone{} \\rar               & \\zxNone{} \\rar &\\\\ 
\\zxNone{} \\rar 	& \\zxNone{} \\rar                            & \\zxNone{} \\rar               & \\zxNone{} \\rar & \\zxNone{} \\rar                            & \\zxNone{} \\rar               & \\zxNone{} \\rar & \\zxNone{} \\rar                            & \\zxNone{} \\rar               & \\zxNone{} \\rar &\\\\ 
\\zxNone{} \\rar 	& |[pauliX]| \\ar[ruuuu, bend right] \\rar    & \\zxNone{} \\rar               & \\zxNone{} \\rar & |[pauliX]| \\ar[ruuuu, bend right] \\rar    & \\zxNone{} \\rar               & \\zxNone{} \\rar & |[pauliX]| \\ar[ruuuu, bend right] \\rar    & \\zxNone{} \\rar               & \\zxNone{} \\rar &\\\\ 
\\zxNone{} \\rar 	& |[pauliZ]| \\ar[ruuuuu, bend right] \\rar   & \\zxNone{} \\rar               & \\zxNone{} \\rar & |[pauliZ]| \\ar[ruuuuu, bend right] \\rar   & \\zxNone{} \\rar               & \\zxNone{} \\rar & |[pauliZ]| \\ar[ruuuuu, bend right] \\rar   & \\zxNone{} \\rar               & \\zxNone{} \\rar &\\\\ 
\\zxNone{} \\rar 	& |[pauliY]| \\ar[ruuuuuu, bend right] \\rar  & \\zxNone{} \\rar               & \\zxNone{} \\rar & |[pauliY]| \\ar[ruuuuuu, bend right] \\rar  & \\zxNone{} \\rar               & \\zxNone{} \\rar & |[pauliY]| \\ar[ruuuuuu, bend right] \\rar  & \\zxNone{} \\rar               & \\zxNone{} \\rar &\\\\ 
\\end{ZX} 
\\end{document}
"""


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

        self.assertEqual(pp.__repr__(), pp_.__repr__(),
                         "Right shift resulted in different pauli Polynomials.")

    def test_circuit_visualisation_svg(self):
        pp = PauliPolynomial(5)

        pp >>= PPhase(Angle(pi)) @ [I, I, X, Z, Y]
        pp >>= PPhase(Angle(pi / 2)) @ [X, X, I, I, Y]
        pp >>= PPhase(Angle(pi / 256)) @ [X, I, I, Z, Y]
        pp >>= PPhase(Angle(pi / 8)) @ [X, X, X, Z, Y]
        pp >>= PPhase(Angle(pi / 4)) @ [X, Z, I, I, Y]
        pp >>= PPhase(Angle(pi / 2)) @ [X, I, I, Y, Y]

        self.assertEqual(pp.to_svg(svg_code_only=True), SVG_CODE_PAULI)

    def test_circuit_visualization_latex(self):
        pp = PauliPolynomial(5)

        pp >>= PPhase(Angle(pi)) @ [I, I, X, Z, Y]
        pp >>= PPhase(Angle(pi)) @ [I, I, X, Z, Y]
        pp >>= PPhase(Angle(pi / 2)) @ [I, I, X, Z, Y]

        self.assertEqual(pp.to_latex(), LATEX_CODE_PAULI)

        pp.to_latex(file_name="test")
        self.assertTrue(os.path.isfile("./test.tex"))
        with open("./test.tex", "r") as f:
            content = f.read()
            self.assertEqual(LATEX_CODE_PAULI, content)
        os.remove("./test.tex")