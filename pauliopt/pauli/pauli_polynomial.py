from pauliopt.pauli.clifford_gates import CliffordGate
from pauliopt.pauli.pauli_gadget import PauliGadget

from pauliopt.topologies import Topology
import math
from pauliopt.pauli.utils import X, Y, Z, I
from pauliopt.utils import SVGBuilder


class PauliPolynomial:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.pauli_gadgets = []

    def __irshift__(self, gadget: PauliGadget):
        if not len(gadget) == self.num_qubits:
            raise Exception(f"Pauli Polynomial has {self.num_qubits}, but Pauli gadget has: {len(gadget)}")
        self.pauli_gadgets.append(gadget)
        return self

    def __rshift__(self, pauli_polynomial):
        for gadget in pauli_polynomial.pauli_gadgets:
            self.pauli_gadgets.append(gadget)
        return self

    def __repr__(self):
        return '\n'.join(map(repr, self.pauli_gadgets))

    def __len__(self):
        return len(self.pauli_gadgets)

    @property
    def num_gadgets(self):
        return len(self.pauli_gadgets)

    def to_qiskit(self, topology=None):
        num_qubits = self.num_qubits
        if topology is None:
            topology = Topology.complete(num_qubits)
        try:
            from qiskit import QuantumCircuit
        except:
            raise Exception("Please install qiskit to export Clifford Regions")

        qc = QuantumCircuit(num_qubits)
        for gadget in self.pauli_gadgets:
            qc.compose(gadget.to_qiskit(topology), inplace=True)

        return qc

    def propagate(self, gate: CliffordGate):
        pp_ = PauliPolynomial(self.num_qubits)
        for gadget in self.pauli_gadgets:
            pp_ >>= gate.propagate_pauli(gadget)
        return pp_

    def copy(self):
        pp_ = PauliPolynomial(self.num_qubits)
        for gadget in self.pauli_gadgets:
            pp_ >>= gadget.copy()
        return pp_

    def two_qubit_count(self, topology, leg_cache=None):
        if leg_cache is None:
            leg_cache = {}
        count = 0
        for gadget in self.pauli_gadgets:
            count += gadget.two_qubit_count(topology, leg_cache=leg_cache)
        return count

    def to_svg(self, hscale: float = 1.0, vscale: float = 1.0, scale: float = 1.0,
               svg_code_only=False):
        vscale *= scale
        hscale *= scale

        x_color = "#FF8888"
        z_color = "#CCFFCC"
        y_color = "ycolor"

        num_qubits = self.num_qubits
        num_gadgets = self.num_gadgets

        # general width and height of a square
        square_width = int(math.ceil(20 * vscale))
        square_height = int(math.ceil(20 * vscale))

        # width of the text of the phases # TODO round floats (!!)
        text_width = int(math.ceil(50 * vscale))

        bend_degree = int(math.ceil(10))

        # margins between the angle and the legs
        margin_angle_x = int(math.ceil(20 * hscale))
        margin_angle_y = int(math.ceil(20 * hscale))

        # margins between each element
        margin_x = int(math.ceil(10 * hscale))
        margin_y = int(math.ceil(10 * hscale))

        font_size = int(10)

        width = num_gadgets * (
                square_width + margin_x + margin_angle_x + text_width) + margin_x
        height = (num_qubits) * (square_height + margin_y) + (
                square_height + margin_y + margin_angle_y)

        builder = SVGBuilder(width, height)
        builder = builder.add_diagonal_fill(x_color, z_color, y_color)

        prev_x = {qubit: 0 for qubit in range(num_qubits)}

        x = margin_x

        for gadget in self.pauli_gadgets:
            paulis = gadget.paulis
            y = margin_y
            text_coords = (square_width + margin_x + margin_angle_x + x, y)
            text_left_lower_corder = (text_coords[0], text_coords[1] + square_height)
            for qubit in range(num_qubits):
                if qubit == 0:
                    y += square_height + margin_y + margin_angle_y
                else:
                    y += square_height + margin_y
                center_coords = (x + square_width, y)
                if paulis[qubit] == I:
                    continue

                builder.line((prev_x[qubit], y + square_height // 2),
                             (x, y + square_height // 2))
                prev_x[qubit] = x + square_width
                builder.line_bend(text_left_lower_corder, center_coords,
                                  degree=qubit * bend_degree)
                if paulis[qubit] == X:
                    builder.square((x, y), square_width, square_height, x_color)
                elif paulis[qubit] == Y:
                    builder.square((x, y), square_width, square_height, y_color)
                elif paulis[qubit] == Z:
                    builder.square((x, y), square_width, square_height, z_color)

            builder = builder.text_with_square(text_coords, text_width, square_height,
                                               str(gadget.angle))
            x += square_width + margin_x + text_width + margin_angle_x
        y = margin_y
        for qubit in range(num_qubits):
            if qubit == 0:
                y += square_height + margin_y + margin_angle_y
            else:
                y += square_height + margin_y
            builder.line((prev_x[qubit], y + square_height // 2),
                         (width, y + square_height // 2))
        svg_code = repr(builder)

        if svg_code_only:
            return svg_code
        try:
            # pylint: disable = import-outside-toplevel
            from IPython.core.display import SVG  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("You must install the 'IPython' library.") from e

        return SVG(svg_code)

    def _repr_svg_(self):
        """
            Magic method for IPython/Jupyter pretty-printing.
            See https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html
        """
        return self.to_svg(svg_code_only=True)
