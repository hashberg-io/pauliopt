"""
A general class for quantum circuits, with a ZX / Gadget representation.
"""

from math import log10
from typing import List

import numpy as np

from pauliopt.gates import *
from pauliopt.utils import SVGBuilder

QISKIT_CONVERSION = {
    "h": lambda qubits, _: H(*qubits),
    "x": lambda qubits, _: X(*qubits),
    "y": lambda qubits, _: Y(*qubits),
    "z": lambda qubits, _: Z(*qubits),
    "s": lambda qubits, _: S(*qubits),
    "sdg": lambda qubits, _: Sdg(*qubits),
    "t": lambda qubits, _: T(*qubits),
    "tdg": lambda qubits, _: Tdg(*qubits),
    "swap": lambda qubits, _: SWAP(*qubits),
    "cx": lambda qubits, _: CX(*qubits),
    "cy": lambda qubits, _: CY(*qubits),
    "cz": lambda qubits, _: CZ(*qubits),
    "ccx": lambda qubits, _: CCX(*qubits),
    "ccz": lambda qubits, _: CCZ(*qubits),
    "rx": lambda qubits, params: Rx(params[0], *qubits),
    "ry": lambda qubits, params: Ry(params[0], *qubits),
    "rz": lambda qubits, params: Rz(params[0], *qubits),
    "crx": lambda qubits, params: CRx(params[0], *qubits),
    "cry": lambda qubits, params: CRy(params[0], *qubits),
    "crz": lambda qubits, params: CRz(params[0], *qubits),
}


def _get_phase_qiskit(params):
    if len(params) == 0:
        return None
    else:
        return params


def _get_qubits_qiskit(qubits, qreg):
    """
    Helper method to read the qubit indices from the qiskit quantum register.
    """
    qubits_ = []
    for qubit in qubits:
        qubits_.append(qreg.index(qubit))

    return tuple(qubits_)


class Circuit:
    """Class for representing quantum circuits."""

    def __init__(self, n_qubits, _gates=None):
        self.n_qubits = n_qubits
        self._gates = [] if _gates is None else _gates

        for gate in self._gates:
            self._check_gate(gate)

    def add_gate(self, gate):
        return self.add_gates([gate])

    def add_gates(self, gates):
        for gate in gates:
            self._check_gate(gate)
        self._gates.extend(gates)
        return self

    def to_phase_circuit(self):
        from pauliopt.phase import PhaseCircuit
        gadgets = [g for gate in self._gates for g in gate.gadgets]
        return PhaseCircuit(self.n_qubits, gadgets)

    def _check_gate(self, gate):
        n_qubits = self.n_qubits
        if not isinstance(gate, Gate):
            raise TypeError(f"{gate} is not a valid gate.")

        if len(set(gate.qubits)) != len(gate.qubits):
            raise ValueError(f"{gate.qubits} are not unique.")

        if any(not (0 <= qubit < n_qubits) for qubit in gate.qubits):
            msg = f"{gate} acts out of range for {n_qubits} qubit circuit."
            raise ValueError(msg)

    def __repr__(self) -> str:
        return f"Circuit({self.n_qubits}, {self._gates})"

    def _to_svg(
        self,
        *,
        zcolor: str = "#CCFFCC",
        xcolor: str = "#FF8888",
        hcolor: str = "#FFFF00",
        hscale: float = 1.0,
        vscale: float = 1.0,
        scale: float = 1.0,
        svg_code_only: bool = False,
    ):
        # pylint: disable = too-many-locals, too-many-statements, too-many-branches
        num_qubits = self.n_qubits
        vscale *= scale
        hscale *= scale

        gates = list(self._gates)
        _layers: List[int] = [0] * num_qubits
        row_widths = [0] * len(gates)
        ds = []
        max_gates_depth: int = 0
        for gate in gates:
            m = min(gate.qubits)
            M = max(gate.qubits)
            d = max(_layers[q] for q in range(m, M + 1))
            ds.append(d)
            max_gates_depth = max(max_gates_depth, d + 1)
            for q in range(m, M + 1):
                _layers[q] = d + 1

            gate_width = getattr(gate, "width", int(ceil(40 * hscale)))
            row_widths[d] = max(row_widths[d], gate_width)
        num_digits = int(ceil(log10(num_qubits)))
        line_height = int(ceil(30 * vscale))
        pad_x = int(ceil(10 * hscale))
        pad_y = int(ceil(20 * vscale))
        r = 6
        total_gate_width = sum(row_widths)
        font_size = 2 * r
        pad_x += font_size * (num_digits + 1)
        width = total_gate_width
        height = pad_y + line_height * (num_qubits + 1)
        builder = SVGBuilder(width, height)
        row_pos = np.cumsum([pad_x] + row_widths).tolist()

        params = {
            "text_off": (10 * hscale, -10 * vscale),
            "zcolor": zcolor,
            "xcolor": xcolor,
            "hcolor": hcolor,
            "line_height": line_height,
            "font_size": font_size,
            "r": r,
        }

        # draw gate
        for d, gate in zip(ds, gates):
            base = row_pos[d], pad_y
            gate.draw(builder, base, row_widths[d], params)

        # label wires
        width = pad_x + total_gate_width + pad_x
        _builder = SVGBuilder(width, height)
        for q in range(num_qubits):
            y = pad_y + (q + 1) * line_height
            txt = f"{str(q):>{num_digits}}"
            _builder.line((pad_x, y), (width - pad_x, y))
            _builder.text((0, y), txt, font_size=font_size)
            _builder.text((width - pad_x + r, y), txt, font_size=font_size)
        _builder >>= builder
        svg_code = repr(_builder)
        if svg_code_only:
            return svg_code
        try:
            # pylint: disable = import-outside-toplevel
            from IPython.core.display import SVG  # type: ignore
        except ModuleNotFoundError:
            raise ModuleNotFoundError("You must install the 'IPython' library.")
        return SVG(svg_code)

    @staticmethod
    def from_qiskit(qc: "qiskit.QuantumCircuit"):
        circ = Circuit(qc.num_qubits)

        for inst in qc:
            qubits = _get_qubits_qiskit(inst.qubits, qc.qregs[0])
            phase = _get_phase_qiskit(inst.operation.params)
            circ.add_gate(QISKIT_CONVERSION[inst.operation.name](qubits, phase))

        return circ

    def to_qiskit(self):
        try:
            from qiskit import QuantumCircuit
        except ModuleNotFoundError:
            raise ModuleNotFoundError("You must install the 'qiskit' library.")

        qc = QuantumCircuit(self.n_qubits)

        for gate in self._gates:
            op, qubits = gate.to_qiskit()
            qc.append(op, qubits)
        return qc
