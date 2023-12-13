from abc import ABC
from itertools import combinations
from math import ceil

from pauliopt.phase import pi
from pauliopt.phase import Z as ZHead
from pauliopt.phase import X as XHead
from pauliopt.phase.phase_circuits import PhaseGadget


class Gate(ABC):
    """ Base class for quantum gates. """
    def __init__(self, *qubits):
        self.qubits = qubits
        self.name = self.__class__.__name__
        if len(qubits) != self.n_qubits:
            name, n_qubits = self.name, self.n_qubits
            raise ValueError(f"{name} takes {n_qubits} qubits, not {len(qubits)}.")

    def __repr__(self) -> str:
        args = map(repr, self.qubits)
        return f"{self.name}({', '.join(args)})"

    @property
    def width(self):
        """ Width of gate used for drawing. """
        if getattr(self, 'draw_as_zx', False):
            n_columns = max(self._gen_columns())
            return n_columns * 20 + 40

        vscale = 1.0
        line_height = int(ceil(30*vscale))
        min_qubit = min(self.qubits)
        max_qubit = max(self.qubits)
        text = repr(self)
        box_height = max(30, (max_qubit - min_qubit) * line_height + 10)
        box_width = max(8 * len(text), box_height * 1.5)
        return int(box_width)

    def _gen_columns(self):
        n_columns = 0
        gate_cls = None
        # shift by 3 columns if a multi-legged gadget was drawn
        big_gadget_drawn = False
        for g in self.decomp:
            if gate_cls != (type(g), getattr(g, 'basis', None)):
                gate_cls = (type(g), getattr(g, 'basis', None))
                n_columns += 3 if big_gadget_drawn else 1
                big_gadget_drawn = False
            if isinstance(g, PhaseGadget) and len(g.qubits) > 1:
                big_gadget_drawn = True
            yield n_columns
        n_columns += 3 if big_gadget_drawn else 1
        big_gadget_drawn = False
        yield n_columns

    def draw_zx(self, builder, base, row_width, params):
        n_columns = max(self._gen_columns())
        step = row_width / n_columns

        busy = set()
        for g, column_idx in zip(self.decomp, self._gen_columns()):
            x, y = base[0] + column_idx * step, base[1]

            if len(g.qubits) > 1:
                # gadget body blocks text
                busy.add(min(g.qubits) + 1)
            text_above = True
            if len(g.qubits) == 1 and tuple(g.qubits)[0] in busy:
                text_above = False
            if isinstance(g, PhaseGadget):
                draw_gadget(builder, (x, y), row_width, params, g, text_above)
            else:
                g.draw(builder, base, row_width, params)

    def draw_box(self, builder, base, row_width, params):
        line_height = params['line_height']
        font_size = params['font_size']

        min_qubit = min(self.qubits)
        max_qubit = max(self.qubits)
        x = base[0] + row_width / 2
        y = base[1] + ((min_qubit + max_qubit) / 2 + 1) * line_height
        text = repr(self)
        box_height = max(30, (max_qubit - min_qubit) * line_height + 10)
        box_width = max(8 * len(text), box_height * 1.5)

        builder.rect((x, y), box_width, box_height, '#FFFFFF')
        builder.text((x, y), text, font_size=font_size, center=True)

    def draw(self, builder, base, row_width, params):
        # builder.line((base[0], 0), (base[0], 100))
        # builder.line((base[0] + row_width, 0), (base[0] + row_width, 100))
        if getattr(self, 'draw_as_zx', False):
            self.draw_zx(builder, base, row_width, params)
        else:
            self.draw_box(builder, base, row_width, params)

    @property
    def gadgets(self):
        """ List of gadgets used to implement this gate. """
        if not hasattr(self, 'decomp'):
            raise NotImplementedError

        gadgets = [
            g if isinstance(g, (ZHead, XHead)) else g.gadgets
            for g in self.decomp
        ]
        return gadgets


class PhaseGate(Gate):
    def __init__(self, phase, *qubits):
        super().__init__(*qubits)
        self.phase = phase

    def __repr__(self) -> str:
        args = map(repr, self.qubits)
        return f"{self.name}({self.phase}, {', '.join(args)})"


class H(Gate):
    n_qubits = 1
    width = 40

    @property
    def decomp(self):
        p = pi / 2
        q, = self.qubits
        return [ZHead(p) @ {q}, XHead(p) @ {q}, ZHead(p) @ {q}]

    def draw(self, builder, base, row_width, params):
        line_height = params['line_height']
        r = params['r']
        hcolor = params['hcolor']
        qubit = self.qubits[0]
        x = base[0] + row_width / 2
        y = base[1] + (qubit+1)*line_height
        builder.rect((x, y), 1.5*r, 1.5*r, hcolor)


class X(Gate):
    n_qubits = 1
    draw_as_zx = True

    @property
    def decomp(self):
        return [XHead(pi) @ {self.qubits[0]}]


class Z(Gate):
    n_qubits = 1
    draw_as_zx = True

    @property
    def decomp(self):
        return [ZHead(pi) @ {self.qubits[0]}]


class Y(Gate):
    n_qubits = 1
    draw_as_zx = True

    @property
    def decomp(self):
        # TODO check this
        return [ZHead(pi) @ {self.qubits[0]}, XHead(pi) @ {self.qubits[0]}]


class S(Gate):
    n_qubits = 1
    draw_as_zx = True

    @property
    def decomp(self):
        return [ZHead(pi / 2) @ {self.qubits[0]}]


class T(Gate):
    n_qubits = 1
    draw_as_zx = True

    @property
    def decomp(self):
        return [ZHead(pi / 4) @ {self.qubits[0]}]


class SWAP(Gate):
    n_qubits = 2
    draw_as_zx = True

    @property
    def decomp(self):
        q0, q1 = self.qubits
        return [CX(q0, q1), CX(q1, q0), CX(q0, q1)]


class CX(Gate):
    n_qubits = 2
    draw_as_zx = True
    width = 40

    @property
    def decomp(self):
        q0, q1 = self.qubits
        return [H(q1), CZ(q0, q1), H(q1)]

    def draw(self, builder, base, row_width, params):
        r = params['r']
        line_height = params['line_height']
        zcolor = params['zcolor']
        xcolor = params['xcolor']

        ctrl, trgt = self.qubits
        x = base[0] + row_width / 2
        y_ctrl = base[1] + (ctrl+1)*line_height
        y_trgt = base[1] + (trgt+1)*line_height
        builder.line((x, y_ctrl), (x, y_trgt))
        builder.circle((x, y_ctrl), r, zcolor)
        builder.circle((x, y_trgt), r, xcolor)


class CY(Gate):
    n_qubits = 2
    draw_as_zx = True

    @property
    def decomp(self):
        q0, q1 = self.qubits
        return [XHead(pi/2) @ {q1}, CZ(q0, q1), XHead(-pi/2) @ {q1}]


class CZ(Gate):
    n_qubits = 2
    draw_as_zx = True

    @property
    def decomp(self):
        q0, q1 = self.qubits
        return [ZHead(pi/2) @ {q0, q1}]

    def draw(self, builder, base, row_width, params):
        r = params['r']
        line_height = params['line_height']
        zcolor = params['zcolor']
        hcolor = params['hcolor']

        ctrl, trgt = self.qubits
        x = base[0] + row_width / 2
        y_ctrl = base[1] + (ctrl+1)*line_height
        y_trgt = base[1] + (trgt+1)*line_height
        y_mid = (y_ctrl + y_trgt) / 2
        builder.line((x, y_ctrl), (x, y_trgt))
        builder.circle((x, y_ctrl), r, zcolor)
        builder.circle((x, y_trgt), r, zcolor)
        builder.rect((x, y_mid), r, r, hcolor)


class CCX(Gate):
    n_qubits = 3
    draw_as_zx = True

    @property
    def decomp(self):
        q0, q1, q2 = self.qubits
        return [H(q2), CCZ(q0, q1, q2), H(q2)]


class CCZ(Gate):
    n_qubits = 3
    draw_as_zx = True

    @property
    def decomp(self):
        gs = []

        for i in range(1, self.n_qubits + 1):
            for qs in combinations(self.qubits, i):
                gs.append(ZHead(-1 ** len(qs) * pi/2) @ set(qs))
        return gs


class Rx(PhaseGate):
    n_qubits = 1
    draw_as_zx = True

    @property
    def decomp(self):
        q, = self.qubits
        return [XHead(self.phase) @ {q}]


class Ry(PhaseGate):
    n_qubits = 1
    draw_as_zx = True

    @property
    def decomp(self):
        q, = self.qubits
        return [XHead(pi/2) @ {q}, ZHead(self.phase) @ {q}, XHead(-pi/2) @ {q}]


class Rz(PhaseGate):
    n_qubits = 1
    draw_as_zx = True

    @property
    def decomp(self):
        q, = self.qubits
        return [ZHead(self.phase) @ {q}]


class CRx(PhaseGate):
    n_qubits = 2
    draw_gadget = True

    @property
    def decomp(self):
        p = self.phase
        q0, q1 = self.qubits
        return [H(q1), CRz(p, q0, q1), H(q1)]


class CRy(PhaseGate):
    n_qubits = 2
    draw_gadget = True

    @property
    def decomp(self):
        p = self.phase
        q0, q1 = self.qubits
        return [XHead(pi/2, q1), CRz(p, q0, q1), XHead(-pi/2, q1)]


class CRz(PhaseGate):
    n_qubits = 2
    draw_as_zx = True

    @property
    def decomp(self):
        p = self.phase
        q0, q1 = self.qubits
        return [ZHead(-p/2) @ {q0, q1}, ZHead(p/2) @ {q1}]


CNOT = CX


def draw_gadget(builder, base, row_width, params, gadget, text_above=True):
    line_height = params['line_height']
    text_off_x, text_off_y = params['text_off']
    font_size = params['font_size']
    r = params['r']

    if gadget.basis == 'Z':
        color1, color2 = params['zcolor'], params['xcolor']
    elif gadget.basis == 'X':
        color1, color2 = params['xcolor'], params['zcolor']
    else:
        raise ValueError

    step = row_width / 4
    body = min(gadget.qubits) + 0.5
    x = base[0]
    x_body = x + step
    y_body = base[1] + (body+1) * line_height

    if len(gadget.qubits) > 1:
        for q in gadget.qubits:
            y = base[1] + (q+1)*line_height
            builder.line((x, y), (x_body, y_body))
            builder.circle((x, y), r, color1)
        builder.line((x_body, y_body), (x_body + step, y_body))
        builder.circle((x_body, y_body), r, color2)
        builder.circle((x_body + step, y_body), r, color1)

        text_x = x_body + step + text_off_x
        text_y = y_body
    else:
        assert len(gadget.qubits) == 1
        q = tuple(gadget.qubits)[0]
        y = base[1] + (q+1)*line_height
        builder.circle((x, y), r, color1)

        text_x = x + text_off_x
        if text_above:
            text_y = y + text_off_y
        else:
            text_y = y - text_off_y

    builder.text((text_x, text_y), str(gadget.angle), font_size=font_size)
