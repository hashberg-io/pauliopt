class SummedOp:
    def __init__(self, ops):
        self.ops = ops

    def __repr__(self):
        expressions = [op.print_expression(indent=" ") for op in self.ops]
        return f"{expressions[0]}\n" + "\n+".join(expressions[1:])

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            new_ops = [op * other for op in self.ops]
            return SummedOp(new_ops)
        elif isinstance(other, Pauli):
            new_ops = [other * op for op in self.ops]
            return SummedOp(new_ops)
        raise TypeError("Unsupported operation: SummedOp can only be multiplied by a scalar or a Pauli term.")