from pauliopt.pytket import convert_cx_layer
from pytket import Circuit
from pauliopt.phase import CXCircuitLayer
from pauliopt.topologies import Topology 

def test_simple():
    topo = Topology(2, {(0, 1)})
    pauliopt_cx = CXCircuitLayer(topo, {(0, 1)})
    pytket_cx = Circuit(2)
    pytket_cx.CX(0, 1)

    result = convert_cx_layer(pauliopt_cx)
    assert result == pytket_cx

def test_empty():
    topo = Topology(2, {(0, 1)})
    pauliopt_cx = CXCircuitLayer(topo)
    pytket_cx = Circuit(2)

    result = convert_cx_layer(pauliopt_cx)
    assert result == pytket_cx
