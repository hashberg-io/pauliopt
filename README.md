# PauliOpt: A Python library to simplify quantum circuits.

[![Generic badge](https://img.shields.io/badge/python-3.8+-green.svg)](https://docs.python.org/3.8/)
[![Checked with Mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](https://github.com/python/mypy)
[![PyPI version shields.io](https://img.shields.io/pypi/v/pauliopt.svg)](https://pypi.python.org/pypi/pauliopt/)
[![PyPI status](https://img.shields.io/pypi/status/pauliopt.svg)](https://pypi.python.org/pypi/pauliopt/)
[![Generic badge](https://img.shields.io/badge/supported%20by-Hashberg%20Quantum-blue)](https://hashberg.io/)

PauliOpt is a Python library to simplify quantum circuits composed of phase and Pauli
gadgets. We currently collect architecture-aware synthesis algorithms for circuits of
phase gadgets and Pauli gadgets, and we plan to add more algorithms in the future.

PauliOpt comes with `PhaseCircuit` and `PauliPolynomial` classes for representing phase and Pauli polynomials,

<img width="553" alt="image" src="https://github.com/hashberg-io/pauliopt/assets/13847804/0f8b1480-43fa-47b5-bc84-47d3e8496c68">

<img width="497" alt="image" src="https://github.com/hashberg-io/pauliopt/assets/13847804/96dbbc2d-f02a-4575-827b-ae85ce45f3bb">

and a `Circuit` class that can convert to and from `qiskit` and `pytket`:
```python
Circuit(3, [H(0), H(1), H(2), CRz(π/2, 0, 1), CRz(π/2, 1, 2), CX(0, 1), CX(1, 2)])
```
<img width="541" alt="image" src="https://github.com/hashberg-io/pauliopt/assets/13847804/21e69336-a4f3-46b8-9e98-6d0cd611fbf1">

We view this library as a collection of algorithms for the simplification of quantum
circuits. We currently support the following algorithms:

- "Annealing Optimisation of Mixed ZX Phase Circuits" ([arXiv:2206.11839](https://arxiv.org/abs/2206.11839))
- "Architecture-Aware Synthesis of Stabilizer Circuits from Clifford Tableaus" ([arXiv:
  2309.08972](https://arxiv.org/abs/2309.08972))

**Please Note:** This software library is in a pre-alpha development stage. It is not
currently suitable for use by the public.

## Installation

You can install the library with `pip`:

```
pip install pauliopt
```

If you already have the library installed and would like the latest version, you can also
upgrade with `pip`:

```
pip install --upgrade pauliopt
```

## Documentation

The [documentation](https://hashberg-io.github.io/pauliopt) for this library
was generated with [pdoc](https://pdoc3.github.io/pdoc/). Jupyter notebooks exemplifying
various aspects of the library are available in the [notebooks](./notebooks) folder.

## Usage

The goal of this libary is to provide a simple interface and a collection of algorithms
relevant for the synthesis of quantum circuits.

### Example 1: Simplifying a circuit of phase gadgets

As a first step we can create a ``PhaseCircuit`` object, which represents a trotterized
circuit of phase gadegets.

```python
from pauliopt.phase import PhaseCircuit, Z, X, pi

circ = PhaseCircuit(4)
circ >>= Z(pi / 2) @ {0, 1}
circ >>= X(pi) @ {0, 2}
circ >>= X(-pi / 4) @ {1, 2, 3}
circ >>= Z(pi / 4) @ {0, 3}
circ >>= X(pi / 2) @ {0, 1, 3}
```

We can then define the topology of the device we want to map the circuit to. For example,
we can define a circle topology with 4 qubits:

```python
from pauliopt.topologies import Topology

topology = Topology.cycle(4)
```

Finally we can run a simulated annealing algorithm to find a nice optimized circuit:

```python
from pauliopt.phase import OptimizedPhaseCircuit

num_cx_layers = 3
opt_circ = OptimizedPhaseCircuit(circ, topology, num_cx_layers, rng_seed=0)
```

### Example 2: Synthesis of Clifford tableau's

You can create a Clifford tableau and append/prepend operations (H, S, CX), with the
following code fragment:

```python
from pauliopt.clifford.tableau import CliffordTableau

ct = CliffordTableau(3)

ct.append_h(0)
ct.append_cnot(0, 2)
ct.append_s(1)
```

You can visualize the tableau with the following code fragment:

```python
print(ct)
```

To synthesize the circuit, you can use the following code fragment (note we have used to
topology from above):

```python
from pauliopt.clifford.tableau_synthesis import synthesize_tableau

qc, perm = synthesize_tableau(ct, topology, include_swaps=False)
qc = qc.to_qiskit()
print(qc)
```

## Unit tests

To run the unit tests, install the additional requirements using
our `requirements-dev.txt` (
recommended python: 3.9), then to launch then, run:

```bash
python -m unittest discover -s ./tests/ -p "test_*.py" -v 
```

(You must run this command from the root directory of the repository.)
