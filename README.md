# PauliOpt: A Python library to simplify quantum circuits.
[![Generic badge](https://img.shields.io/badge/python-3.8+-green.svg)](https://docs.python.org/3.8/)
[![Checked with Mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](https://github.com/python/mypy)
[![PyPI version shields.io](https://img.shields.io/pypi/v/pauliopt.svg)](https://pypi.python.org/pypi/pauliopt/)
[![PyPI status](https://img.shields.io/pypi/status/pauliopt.svg)](https://pypi.python.org/pypi/pauliopt/)
[![Generic badge](https://img.shields.io/badge/supported%20by-Hashberg%20Quantum-blue)](https://hashberg.io/)

PauliOpt is a Python library to simplify quantum circuits composed of phase and Pauli gadgets.

<img src="phase_gadget_snippet.png" width="430" title="Snippet of a phase gadget.">

The [documentation](https://sg495.github.io/pauliopt/pauliopt/index.html) for this library was generated with [pdoc](https://pdoc3.github.io/pdoc/).
Jupyter notebooks exemplifying various aspects of the library are available in the [notebooks](./notebooks) folder.

**Please Note:** This software library is in a pre-alpha development stage. It is not currently suitable for use by the public.

You can install the library with `pip`:

```
pip install pauliopt
```

If you already have the library installed and would like the latest version, you can also upgrade with `pip`:

```
pip install --upgrade pauliopt
```

## Optimization of Circuits of Mixed ZX Phase Gadgets

**Step 1.** Create a circuit of Z and X phase gadgets.

<img src="readme-example-1.png" width="800" title="Creation of a circuit of phase gadgets.">

**Step 2.** Select the desired qubit topology.

<img src="readme-example-2.png" width="800" title="Selection of a topology.">

**Step 3.** Instantiate an optimizer for the desired circuit and topology.

<img src="readme-example-3.png" width="800" title="Instantiation of an optimizer.">

**Step 4.** Run a few iterations of simulated annealing and look at the simplified circuit.

<img src="readme-example-4.png" width="800" title="Annealing.">

## Unit tests


To run the unit tests, install the additional requirements using our `requirements.txt` (recommended python: 3.9), then to launch then, run:

```bash
python -m unittest discover -s ./tests/ -p "test_*.py"
```