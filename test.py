import pauliopt.pauliopt

from pauliopt.clifford_.tableau_ import CliffordTableau

if __name__ == '__main__':
    ct = pauliopt.pauliopt.clifford.CliffordTableau(2)
    ct.append_h(0)

    print(ct)
    # ct.append_cnot(0, 1)
    # prin
    # t(ct)

    print(ct.get_tableau())

    print(ct.get_signs())


    ct = CliffordTableau(2)
    ct.append_h(0)

    print(ct.tableau)
    # ct.append_cnot(0, 1)
    # prin
    # t(ct)

    print(ct.signs)
