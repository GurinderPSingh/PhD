import numpy as np

class CircuitStamps:
    @staticmethod
    def conductance_stamp(G, node1, node2, value):
        if node1 > 0:
            G[node1 - 1, node1 - 1] += value
        if node2 > 0:
            G[node2 - 1, node2 - 1] += value
        if node1 > 0 and node2 > 0:
            G[node1 - 1, node2 - 1] -= value
            G[node2 - 1, node1 - 1] -= value
        return G

    @staticmethod
    def capacitance_stamp(C, node1, node2, value):
        # Storing raw capacitance in matrix C
        if node1 > 0:
            C[node1 - 1, node1 - 1] += value
        if node2 > 0:
            C[node2 - 1, node2 - 1] += value
        if node1 > 0 and node2 > 0:
            C[node1 - 1, node2 - 1] -= value
            C[node2 - 1, node1 - 1] -= value
        return C

    @staticmethod
    def voltage_source_stamp(G, C, RHS, node1, node2, voltage, current_index):
        """
        Stamps an independent voltage source from node1 to node2.
        - Expands G and C by one row & column to account for the new current variable.
        - Stamps +1/-1 in G for the source.
        - Puts 'voltage' into RHS at current_index.
        """
        G_expanded = np.pad(G, ((0,1),(0,1)), mode='constant', constant_values=0)
        C_expanded = np.pad(C, ((0,1),(0,1)), mode='constant', constant_values=0)
        RHS_expanded = np.pad(RHS, (0,1), mode='constant', constant_values=0)

        # If node1 > 0 => stamp +1 at G[node1-1, current_index], etc.
        if node1 > 0:
            G_expanded[node1-1, current_index] =  1
            G_expanded[current_index, node1-1] =  1
        if node2 > 0:
            G_expanded[node2-1, current_index] = -1
            G_expanded[current_index, node2-1] = -1

        # The voltage source imposes V(node1) - V(node2) = voltage in RHS
        RHS_expanded[current_index] = voltage

        return G_expanded, C_expanded, RHS_expanded


def solve_mna(G, RHS):
    """Solves G * x = RHS."""
    return np.linalg.solve(G, RHS)


if __name__ == "__main__":

    # 1) Basic Setup
    num_nodes = 2  # We have 3 "named" nodes (1, 2, 3), plus ground (0)
    G = np.zeros((num_nodes, num_nodes), dtype=complex)
    C = np.zeros((num_nodes, num_nodes), dtype=complex)
    RHS = np.zeros(num_nodes, dtype=complex)

    unknowns = ["V1", "V2", "V3"]  # Node voltages
    # Extra unknowns (currents) will be appended below

    # 2) Stamp a resistor between Node 1 and Node 2 (10 S => 0.1 Ω if purely a resistor)
    G = CircuitStamps.conductance_stamp(G, 1, 2, 10)

    # 3) Stamp a capacitor between Node 2 and ground (C = 0.5 F for demonstration)
    C = CircuitStamps.capacitance_stamp(C, 2, 0, 5e-1)

    # 4) Stamp a voltage source from Node 1 to ground with amplitude = 2 V
    #    We use index = num_nodes (3) for the FIRST extra current variable
    G, C, RHS = CircuitStamps.voltage_source_stamp(G, C, RHS, 1, 0, 2, num_nodes)
    unknowns.append("I_VS1")  # name for the new current variable

    # Now G, C => 4×4, RHS => length 4
    # unknowns => [V1, V2, V3, I_VS1]

    # -------------------------------------------------------------------------
    # AC ANALYSIS at frequency f (non-zero), so the capacitor is not open-circuited.
    # -------------------------------------------------------------------------
    f = 1000  # 1 kHz
    omega = 2 * np.pi * f
    s = 1j * omega

    # Build the final MNA matrix for AC:
    # M = G + s*C
    M = G + s * C

    # 5) Solve M * x = RHS
    x = solve_mna(M, RHS)

    # 6) Print solutions
    print("AC Analysis at f = 1000 Hz:")
    for var_name, val in zip(unknowns, x):
        print(f"  {var_name} = {val.real:.4f} + {val.imag:.4f}j")
