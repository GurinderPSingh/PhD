import numpy as np
from fpdf import FPDF
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

class CircuitStamps:
    """
    Class for creating conductance, capacitance, and source matrix stamps for nodal analysis in the frequency domain.
    """

    @staticmethod
    def conductance_stamp(matrix, node1, node2, value):
        if node1 > 0:
            matrix[node1 - 1, node1 - 1] += value
        if node2 > 0:
            matrix[node2 - 1, node2 - 1] += value
        if node1 > 0 and node2 > 0:
            matrix[node1 - 1, node2 - 1] -= value
            matrix[node2 - 1, node1 - 1] -= value
        return matrix

    @staticmethod
    def capacitance_stamp(c_matrix, node1, node2, value):
        if node1 > 0:
            c_matrix[node1 - 1, node1 - 1] += value
        if node2 > 0:
            c_matrix[node2 - 1, node2 - 1] += value
        if node1 > 0 and node2 > 0:
            c_matrix[node1 - 1, node2 - 1] -= value
            c_matrix[node2 - 1, node1 - 1] -= value
        return c_matrix

    @staticmethod
    def pad_matrices(g_matrix, c_matrix, rhs):
        g_matrix = np.pad(g_matrix, ((0, 1), (0, 1)), mode='constant', constant_values=0 + 0j)
        c_matrix = np.pad(c_matrix, ((0, 1), (0, 1)), mode='constant', constant_values=0 + 0j)
        rhs = np.pad(rhs, (0, 1), mode='constant', constant_values=0 + 0j)
        return g_matrix, c_matrix, rhs

    @staticmethod
    def voltage_source_stamp(g_matrix, c_matrix, rhs, node1, node2, voltage, current_index):
        g_matrix, c_matrix, rhs = CircuitStamps.pad_matrices(g_matrix, c_matrix, rhs)
        if node1 > 0:
            g_matrix[node1 - 1, current_index] = 1
            g_matrix[current_index, node1 - 1] = 1
        if node2 > 0:
            g_matrix[node2 - 1, current_index] = -1
            g_matrix[current_index, node2 - 1] = -1
        rhs[current_index] = voltage
        return g_matrix, c_matrix, rhs

    @staticmethod
    def current_source_stamp(rhs, node1, node2, current):
        if node1 > 0:
            rhs[node1 - 1] -= current
        if node2 > 0:
            rhs[node2 - 1] += current
        return rhs

    @staticmethod
    def inductor_stamp(g_matrix, c_matrix, rhs, node1, node2, inductance, current_index):
        g_matrix, c_matrix, rhs = CircuitStamps.pad_matrices(g_matrix, c_matrix, rhs)
        if node1 > 0:
            g_matrix[node1 - 1, current_index] = 1
            g_matrix[current_index, node1 - 1] = 1
        if node2 > 0:
            g_matrix[node2 - 1, current_index] = -1
            g_matrix[current_index, node2 - 1] = -1
        c_matrix[current_index, current_index] = inductance
        return g_matrix, c_matrix, rhs

    @staticmethod
    def vccs_stamp(g_matrix, c_matrix, rhs, node1, node2, ctrl_node1, ctrl_node2, gm):
        g_matrix, c_matrix, rhs = CircuitStamps.pad_matrices(g_matrix, c_matrix, rhs)
        if node1 > 0 and ctrl_node1 > 0:
            g_matrix[node1 - 1, ctrl_node1 - 1] += gm
        if node1 > 0 and ctrl_node2 > 0:
            g_matrix[node1 - 1, ctrl_node2 - 1] -= gm
        if node2 > 0 and ctrl_node1 > 0:
            g_matrix[node2 - 1, ctrl_node1 - 1] -= gm
        if node2 > 0 and ctrl_node2 > 0:
            g_matrix[node2 - 1, ctrl_node2 - 1] += gm
        return g_matrix, c_matrix, rhs

    @staticmethod
    def vcvs_stamp(g_matrix, c_matrix, rhs, node1, node2, ctrl_node1, ctrl_node2, gain, current_index):
        g_matrix, c_matrix, rhs = CircuitStamps.pad_matrices(g_matrix, c_matrix, rhs)
        if node1 > 0:
            g_matrix[node1 - 1, current_index] = 1
            g_matrix[current_index, node1 - 1] = 1
        if node2 > 0:
            g_matrix[node2 - 1, current_index] = -1
            g_matrix[current_index, node2 - 1] = -1
        if ctrl_node1 > 0:
            g_matrix[current_index, ctrl_node1 - 1] -= gain
        if ctrl_node2 > 0:
            g_matrix[current_index, ctrl_node2 - 1] += gain
        return g_matrix, c_matrix, rhs

    @staticmethod
    def cccs_stamp(g_matrix, c_matrix, rhs, ctrl_index, node1, node2, gain):
        g_matrix, c_matrix, rhs = CircuitStamps.pad_matrices(g_matrix, c_matrix, rhs)
        if node1 > 0:
            g_matrix[node1 - 1, ctrl_index] += gain
        if node2 > 0:
            g_matrix[node2 - 1, ctrl_index] -= gain
        return g_matrix, c_matrix, rhs

    @staticmethod
    def ccvs_stamp(g_matrix, c_matrix, rhs, node1, node2, ctrl_index, gain, current_index):
        g_matrix, c_matrix, rhs = CircuitStamps.pad_matrices(g_matrix, c_matrix, rhs)
        if node1 > 0:
            g_matrix[node1 - 1, current_index] = 1
            g_matrix[current_index, node1 - 1] = 1
        if node2 > 0:
            g_matrix[node2 - 1, current_index] = -1
            g_matrix[current_index, node2 - 1] = -1
        g_matrix[current_index, ctrl_index] -= gain
        return g_matrix, c_matrix, rhs

def solve_mna(G, C, RHS, s):
    """
    Solves (G + sC) * X = RHS for X (unknowns) and estimates FLOPs.
    """
    G_sC = G + s * C
    n = G_sC.shape[0]
    flops = (2 / 3) * n**3 + 2 * n**2  # FLOPs estimation

    try:
        start_time = time.perf_counter()
        X = np.linalg.solve(G_sC, RHS)
        solve_time = time.perf_counter() - start_time
        return X, flops, solve_time
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Matrix is singular. Check circuit connections and constraints.\n{e}")

if __name__ == "__main__":
    num_nodes = 4
    unknowns = [f"V_{i+1}" for i in range(num_nodes)]
    G = np.zeros((num_nodes, num_nodes), dtype=np.complex128)
    C = np.zeros((num_nodes, num_nodes), dtype=np.complex128)
    RHS = np.zeros(num_nodes, dtype=np.complex128)

    # Frequency range setup
    f_min = 1e-3
    f_max = 1000000000
    num_points = 10000
    frequencies = np.linspace(f_min, f_max, num_points)
    solutions = []
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Adding components
    G = CircuitStamps.conductance_stamp(G, 1, 2, 10)
    C = CircuitStamps.capacitance_stamp(C, 2, 3, 5e-6)
    G, C, RHS = CircuitStamps.inductor_stamp(G, C, RHS, 3, 4, 1, num_nodes)
    unknowns.append("I_L1")
    G, C, RHS = CircuitStamps.voltage_source_stamp(G, C, RHS, 1, 0, 2, num_nodes + 1)
    unknowns.append("I_VS1")
    RHS = CircuitStamps.current_source_stamp(RHS, 1, 0, 0.01)
    # G, C, RHS = CircuitStamps.vccs_stamp(G, C, RHS, 4, 0, 1, 2, 0.1)  # Voltage-controlled current source
    G, C, RHS = CircuitStamps.vcvs_stamp(G, C, RHS, 1, 2, 3, 4, 2, num_nodes + 2)
    unknowns.append("I_VCVS1")
    # G, C, RHS = CircuitStamps.cccs_stamp(G, C, RHS, num_nodes + 2, 1, 2, 1.5)
    G, C, RHS = CircuitStamps.ccvs_stamp(G, C, RHS, 1, 2, num_nodes + 2, 1.2, num_nodes + 3)
    unknowns.append("I_CCVS1")

    # Solve for each frequency
    start_time = time.perf_counter()
    for f in frequencies:
        omega = 2 * np.pi * f
        s = 1j * omega
        try:
            X, flops, solve_time = solve_mna(G, C, RHS, s)
            solutions.append(X)
            pdf.cell(0, 10, txt=f"Frequency: {f:.2f} Hz", ln=True)
            pdf.cell(0, 10, txt=f"FLOPs: {flops:.0f}, Solve Time: {solve_time:.6f} seconds", ln=True)
            for name, value in zip(unknowns, X):
                pdf.cell(0, 10, txt=f"{name}: {value.real:.2f} + {value.imag:.2f}j", ln=True)
        except ValueError as e:
            print(f"Frequency {f:.2f} Hz caused an issue: {e}")
            pdf.cell(0, 10, txt=f"Frequency {f:.2f} Hz caused an issue: {e}", ln=True)
            solutions.append(None)
    end_time = time.perf_counter()

    pdf.cell(0, 10, txt=f"Total Simulation Time: {end_time - start_time:.2f} seconds", ln=True)
    pdf.output("Frequency_Analysis_Results.pdf")

    print(f"Results saved to Frequency_Analysis_Results.pdf")
