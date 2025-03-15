import numpy as np
from fpdf import FPDF
import time

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
        """
        Pads G, C, and RHS matrices consistently to ensure dimensions match.
        """
        g_matrix = np.pad(g_matrix, ((0, 1), (0, 1)), mode='constant', constant_values=0 + 0j)
        c_matrix = np.pad(c_matrix, ((0, 1), (0, 1)), mode='constant', constant_values=0 + 0j)
        rhs = np.pad(rhs, (0, 1), mode='constant', constant_values=0 + 0j)
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
    def cccs_stamp(g_matrix, c_matrix, ctrl_index, node1, node2, gain):
        g_matrix, c_matrix, _ = CircuitStamps.pad_matrices(g_matrix, c_matrix, np.zeros(g_matrix.shape[0]))
        if node1 > 0:
            g_matrix[node1 - 1, ctrl_index] += gain
        if node2 > 0:
            g_matrix[node2 - 1, ctrl_index] -= gain
        return g_matrix, c_matrix

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

    # def format_matrix(matrix):
    #     """
    #     Formats a matrix into a readable string with aligned columns.
    #     """
    #     formatted = []
    #     for row in matrix:
    #         formatted.append(" ".join(f"{val.real:.2f}+{val.imag:.2f}j" for val in row))
    #     return "\n".join(formatted)

# def solve_mna(G, C, RHS, s):
#     """
#     Solves (G + sC) * X = RHS for X (unknowns).
#     """
#     G_sC = G + s * C
#     if G_sC.shape[0] != RHS.shape[0]:
#         raise ValueError(f"Dimension mismatch: G+sC is {G_sC.shape}, RHS is {RHS.shape}")
#     return np.linalg.solve(G_sC, RHS)
# def solve_mna(G, C, RHS, s):
#     """
#     Solves (G + sC) * X = RHS for X (unknowns).
#     """
#     G_sC = G + s * C
#
#     # Debugging dimensions and matrix content
#     print("Matrix Dimensions:")
#     print(f"G+sC: {G_sC.shape}, RHS: {RHS.shape}")
#     print("\nG+sC Matrix:")
#     print(format_matrix(G_sC))
#     print("\nRHS Vector:")
#     print(" ".join(f"{val.real:.2f}+{val.imag:.2f}j" for val in RHS))
#
#     # Check for dimension mismatch
#     if G_sC.shape[0] != RHS.shape[0]:
#         raise ValueError(f"Dimension mismatch: G+sC is {G_sC.shape}, RHS is {RHS.shape}")
#
#     try:
#         return np.linalg.solve(G_sC, RHS)
#     except np.linalg.LinAlgError as e:
#         print("\nError: Singular matrix detected. Check for floating nodes or missing constraints.")
#         print("Debugging Information:")
#         print("\nConductance Matrix (G):")
#         print(format_matrix(G))
#         print("\nCapacitance Matrix (C):")
#         print(format_matrix(C))
#         print("\nRHS Vector:")
#         print(" ".join(f"{val.real:.2f}+{val.imag:.2f}j" for val in RHS))
#         raise ValueError(f"Matrix is singular. Check circuit connections and constraints.\n{e}")
def solve_mna(G, C, RHS, s):
    """
    Solves (G + sC) * X = RHS for X (unknowns), measures computation time, and estimates FLOPs.
    """
    G_sC = G + s * C

    # Debugging dimensions and matrix content
    print("Matrix Dimensions:")
    print(f"G+sC: {G_sC.shape}, RHS: {RHS.shape}")
    print("\nG+sC Matrix:")
    print(format_matrix(G_sC))
    print("\nRHS Vector:")
    print(" ".join(f"{val.real:.2f}+{val.imag:.2f}j" for val in RHS))

    # Check for dimension mismatch
    if G_sC.shape[0] != RHS.shape[0]:
        raise ValueError(f"Dimension mismatch: G+sC is {G_sC.shape}, RHS is {RHS.shape}")

    try:
        # Measure computation time
        start_time = time.perf_counter()
        X = np.linalg.solve(G_sC, RHS)
        end_time = time.perf_counter()

        # Estimate FLOPs
        n = G_sC.shape[0]
        flops = (2 / 3) * n**3 + 2 * n**2

        # Print computation details
        print(f"\nSolve Time: {end_time - start_time:.6f} seconds")
        print(f"Estimated FLOPs: {flops:.0f}")

        return X
    except np.linalg.LinAlgError as e:
        print("\nError: Singular matrix detected. Check for floating nodes or missing constraints.")
        raise ValueError(f"Matrix is singular. Check circuit connections and constraints.\n{e}")



def format_matrix(matrix):
    formatted = []
    for row in matrix:
        formatted.append(" ".join(f"{val.real:.2f}+{val.imag:.2f}j" for val in row))
    return "\n".join(formatted)


if __name__ == "__main__":
    num_nodes = 4
    unknowns = [f"V_{i+1}" for i in range(num_nodes)]
    G = np.zeros((num_nodes, num_nodes), dtype=np.complex128)
    C = np.zeros((num_nodes, num_nodes), dtype=np.complex128)
    RHS = np.zeros(num_nodes, dtype=np.complex128)

    # Frequency range setup
    f_min = 1
    f_max = 10000000
    num_points = 1000
    frequencies = np.linspace(f_min, f_max, num_points)
    solutions = []

    # Adding components
    G = CircuitStamps.conductance_stamp(G, 1, 2, 10)
    C = CircuitStamps.capacitance_stamp(C, 2, 3, 5e-6)
    G, C, RHS = CircuitStamps.inductor_stamp(G, C, RHS, 3, 4, 1, num_nodes)
    unknowns.append("I_L1")
    G, C, RHS = CircuitStamps.voltage_source_stamp(G, C, RHS, 1, 0, 2, num_nodes + 1)
    unknowns.append("I_VS1")
    CircuitStamps.current_source_stamp(RHS, 1, 0, 0.01)
    # G, C, RHS = CircuitStamps.vccs_stamp(G, C, RHS, 4, 0, 1, 2, 0.1)  # Voltage-controlled current source
    G, C, RHS = CircuitStamps.vcvs_stamp(G, C, RHS, 1, 2, 3, 4, 2, num_nodes + 2)
    unknowns.append("I_VCVS1")
    # G, C = CircuitStamps.cccs_stamp(G, C, num_nodes + 2, 1, 2, 1.5)
    G, C, RHS = CircuitStamps.ccvs_stamp(G, C, RHS, 1, 2, num_nodes + 2, 1.2, num_nodes + 3)
    unknowns.append("I_CCVS1")

    # Format matrices
    formatted_G = format_matrix(G)
    formatted_C = format_matrix(C)
    formatted_RHS = " ".join(f"{val.real:.2f}+{val.imag:.2f}j" for val in RHS)

    # Save to PDF
    output_text = f"Conductance Matrix (G):\n{formatted_G}\n\nCapacitance Matrix (C):\n{formatted_C}\n\nRight-Hand Side Vector (RHS):\n{formatted_RHS}"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in output_text.split('\n'):
        pdf.cell(0, 10, txt=line, ln=True)
    pdf.output("MNA_system_with_G_plus_SC.pdf")

    print("MNA system saved to MNA_system_with_G_plus_SC.pdf")
    # Solve for each frequency
    for f in frequencies:
        omega = 2 * np.pi * f
        s = 1j * omega
        if f == 0:
            print("Performing DC analysis...")
        try:
            X = solve_mna(G, C, RHS, s)
            solutions.append(X)
        except ValueError as e:
                print(f"Frequency {f} Hz caused an issue: {e}")
                solutions.append(X)
    # Solve and display results
        solutions = solve_mna(G, C, RHS, s)
        print("\nUnknowns:")
        for name, value in zip(unknowns, solutions):
            print(f"{name}: {value.real:.2f} + {value.imag:.2f}j V/A")
