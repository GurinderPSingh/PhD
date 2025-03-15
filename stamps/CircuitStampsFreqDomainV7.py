import numpy as np
from fpdf import FPDF
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages

class CircuitStamps:
    """
    Class for creating conductance, capacitance, and source matrix stamps for nodal analysis in the frequency domain.
    this versions is almost same as that of V6 new functionality is for saving plots in [df
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
        return G_sC,X, flops, solve_time
    except np.linalg.LinAlgError as e:
        return None, flops, 0  # Return None for solutions if matrix is singular

def format_matrix_for_pdf(matrix):
    """
    Formats a matrix into a readable string for saving into a PDF.
    """
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
    f_min = 1e-3
    f_max = 1e9  # 1 GHz
    num_points = int(1e3)  # 1 million points
    frequencies = np.linspace(f_min, f_max, num_points)
    pdf_save_points = 100  # Save 1000 points to the PDF for summary
    solutions = []

    # PDF Setup
    total_flops = 0
    total_time = 0
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)

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
    for idx, f in enumerate(tqdm(frequencies, desc="Frequency Sweep")):
        # for f in tqdm(frequencies, desc="Frequency Sweep"):
        omega = 2 * np.pi * f
        s = 1j * omega
        G_sC, X, flops, solve_time = solve_mna(G, C, RHS, s)
        total_flops += flops
        total_time += solve_time
        solutions.append(X)
        # Save MNA for first frequency into PDF for clarity
        if idx == 0:
            # if len(solutions) == 0: # Save MNA system only for the first frequency
            pdf.cell(0, 10, txt=f"MNA System for Frequency {f:.2f} Hz:", ln=True)
            pdf.cell(0, 10, txt="Matrix (G + sC):", ln=True)
            formatted_G_sC = format_matrix_for_pdf(G_sC)
            for line in formatted_G_sC.split("\n"):
                pdf.cell(0, 10, txt=line, ln=True)
            pdf.cell(0, 10, txt="RHS (B):", ln=True)
            pdf.cell(0, 10, txt=" ".join(f"{val.real:.2f}+{val.imag:.2f}j" for val in RHS), ln=True)

        # Save selected points to PDF
        # if len(solutions) % (num_points // 1000) == 0:
        if idx % (num_points // pdf_save_points) == 0:
            pdf.cell(0, 10, txt=f"Frequency: {f:.2f} Hz", ln=True)
            pdf.cell(0, 10, txt=f"FLOPs: {flops:.0f}, Solve Time: {solve_time:.6f} seconds", ln=True)
            if X is not None:
                for name, value in zip(unknowns, X):
                    pdf.cell(0, 10, txt=f"{name}: {value.real:.2f} + {value.imag:.2f}j", ln=True)
    end_time = time.perf_counter()

    # Add summary to PDF
    pdf.cell(0, 10, txt=f"Total FLOPs: {total_flops:.0f}", ln=True)
    pdf.cell(0, 10, txt=f"Total Simulation Time: {end_time - start_time:.2f} seconds", ln=True)
    pdf.output("Frequency_Analysis_Summary.pdf")

    print(f"Results saved to Frequency_Analysis_Summary.pdf")
    print(f"Total Simulation Time: {end_time - start_time:.2f} seconds")
    # Print totals
    print(f"Total FLOPs: {total_flops:.0f}")
    print(f"Total Simulation Time: {end_time - start_time:.2f} seconds")

    # Plot results for all nodes

# Example: Frequency points and simulated solutions
# Save magnitude and phase plots to a single PDF
    with PdfPages("Frequency_Response_Plots.pdf") as pdf:
        for i, name in enumerate(unknowns):
            # Extract magnitudes for the current node across all frequencies
            print(f"Type of solutions: {type(solutions)}")
            print(f"Length of solutions: {len(solutions)}")  # Number of nodes
            print(f"Shape of first solution: {solutions[0].shape}")  # Should match len(frequencies)


            print(f"Extracting magnitudes for node {name}")
            print(f"Extracting Phase for node {name}")

            magnitudes = [np.abs(sol[i]) if sol is not None else 0 for sol in solutions]

            # Extract phases for the current node across all frequencies
            phases = [np.angle(sol[i], deg=True) if sol is not None else 0 for sol in solutions]

            # Ensure lengths match before plotting
            if len(frequencies) != len(magnitudes) or len(frequencies) != len(phases):
                print(f"Mismatch: frequencies={len(frequencies)}, magnitudes={len(magnitudes)}, phases={len(phases)}")
                raise ValueError("Mismatch in dimensions of frequencies and solutions.")

            # Generate the magnitude plot
            plt.figure(figsize=(10, 6))
            plt.semilogx(frequencies, magnitudes, label=f"Magnitude of {name}")
            plt.title(f"Frequency Response (Magnitude) of {name}")
            plt.xlabel("Frequency (Hz) [Log Scale]")
            plt.ylabel("Voltage Magnitude")
            plt.grid(which='both', linestyle='--', linewidth=0.5)
            plt.legend()
            pdf.savefig()  # Save the magnitude plot to the PDF
            plt.close()

            # Generate the phase plot
            plt.figure(figsize=(10, 6))
            plt.semilogx(frequencies, phases, label=f"Phase of {name}")
            plt.title(f"Frequency Response (Phase) of {name}")
            plt.xlabel("Frequency (Hz) [Log Scale]")
            plt.ylabel("Phase (Degrees)")
            plt.grid(which='both', linestyle='--', linewidth=0.5)
            plt.legend()
            pdf.savefig()  # Save the phase plot to the PDF
            plt.close()

            print("All plots (magnitude and phase) have been saved to 'Frequency_Response_Plots.pdf'")
