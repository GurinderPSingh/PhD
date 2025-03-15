from multiprocessing import Pool, cpu_count
import numpy as np
from fpdf import FPDF
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
import os
import re

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

        # Debugging logs
        print(f"Stamping voltage source: Node1={node1}, Node2={node2}, Voltage={voltage}")
        print(f"Matrix Dimensions: {g_matrix.shape}, Current Index: {current_index}")
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

def solve_mna_for_frequency(f, G, C, RHS):
    """
    Solve MNA for a single frequency.
    """
    omega = 2 * np.pi * f
    s = 1j * omega
    G_sC = G + s * C
    n = G_sC.shape[0]
    flops = (2 / 3) * n**3 + 2 * n**2  # FLOPs estimation

    try:
        start_time = time.perf_counter()
        X = np.linalg.solve(G_sC, RHS)
        solve_time = time.perf_counter() - start_time
        return f, G_sC, X, flops, solve_time
    except np.linalg.LinAlgError:
        return f, G_sC, None, flops, 0  # None for failed solutions

def solve_frequency_wrapper(args):
    """
    Wrapper function to unpack arguments for multiprocessing.
    """
    f, G, C, RHS = args
    return solve_mna_for_frequency(f, G, C, RHS)


def format_matrix_for_pdf(matrix):
    """
    Formats a matrix into a readable string for saving into a PDF.
    """
    formatted = []
    for row in matrix:
        formatted.append(" ".join(f"{val.real:.2f}+{val.imag:.2f}j" for val in row))
    return "\n".join(formatted)

def parse_netlist(file_path):
    """
    Parse the netlist file and extract circuit elements.
    """
    elements = []
    nodes = set()

    with open(file_path, 'r') as file:
        for line in file:
            line = line.split(';')[0].strip()  # Remove comments
            if not line:
                continue
            tokens = re.split(r'\s+', line)
            element_type = tokens[0][0].upper()  # First character determines type
            name = tokens[0]
            connections = list(map(int, tokens[1:3]))
            value = float(tokens[3]) if len(tokens) > 3 else None
            control = list(map(int, tokens[3:5])) if element_type in ['VCCS', 'VCVS', 'CCCS', 'CCVS'] else None
            gain = float(tokens[5]) if len(tokens) > 5 else None

            elements.append({
                "type": element_type,
                "name": name,
                "connections": connections,
                "value": value,
                "control": control,
                "gain": gain,
            })
            nodes.update(connections)
            if control:
                nodes.update(control)

    return elements, max(nodes) + 1  # Total nodes (0-based indexing)

def construct_circuit(elements, num_nodes):
    """
    Construct the circuit matrices (G, C, RHS) based on the parsed elements.
    """
    G = np.zeros((num_nodes, num_nodes), dtype=np.complex128)
    C = np.zeros((num_nodes, num_nodes), dtype=np.complex128)
    RHS = np.zeros(num_nodes, dtype=np.complex128)
    unknowns = []

    for element in elements:
        n1, n2 = element["connections"]
        if element["type"] == 'R':  # Resistor
            G = CircuitStamps.conductance_stamp(G, n1, n2, 1 / element["value"])
        elif element["type"] == 'C':  # Capacitor
            C = CircuitStamps.capacitance_stamp(C, n1, n2, element["value"])
        elif element["type"] == 'L':  # Inductor
            index = len(unknowns) + num_nodes
            G, C, RHS = CircuitStamps.inductor_stamp(G, C, RHS, n1, n2, element["value"], index)
            unknowns.append(f"I_{element['name']}")
        elif element["type"] == 'V':  # Voltage source
            index = len(unknowns) + num_nodes
            G, C, RHS = CircuitStamps.voltage_source_stamp(G, C, RHS, n1, n2, element["value"], index)
            unknowns.append(f"I_{element['name']}")
        elif element["type"] == 'I':  # Current source
            RHS = CircuitStamps.current_source_stamp(RHS, n1, n2, element["value"])
        elif element["type"] == 'VCCS':  # Voltage-controlled current source
            n3, n4 = element["control"]
            G, C, RHS = CircuitStamps.vccs_stamp(G, C, RHS, n1, n2, n3, n4, element["gain"])
        elif element["type"] == 'VCVS':  # Voltage-controlled voltage source
            n3, n4 = element["control"]
            index = len(unknowns) + num_nodes
            G, C, RHS = CircuitStamps.vcvs_stamp(G, C, RHS, n1, n2, n3, n4, element["gain"], index)
            unknowns.append(f"I_{element['name']}")
        elif element["type"] == 'CCCS':  # Current-controlled current source
            ctrl_index = len(unknowns) + num_nodes
            G, C, RHS = CircuitStamps.cccs_stamp(G, C, RHS, ctrl_index, n1, n2, element["gain"])
        elif element["type"] == 'CCVS':  # Current-controlled voltage source
            ctrl_index = len(unknowns) + num_nodes
            index = len(unknowns) + num_nodes + 1
            G, C, RHS = CircuitStamps.ccvs_stamp(G, C, RHS, n1, n2, ctrl_index, element["gain"], index)
            unknowns.append(f"I_{element['name']}")

    return G, C, RHS, unknowns


if __name__ == "__main__":
    # Netlist directory and file selection
    netlist_directory = "E:\\PHDRepo\\stamps"
    netlist_file = os.path.join(netlist_directory, "circuit.netlist")

    # Parse the netlist
    elements, num_nodes = parse_netlist(netlist_file)

    # Construct the circuit
    G, C, RHS, unknowns = construct_circuit(elements, num_nodes)

    # Frequency sweep setup
    f_min = 1e-3  # Start frequency: 1 mHz
    f_max = 1e9  # End frequency: 1 GHz
    num_points = 1000  # Number of frequency points
    frequencies = np.linspace(f_min, f_max, num_points)

    # Solve the circuit using multiprocessing
    args = [(f, G, C, RHS) for f in frequencies]
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(solve_frequency_wrapper, args), total=num_points, desc="Frequency Sweep"))

    # Initialize accumulators
    total_flops = 0
    total_time = 0
    solutions = []
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)

    # Process results
    for idx, (f, G_sC, X, flops, solve_time) in enumerate(results):
        total_flops += flops
        total_time += solve_time
        solutions.append(X)

        # Save MNA for the first frequency into the PDF for clarity
        if idx == 0:
            pdf.cell(0, 10, txt=f"MNA System for Frequency {f:.2f} Hz:", ln=True)
            pdf.cell(0, 10, txt="Matrix (G + sC):", ln=True)
            formatted_G_sC = format_matrix_for_pdf(G_sC)
            for line in formatted_G_sC.split("\n"):
                pdf.cell(0, 10, txt=line, ln=True)
            pdf.cell(0, 10, txt="RHS (B):", ln=True)
            pdf.cell(0, 10, txt=" ".join(f"{val.real:.2f}+{val.imag:.2f}j" for val in RHS), ln=True)

        # Save selected points to PDF
        if idx % (num_points // 100) == 0:  # Save approximately 100 points
            pdf.cell(0, 10, txt=f"Frequency: {f:.2f} Hz", ln=True)
            pdf.cell(0, 10, txt=f"FLOPs: {flops:.0f}, Solve Time: {solve_time:.6f} seconds", ln=True)
            if X is not None:
                for name, value in zip(unknowns, X):
                    pdf.cell(0, 10, txt=f"{name}: {value.real:.2f} + {value.imag:.2f}j", ln=True)

    # Add summary to PDF
    pdf.cell(0, 10, txt=f"Total FLOPs: {total_flops:.0f}", ln=True)
    pdf.cell(0, 10, txt=f"Total Simulation Time: {total_time:.2f} seconds", ln=True)
    pdf.output("Frequency_Analysis_Summary.pdf")

    print(f"Results saved to Frequency_Analysis_Summary.pdf")
    print(f"Total Simulation Time: {total_time:.2f} seconds")
    print(f"Total FLOPs: {total_flops:.0f}")

    # Plot results for all nodes and save to a PDF
    with PdfPages("Frequency_Response_Plots.pdf") as pdf_pages:
        for i, name in enumerate(unknowns):
            # Extract magnitudes and phases for the current node
            magnitudes = [np.abs(sol[i]) if sol is not None else 0 for sol in solutions]
            phases = [np.angle(sol[i], deg=True) if sol is not None else 0 for sol in solutions]

            # Generate magnitude plot
            plt.figure(figsize=(10, 6))
            plt.semilogx(frequencies, magnitudes, label=f"Magnitude of {name}")
            plt.title(f"Frequency Response (Magnitude) of {name}")
            plt.xlabel("Frequency (Hz) [Log Scale]")
            plt.ylabel("Voltage Magnitude")
            plt.grid(which='both', linestyle='--', linewidth=0.5)
            plt.legend()
            pdf_pages.savefig()
            plt.close()

            # Generate phase plot
            plt.figure(figsize=(10, 6))
            plt.semilogx(frequencies, phases, label=f"Phase of {name}")
            plt.title(f"Frequency Response (Phase) of {name}")
            plt.xlabel("Frequency (Hz) [Log Scale]")
            plt.ylabel("Phase (Degrees)")
            plt.grid(which='both', linestyle='--', linewidth=0.5)
            plt.legend()
            pdf_pages.savefig()
            plt.close()

    print("All plots (magnitude and phase) have been saved to 'Frequency_Response_Plots.pdf'")
