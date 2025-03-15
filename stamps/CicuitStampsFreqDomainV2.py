import numpy as np
from fpdf import FPDF

class CircuitStamps:
    """
    Class for creating conductance, capacitance, and source matrix stamps for nodal analysis in the frequency domain.
    here the extra variables wheicha re added due to addition of extra rwos and colums due to voltage sources are removed in the final output.
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

    @staticmethod
    def capacitance_stamp(matrix, node1, node2, value, s):
        admittance = s * value
        if node1 > 0:
            matrix[node1 - 1, node1 - 1] += admittance
        if node2 > 0:
            matrix[node2 - 1, node2 - 1] += admittance
        if node1 > 0 and node2 > 0:
            matrix[node1 - 1, node2 - 1] -= admittance
            matrix[node2 - 1, node1 - 1] -= admittance

    @staticmethod
    def inductor_stamp(matrix, rhs, node1, node2, inductance, s, current_index):
        size = matrix.shape[0]
        matrix = np.pad(matrix, ((0, 1), (0, 1)), mode='constant', constant_values=0 + 0j)
        rhs = np.pad(rhs, (0, 1), mode='constant', constant_values=0 + 0j)

        if node1 > 0:
            matrix[node1 - 1, current_index] = 1
            matrix[current_index, node1 - 1] = 1
        if node2 > 0:
            matrix[node2 - 1, current_index] = -1
            matrix[current_index, node2 - 1] = -1

        matrix[current_index, current_index] = -1 / (s * inductance)
        return matrix, rhs

    @staticmethod
    def voltage_source_stamp(matrix, rhs, node1, node2, voltage, current_index):
        size = matrix.shape[0]
        matrix = np.pad(matrix, ((0, 1), (0, 1)), mode='constant', constant_values=0 + 0j)
        rhs = np.pad(rhs, (0, 1), mode='constant', constant_values=0 + 0j)

        if node1 > 0:
            matrix[node1 - 1, current_index] = 1
            matrix[current_index, node1 - 1] = 1
        if node2 > 0:
            matrix[node2 - 1, current_index] = -1
            matrix[current_index, node2 - 1] = -1

        rhs[current_index] = voltage
        return matrix, rhs

    @staticmethod
    def current_source_stamp(rhs, node1, node2, current):
        if node1 > 0:
            rhs[node1 - 1] -= current
        if node2 > 0:
            rhs[node2 - 1] += current

def format_matrix(matrix):
    """
    Formats a matrix into a readable string with aligned columns.
    """
    formatted = []
    for row in matrix:
        formatted.append(" ".join(f"{val.real:.2f}+{val.imag:.2f}j" for val in row))
    return "\n".join(formatted)

def solve_mna(G, RHS):
    """
    Solves G * V = RHS for V (node voltages).
    """
    # Check for dimension mismatch
    if G.shape[0] != RHS.shape[0]:
        raise ValueError(f"Dimension mismatch: G is {G.shape}, RHS is {RHS.shape}")
    return np.linalg.solve(G, RHS)

if __name__ == "__main__":
    num_nodes = 3  # Number of nodes
    G = np.zeros((num_nodes, num_nodes), dtype=np.complex128)  # Conductance matrix
    RHS = np.zeros(num_nodes, dtype=np.complex128)  # Right-hand side vector

    # Laplace variable for frequency-domain analysis
    f = 1000  # Frequency in Hz
    omega = 2 * np.pi * f
    s = 1j * omega

    # Add components
    CircuitStamps.conductance_stamp(G, 1, 2, 10)  # 10 ohms between nodes 1 and 2
    CircuitStamps.capacitance_stamp(G, 2, 3, 5e-6, s)  # 5 uF between nodes 2 and 3
    G, RHS = CircuitStamps.inductor_stamp(G, RHS, 3, 0, 1, s, num_nodes)  # 1 H inductor
    G, RHS = CircuitStamps.voltage_source_stamp(G, RHS, 1, 0, 2, num_nodes+1)  # 2V AC voltage source


    # Format MNA for better readability
    formatted_G = format_matrix(G)
    formatted_RHS = " ".join(f"{val.real:.2f}+{val.imag:.2f}j" for val in RHS)

    # Save MNA system to PDF
    output_text = "Conductance Matrix (G):\n" + formatted_G + "\n\nRight-Hand Side Vector (RHS):\n" + formatted_RHS

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for line in output_text.split('\n'):
        pdf.cell(0, 10, txt=line, ln=True)

    pdf.output("MNA_system.pdf")

    print("MNA system saved to MNA_system.pdf")

    # Solve for node voltages
    voltages = solve_mna(G, RHS)

    # Display results (only node voltages)
    print("\nNode Voltages:")
    for i in range(num_nodes):
        print(f"Node {i + 1}: {voltages[i].real:.2f} + {voltages[i].imag:.2f}j V")
