import numpy as np
from fpdf import FPDF

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
    def vccs_stamp(matrix, node1, node2, ctrl_node1, ctrl_node2, gm):
        if node1 > 0 and ctrl_node1 > 0:
            matrix[node1 - 1, ctrl_node1 - 1] += gm
        if node1 > 0 and ctrl_node2 > 0:
            matrix[node1 - 1, ctrl_node2 - 1] -= gm
        if node2 > 0 and ctrl_node1 > 0:
            matrix[node2 - 1, ctrl_node1 - 1] -= gm
        if node2 > 0 and ctrl_node2 > 0:
            matrix[node2 - 1, ctrl_node2 - 1] += gm

    @staticmethod
    def vcvs_stamp(matrix, rhs, node1, node2, ctrl_node1, ctrl_node2, gain, current_index):
        size = matrix.shape[0]
        matrix = np.pad(matrix, ((0, 1), (0, 1)), mode='constant', constant_values=0 + 0j)
        rhs = np.pad(rhs, (0, 1), mode='constant', constant_values=0 + 0j)

        if node1 > 0:
            matrix[node1 - 1, current_index] = 1
            matrix[current_index, node1 - 1] = 1
        if node2 > 0:
            matrix[node2 - 1, current_index] = -1
            matrix[current_index, node2 - 1] = -1
        if ctrl_node1 > 0:
            matrix[current_index, ctrl_node1 - 1] -= gain
        if ctrl_node2 > 0:
            matrix[current_index, ctrl_node2 - 1] += gain

        return matrix, rhs

    @staticmethod
    def cccs_stamp(matrix, ctrl_index, node1, node2, gain):
        if node1 > 0:
            matrix[node1 - 1, ctrl_index] += gain
        if node2 > 0:
            matrix[node2 - 1, ctrl_index] -= gain

    @staticmethod
    def ccvs_stamp(matrix, rhs, node1, node2, ctrl_index, gain, current_index):
        size = matrix.shape[0]
        matrix = np.pad(matrix, ((0, 1), (0, 1)), mode='constant', constant_values=0 + 0j)
        rhs = np.pad(rhs, (0, 1), mode='constant', constant_values=0 + 0j)

        if node1 > 0:
            matrix[node1 - 1, current_index] = 1
            matrix[current_index, node1 - 1] = 1
        if node2 > 0:
            matrix[node2 - 1, current_index] = -1
            matrix[current_index, node2 - 1] = -1

        matrix[current_index, ctrl_index] -= gain
        return matrix, rhs

    @staticmethod
    def current_source_stamp(rhs, node1, node2, current):
        """
        Add a current source stamp to the right-hand side vector.
        """
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

if __name__ == "__main__":
    # Example usage
    size = 4  # Number of nodes
    G = np.zeros((size, size), dtype=np.complex128)  # Conductance matrix (complex)
    RHS = np.zeros(size + 1, dtype=np.complex128)  # Right-hand side vector (complex)

    # Laplace variable for frequency-domain analysis
    s = 1j * 2 * np.pi * 1000  # Example frequency: 1 kHz

    # Add components
    CircuitStamps.conductance_stamp(G, 1, 2, 10)  # 10 ohms between node 1 and 2
    CircuitStamps.capacitance_stamp(G, 2, 3, 5e-6, s)  # 5 uF between node 2 and 3
    CircuitStamps.current_source_stamp(RHS, 1, 0, 0.01)  # 10 mA current source
    G, RHS = CircuitStamps.inductor_stamp(G, RHS, 3, 4, 1, s, size)  # 1 H inductor
    G, RHS = CircuitStamps.voltage_source_stamp(G, RHS, 1, 2, 5, size + 1)  # 5 V voltage source

    # Format output for better readability
    formatted_G = format_matrix(G)
    formatted_RHS = " ".join(f"{val.real:.2f}+{val.imag:.2f}j" for val in RHS)

    # Save formatted output to PDF
    output_text = "Conductance Matrix (G):\n" + formatted_G + "\n\nRight-Hand Side Vector (RHS):\n" + formatted_RHS

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for line in output_text.split('\n'):
        pdf.cell(0, 10, txt=line, ln=True)

    pdf.output("output.pdf")

    print("Conductance Matrix (G):")
    print(formatted_G)

    print("\nRight-Hand Side Vector (RHS):")
    print(formatted_RHS)
    print("\nOutput saved to output.pdf")
