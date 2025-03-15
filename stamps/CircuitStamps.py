class CircuitStamps:
    """
    Class for creating conductance, capacitance, and source matrix stamps for nodal analysis.
    """

    @staticmethod
    def conductance_stamp(matrix, node1, node2, value):
        """
        Add a conductance stamp to the nodal analysis matrix.

        Parameters:
        matrix (list of list): The conductance matrix to update.
        node1 (int): The first node of the conductance.
        node2 (int): The second node of the conductance.
        value (float): Conductance value (G = 1/R).
        """
        if node1 > 0:
            matrix[node1 - 1][node1 - 1] += value
        if node2 > 0:
            matrix[node2 - 1][node2 - 1] += value
        if node1 > 0 and node2 > 0:
            matrix[node1 - 1][node2 - 1] -= value
            matrix[node2 - 1][node1 - 1] -= value

    @staticmethod
    def capacitance_stamp(matrix, node1, node2, value):
        """
        Add a capacitance stamp to the nodal analysis matrix.

        Parameters:
        matrix (list of list): The capacitance matrix to update.
        node1 (int): The first node of the capacitance.
        node2 (int): The second node of the capacitance.
        value (float): Capacitance value.
        """
        if node1 > 0:
            matrix[node1 - 1][node1 - 1] += value
        if node2 > 0:
            matrix[node2 - 1][node2 - 1] += value
        if node1 > 0 and node2 > 0:
            matrix[node1 - 1][node2 - 1] -= value
            matrix[node2 - 1][node1 - 1] -= value

    @staticmethod
    def voltage_source_stamp(matrix, rhs, node1, node2, voltage, current_index):
        """
        Add a voltage source stamp to the nodal analysis matrix.

        Parameters:
        matrix (list of list): The nodal matrix to update.
        rhs (list): The right-hand side vector to update.
        node1 (int): The positive node of the voltage source.
        node2 (int): The negative node of the voltage source.
        voltage (float): Voltage value.
        current_index (int): Index for the current variable of the voltage source.
        """
        size = len(matrix)
        matrix.append([0] * (size + 1))
        for row in matrix:
            row.append(0)

        if node1 > 0:
            matrix[node1 - 1][current_index] = 1
            matrix[current_index][node1 - 1] = 1
        if node2 > 0:
            matrix[node2 - 1][current_index] = -1
            matrix[current_index][node2 - 1] = -1

        rhs.append(0)
        rhs[current_index] = voltage

    @staticmethod
    def current_source_stamp(rhs, node1, node2, current):
        """
        Add a current source stamp to the right-hand side vector.

        Parameters:
        rhs (list): The right-hand side vector to update.
        node1 (int): The positive node of the current source.
        node2 (int): The negative node of the current source.
        current (float): Current value.
        """
        if node1 > 0:
            rhs[node1 - 1] -= current
        if node2 > 0:
            rhs[node2 - 1] += current

    @staticmethod
    def vccs_stamp(matrix, node1, node2, ctrl_node1, ctrl_node2, transconductance):
        """
        Add a voltage-controlled current source (VCCS) stamp to the nodal analysis matrix.

        Parameters:
        matrix (list of list): The nodal matrix to update.
        node1 (int): The positive output node.
        node2 (int): The negative output node.
        ctrl_node1 (int): The positive control node.
        ctrl_node2 (int): The negative control node.
        transconductance (float): Transconductance value (Gm).
        """
        if node1 > 0 and ctrl_node1 > 0:
            matrix[node1 - 1][ctrl_node1 - 1] += transconductance
        if node1 > 0 and ctrl_node2 > 0:
            matrix[node1 - 1][ctrl_node2 - 1] -= transconductance
        if node2 > 0 and ctrl_node1 > 0:
            matrix[node2 - 1][ctrl_node1 - 1] -= transconductance
        if node2 > 0 and ctrl_node2 > 0:
            matrix[node2 - 1][ctrl_node2 - 1] += transconductance

    @staticmethod
    def vcvs_stamp(matrix, rhs, node1, node2, ctrl_node1, ctrl_node2, gain, current_index):
        """
        Add a voltage-controlled voltage source (VCVS) stamp to the nodal analysis matrix.

        Parameters:
        matrix (list of list): The nodal matrix to update.
        rhs (list): The right-hand side vector to update.
        node1 (int): The positive output node.
        node2 (int): The negative output node.
        ctrl_node1 (int): The positive control node.
        ctrl_node2 (int): The negative control node.
        gain (float): Voltage gain.
        current_index (int): Index for the current variable of the VCVS.
        """
        size = len(matrix)
        matrix.append([0] * (size + 1))
        for row in matrix:
            row.append(0)

        if node1 > 0:
            matrix[node1 - 1][current_index] = 1
            matrix[current_index][node1 - 1] = 1
        if node2 > 0:
            matrix[node2 - 1][current_index] = -1
            matrix[current_index][node2 - 1] = -1
        if ctrl_node1 > 0:
            matrix[current_index][ctrl_node1 - 1] -= gain
        if ctrl_node2 > 0:
            matrix[current_index][ctrl_node2 - 1] += gain

        rhs.append(0)

    @staticmethod
    def ccvs_stamp(matrix, rhs, node1, node2, ctrl_index, gain, current_index):
        """
        Add a current-controlled voltage source (CCVS) stamp to the nodal analysis matrix.

        Parameters:
        matrix (list of list): The nodal matrix to update.
        rhs (list): The right-hand side vector to update.
        node1 (int): The positive output node.
        node2 (int): The negative output node.
        ctrl_index (int): The index of the controlling current.
        gain (float): Current gain.
        current_index (int): Index for the current variable of the CCVS.
        """
        size = len(matrix)
        matrix.append([0] * (size + 1))
        for row in matrix:
            row.append(0)

        if node1 > 0:
            matrix[node1 - 1][current_index] = 1
            matrix[current_index][node1 - 1] = 1
        if node2 > 0:
            matrix[node2 - 1][current_index] = -1
            matrix[current_index][node2 - 1] = -1

        matrix[current_index][ctrl_index] -= gain
        rhs.append(0)

    @staticmethod
    def cccs_stamp(matrix, ctrl_index, node1, node2, gain):
        """
        Add a current-controlled current source (CCCS) stamp to the nodal analysis matrix.

        Parameters:
        matrix (list of list): The nodal matrix to update.
        ctrl_index (int): The index of the controlling current.
        node1 (int): The positive output node.
        node2 (int): The negative output node.
        gain (float): Current gain.
        """
        if node1 > 0:
            matrix[node1 - 1][ctrl_index] += gain
        if node2 > 0:
            matrix[node2 - 1][ctrl_index] -= gain

    @staticmethod
    def inductor_stamp(matrix, rhs, node1, node2, inductance, current_index):
        """
        Add an inductor stamp to the nodal analysis matrix.

        Parameters:
        matrix (numpy.ndarray): The nodal matrix to update.
        rhs (numpy.ndarray): The right-hand side vector to update.
        node1 (int): The positive node of the inductor.
        node2 (int): The negative node of the inductor.
        inductance (float): Inductance value (L).
        current_index (int): Index for the current variable of the inductor.
        """
        size = matrix.shape[0]
        matrix = np.pad(matrix, ((0, 1), (0, 1)), mode='constant')  # Add a new row and column
        rhs = np.pad(rhs, (0, 1), mode='constant')  # Add a new entry to the RHS

        if node1 > 0:
            matrix[node1 - 1, current_index] = 1
            matrix[current_index, node1 - 1] = 1
        if node2 > 0:
            matrix[node2 - 1, current_index] = -1
            matrix[current_index, node2 - 1] = -1

        matrix[current_index, current_index] = -1 / inductance

        return matrix, rhs


if __name__ == "__main__":
    import numpy as np

    # Example usage
    size = 4  # Number of nodes
    G = np.zeros((size, size))  # Conductance matrix
    C = np.zeros((size, size))  # Capacitance matrix
    RHS = np.zeros(size + 1)  # Right-hand side vector (extended for voltage/inductor currents)

    # Add components
    CircuitStamps.conductance_stamp(G, 1, 2, 10)  # 10 ohms between node 1 and 2
    CircuitStamps.capacitance_stamp(C, 2, 3, 5e-1)  # 5 uF between node 2 and 3
    CircuitStamps.current_source_stamp(RHS, 1, 0, 0.01)  # 10 mA current source from node 1 to ground

    # Add an inductor between nodes 3 and 4 (1 H inductance)
    G, RHS = CircuitStamps.inductor_stamp(G, RHS, 3, 4, 1, size)

    print("Conductance Matrix (G):")
    print(G)

    print("\nCapacitance Matrix (C):")
    print(C)

    print("\nRight-Hand Side Vector (RHS):")
    print(RHS)

    # Additional debug printouts to check the impact of inductor
    print("\nMatrix Size after adding inductor:")
    print(f"Conductance Matrix dimensions: {G.shape}")
    print(f"RHS Vector length: {len(RHS)}")
