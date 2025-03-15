import numpy as np
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

###############################################################################
#                           STAMPING UTILITIES
###############################################################################
class CircuitStamps:
    @staticmethod
    def expand_system_size(G, C, L, RHS):
        """
        Expand G, C, L from (n x n) to (n+1 x n+1),
        and RHS from length n to (n+1).

        Returns:
          G_expanded, C_expanded, L_expanded, RHS_expanded, new_idx
        """
        n_old = G.shape[0]
        G_expanded = np.pad(G, ((0,1),(0,1)), mode='constant')
        C_expanded = np.pad(C, ((0,1),(0,1)), mode='constant')
        L_expanded = np.pad(L, ((0,1),(0,1)), mode='constant')
        RHS_expanded = np.pad(RHS, (0,1), mode='constant')
        return G_expanded, C_expanded, L_expanded, RHS_expanded, n_old

    @staticmethod
    def conductance_stamp(G, node1, node2, value):
        """
        Resistor stamp (or any linear conductance).
        G[node1-1,node1-1] += value, etc.
        """
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
        """
        Capacitor => +s*C in the final MNA matrix.
        We'll just store 'value' here; multiply by s at solve-time.
        """
        if node1 > 0:
            C[node1 - 1, node1 - 1] += value
        if node2 > 0:
            C[node2 - 1, node2 - 1] += value
        if node1 > 0 and node2 > 0:
            C[node1 - 1, node2 - 1] -= value
            C[node2 - 1, node1 - 1] -= value
        return C

    @staticmethod
    def voltage_source_stamp(G, C, L, RHS, node1, node2, voltage):
        """
        Classical MNA: add new current variable i_VS.
        v(node1) - v(node2) = voltage.
        """
        G_expanded, C_expanded, L_expanded, RHS_expanded, idx = \
            CircuitStamps.expand_system_size(G, C, L, RHS)

        # KCL for node1 -> +i_VS, node2 -> -i_VS
        if node1 > 0:
            G_expanded[node1 - 1, idx] += 1
            G_expanded[idx, node1 - 1] += 1
        if node2 > 0:
            G_expanded[node2 - 1, idx] -= 1
            G_expanded[idx, node2 - 1] -= 1

        # voltage source equation => RHS_expanded[idx] = voltage
        RHS_expanded[idx] = voltage

        return G_expanded, C_expanded, L_expanded, RHS_expanded, idx

    @staticmethod
    def current_source_stamp(RHS, node1, node2, value):
        """
        Independent current source: +value into node1, -value from node2.
        """
        if node1 > 0:
            RHS[node1 - 1] += value
        if node2 > 0:
            RHS[node2 - 1] -= value
        return RHS

    @staticmethod
    def inductor_stamp(G, C, L, RHS, node1, node2, inductance):
        """
        Classical MNA inductor: v(node1)-v(node2) - sL*i_L = 0
        We'll store '-inductance' in L so that final = + s * (-L) = -sL.
        """
        G_expanded, C_expanded, L_expanded, RHS_expanded, idx = \
            CircuitStamps.expand_system_size(G, C, L, RHS)

        # KCL => +i_L at node1, -i_L at node2
        if node1 > 0:
            G_expanded[node1 - 1, idx] += 1
            G_expanded[idx, node1 - 1] += 1
        if node2 > 0:
            G_expanded[node2 - 1, idx] -= 1
            G_expanded[idx, node2 - 1] -= 1

        # Inductor eq => store -L on the diagonal
        L_expanded[idx, idx] = -inductance

        return G_expanded, C_expanded, L_expanded, RHS_expanded, idx

    ############################################################################
    # Dependent Sources
    ############################################################################
    @staticmethod
    def vccs_stamp(G, node_out_p, node_out_m, node_in_p, node_in_m, gm):
        """
        Voltage-Controlled Current Source:
          i_out = gm * (v_in_p - v_in_m)

        The current flows into node_out_p and out of node_out_m.
        So stamp:
          G[node_out_p, node_in_p] += gm
          G[node_out_p, node_in_m] -= gm
          G[node_out_m, node_in_p] -= gm
          G[node_out_m, node_in_m] += gm
        (Indices must be adjusted by -1 if node>0).
        """
        # If any of these nodes is 0 => ground => skip or reduce stamp accordingly
        p_out = node_out_p - 1 if node_out_p > 0 else None
        m_out = node_out_m - 1 if node_out_m > 0 else None
        p_in  = node_in_p  - 1 if node_in_p  > 0 else None
        m_in  = node_in_m  - 1 if node_in_m  > 0 else None

        if p_out is not None and p_in is not None:
            G[p_out, p_in] += gm
        if p_out is not None and m_in is not None:
            G[p_out, m_in] -= gm
        if m_out is not None and p_in is not None:
            G[m_out, p_in] -= gm
        if m_out is not None and m_in is not None:
            G[m_out, m_in] += gm

        return G

    @staticmethod
    def vcvs_stamp(G, C, L, RHS, node_out_p, node_out_m, node_in_p, node_in_m, gain):
        """
        Voltage-Controlled Voltage Source:
          v_out = gain * (v_in_p - v_in_m)

        This is like a voltage source whose value depends on a node voltage difference.
        In classical MNA, we add a new current unknown i_VCVS, new row/col.
        The voltage source eqn => v_out_p - v_out_m - gain*(v_in_p - v_in_m) = 0
        """
        G_expanded, C_expanded, L_expanded, RHS_expanded, idx = \
            CircuitStamps.expand_system_size(G, C, L, RHS)

        # KCL for i_VCVS at node_out_p (flow in), node_out_m (flow out)
        if node_out_p > 0:
            G_expanded[node_out_p - 1, idx] += 1
            G_expanded[idx, node_out_p - 1] += 1
        if node_out_m > 0:
            G_expanded[node_out_m - 1, idx] -= 1
            G_expanded[idx, node_out_m - 1] -= 1

        # Now we must stamp the controlling voltage in the same row:
        #   Vout_p - Vout_m - gain*(Vin_p - Vin_m) = 0
        # That means:
        #   +1 at [idx, node_in_p], -1*g at [idx, node_in_m], etc.
        p_in = node_in_p - 1 if node_in_p>0 else None
        m_in = node_in_m - 1 if node_in_m>0 else None

        if p_in is not None:
            G_expanded[idx, p_in] -= gain
        if m_in is not None:
            G_expanded[idx, m_in] += gain

        return G_expanded, C_expanded, L_expanded, RHS_expanded, idx

    @staticmethod
    def cccs_stamp(G, node_out_p, node_out_m, branch_idx_in, gain):
        """
        Current-Controlled Current Source:
          i_out = gain * i_in
        'i_in' is the current of some other branch (voltage source or inductor),
        which is tracked in the MNA as an unknown with index 'branch_idx_in'.

        The output current flows into node_out_p, out node_out_m:
          i_out = + gain * i_in => G[node_out_p, branch_idx_in] += gain
          i_out = - gain * i_in => G[node_out_m, branch_idx_in] -= gain
        """
        p_out = node_out_p - 1 if node_out_p > 0 else None
        m_out = node_out_m - 1 if node_out_m > 0 else None

        if p_out is not None:
            G[p_out, branch_idx_in] += gain
        if m_out is not None:
            G[m_out, branch_idx_in] -= gain

        return G

    @staticmethod
    def ccvs_stamp(G, C, L, RHS, node_out_p, node_out_m, branch_idx_in, transresistance):
        """
        Current-Controlled Voltage Source:
          v_out = transresistance * i_in
        We add a new branch current unknown i_CCVS.
        The voltage eqn => (v_out_p - v_out_m) - transresistance*i_in = 0
        """
        G_expanded, C_expanded, L_expanded, RHS_expanded, idx_new = \
            CircuitStamps.expand_system_size(G, C, L, RHS)

        # KCL for i_CCVS
        if node_out_p > 0:
            G_expanded[node_out_p - 1, idx_new] += 1
            G_expanded[idx_new, node_out_p - 1] += 1
        if node_out_m > 0:
            G_expanded[node_out_m - 1, idx_new] -= 1
            G_expanded[idx_new, node_out_m - 1] -= 1

        # Now we also stamp -transresistance on [idx_new, branch_idx_in]
        # (since v_out = + transresistance * i_in => MNA eqn => v_out - transresistance*i_in=0)
        G_expanded[idx_new, branch_idx_in] -= transresistance

        return G_expanded, C_expanded, L_expanded, RHS_expanded, idx_new

###############################################################################
#                     SOLVER AND FREQUENCY SWEEP CODE
###############################################################################
def solve_mna(M, RHS):
    """ Solve M x = RHS """
    return np.linalg.solve(M, RHS)

def solve_frequency_wrapper(args):
    """
    Helper function for parallel processing.
    Rebuild or update the matrix with s = j*2πf, then solve for x.
    Also compute approximate FLOPs for an n×n solve.
    """
    f, base_G, base_C, base_L, base_RHS = args

    omega = 2 * np.pi * f
    s = 1j * omega

    # Build final M = G + s*C + s*L
    # (Here we assume base_G, base_C, base_L are all same dimension)
    M = base_G + s*base_C + s*base_L

    # measure dimension
    n = M.shape[0]
    # approximate FLOPs for LU ~ (2/3)*n^3 + 2*n^2
    flops = (2/3)*n**3 + 2*n**2

    # Solve
    x = solve_mna(M, base_RHS)

    return f, x, flops

###############################################################################
#                                 MAIN
###############################################################################
if __name__ == "__main__":
    import math

    # ------------------------------
    # 1) Define base circuit size
    # ------------------------------
    num_nodes = 2
    # Start with just node voltages => 2x2
    G = np.zeros((num_nodes, num_nodes), dtype=complex)
    C = np.zeros((num_nodes, num_nodes), dtype=complex)
    L = np.zeros((num_nodes, num_nodes), dtype=complex)
    RHS = np.zeros(num_nodes, dtype=complex)

    # Track unknown names
    unknowns = ["V1", "V2"]

    # -------------------------------------------------------
    # 2) Example Stamping of Some Elements
    # -------------------------------------------------------
    # Resistor 10Ω between node1 and node2 => Conductance=1/10=0.1
    G = CircuitStamps.conductance_stamp(G, 1, 2, 0.1)

    # Capacitor 1e-6 F between node2 and ground
    C = CircuitStamps.capacitance_stamp(C, 2, 0, 1e-6)

    # Voltage source 5 V from node1 to ground => new i_VS
    G, C, L, RHS, idx_vs = CircuitStamps.voltage_source_stamp(G, C, L, RHS, 1, 0, 5)
    unknowns.append("I_VS")

    # Current source 0.01 A from node2 to ground
    RHS = CircuitStamps.current_source_stamp(RHS, 2, 0, 0.01)

    # Inductor 1e-3 H from node1 to ground => new i_L
    G, C, L, RHS, idx_l = CircuitStamps.inductor_stamp(G, C, L, RHS, 1, 0, 1e-3)
    unknowns.append("I_L")

    # Example usage of a dependent source (VCCS):
    # Suppose a transconductance gm=2 S, controlling voltage = (node1 - ground),
    # output current into node2, out of ground. That’s basically i2=2*(V1 - V0).
    # node_out_p=2, node_out_m=0, node_in_p=1, node_in_m=0
    G = CircuitStamps.vccs_stamp(G, 2, 0, 1, 0, 2.0)

    # Now G, C, L, RHS are bigger. Let’s see how big:
    n_size = G.shape[0]
    print(f"Final MNA system dimension: {n_size}x{n_size}")
    print("Unknowns:", unknowns)

    # ------------------------------
    # 3) Frequency Sweep Setup
    # ------------------------------
    # Sweep from 1 Hz to 1 GHz in 10k log-spaced points
    num_points = 10000
    freqs = np.logspace(np.log10(1), np.log10(1e9), num_points)

    # We'll store solutions in an array: x_solutions[freq_idx, var_idx]
    x_solutions = np.zeros((num_points, n_size), dtype=complex)

    # For performance, we'll create a list of arguments to pass to each parallel job
    # We'll keep the base G,C,L,RHS. Each frequency is solved separately.
    parallel_args = [(f, G, C, L, RHS) for f in freqs]

    # ------------------------------
    # 4) Solve in Parallel + Timing
    # ------------------------------
    start_time = time.time()

    # We'll accumulate total flops
    total_flops = 0

    # Use multiprocessing for the frequency sweep
    pool_size = min(cpu_count(), 8)  # for example, limit to 8
    with Pool(processes=pool_size) as pool:
        results_iter = pool.imap(solve_frequency_wrapper, parallel_args)
        # Wrap in tqdm for progress bar
        for i, (f_cur, x_cur, flops_cur) in enumerate(tqdm(results_iter,
                                                           total=num_points,
                                                           desc="Frequency Sweep")):
            x_solutions[i, :] = x_cur
            total_flops += flops_cur

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nSweep completed in {elapsed_time:.2f} seconds.")
    print(f"Approx. total FLOPs = {total_flops:.2e}")

    # ------------------------------
    # 5) Post-Processing:
    #    Magnitude & Phase of each node
    # ------------------------------
    # Remember unknowns: [V1, V2, I_VS, I_L, ...]
    # Typically we only plot node voltages (the first 'num_nodes' entries).
    node_indices = range(num_nodes)  # 0..(num_nodes-1)
    magnitudes = np.abs(x_solutions[:, node_indices])
    phases_deg = np.angle(x_solutions[:, node_indices], deg=True)

    # ------------------------------
    # 6) Plot and Save to PDF
    # ------------------------------
    output_pdf = "simulation_results.pdf"
    with PdfPages(output_pdf) as pdf:
        # Plot magnitude for each node
        plt.figure(figsize=(7,5))
        for i_node in node_indices:
            plt.semilogx(freqs, magnitudes[:, i_node], label=f"{unknowns[i_node]} magnitude")
        plt.title("Node Voltage Magnitudes")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude [V]")
        plt.legend()
        plt.grid(True, which="both")
        pdf.savefig()
        plt.close()

        # Plot phase for each node
        plt.figure(figsize=(7,5))
        for i_node in node_indices:
            plt.semilogx(freqs, phases_deg[:, i_node], label=f"{unknowns[i_node]} phase")
        plt.title("Node Voltage Phases")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Phase [degrees]")
        plt.legend()
        plt.grid(True, which="both")
        pdf.savefig()
        plt.close()

    # Optionally, you might also save tabular data or additional plots
    print(f"\nAll results saved to {output_pdf}")
