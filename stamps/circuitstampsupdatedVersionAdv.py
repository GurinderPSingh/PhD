import numpy as np
import time
import math
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

        G_expanded = np.pad(G, ((0,1),(0,1)), mode='constant', constant_values=0)
        C_expanded = np.pad(C, ((0,1),(0,1)), mode='constant', constant_values=0)
        L_expanded = np.pad(L, ((0,1),(0,1)), mode='constant', constant_values=0)
        RHS_expanded = np.pad(RHS, (0,1), mode='constant', constant_values=0)

        return G_expanded, C_expanded, L_expanded, RHS_expanded, n_old

    @staticmethod
    def conductance_stamp(G, node1, node2, value):
        """
        Resistor or linear conductance. value = 1/R.
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
        Capacitor => s*C in the final M = G + sC + sL.
        Here we just store 'value' in C.
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
        Ideal voltage source => new current unknown i_VS.
        v(node1) - v(node2) = voltage
        """
        Gx, Cx, Lx, RHSx, idx = CircuitStamps.expand_system_size(G, C, L, RHS)

        # KCL => +i_VS at node1, -i_VS at node2
        if node1 > 0:
            Gx[node1 - 1, idx] += 1
            Gx[idx, node1 - 1] += 1
        if node2 > 0:
            Gx[node2 - 1, idx] -= 1
            Gx[idx, node2 - 1] -= 1

        # The known voltage in RHS
        RHSx[idx] = voltage

        return Gx, Cx, Lx, RHSx, idx

    @staticmethod
    def current_source_stamp(RHS, node1, node2, value):
        """
        Independent current source: +value into node1, -value out node2.
        """
        if node1 > 0:
            RHS[node1 - 1] += value
        if node2 > 0:
            RHS[node2 - 1] -= value
        return RHS

    @staticmethod
    def inductor_stamp(G, C, L, RHS, node1, node2, inductance):
        """
        Classical MNA inductor:
        v(node1) - v(node2) - sL*i_L = 0
        We store -L on L[idx, idx], so in final M = G + sC + sL => s*(-L) = -sL
        => v(node1)-v(node2)-sL*i_L=0
        """
        Gx, Cx, Lx, RHSx, idx = CircuitStamps.expand_system_size(G, C, L, RHS)

        # KCL => +i_L at node1, -i_L at node2
        if node1 > 0:
            Gx[node1 - 1, idx] += 1
            Gx[idx, node1 - 1] += 1
        if node2 > 0:
            Gx[node2 - 1, idx] -= 1
            Gx[idx, node2 - 1] -= 1

        # Place -L on diagonal
        Lx[idx, idx] = -inductance

        return Gx, Cx, Lx, RHSx, idx

    # ------------------------- DEPENDENT SOURCES -------------------------
    @staticmethod
    def vccs_stamp(G, node_out_p, node_out_m, node_in_p, node_in_m, gm):
        """
        Voltage-Controlled Current Source (VCCS):
          i_out = gm * [v(node_in_p) - v(node_in_m)]
        Stamped directly in G like conductances:
          +gm => (out_p, in_p)
          -gm => (out_p, in_m)
          -gm => (out_m, in_p)
          +gm => (out_m, in_m)
        """
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
    def vcvs_stamp(G, C, L, RHS,
                   node_out_p, node_out_m,
                   node_in_p, node_in_m,
                   gain):
        """
        Voltage-Controlled Voltage Source (VCVS):
          v_out = gain * (v_in_p - v_in_m)
        => new branch current unknown i_VCVS
        => eqn: v(node_out_p) - v(node_out_m) - gain*(v(node_in_p) - v(node_in_m))=0
        """
        Gx, Cx, Lx, RHSx, idx = CircuitStamps.expand_system_size(G, C, L, RHS)

        # KCL for i_VCVS at out_p and out_m
        if node_out_p > 0:
            Gx[node_out_p - 1, idx] += 1
            Gx[idx, node_out_p - 1] += 1
        if node_out_m > 0:
            Gx[node_out_m - 1, idx] -= 1
            Gx[idx, node_out_m - 1] -= 1

        # The controlling voltage => -gain*(v_in_p - v_in_m)
        p_in = node_in_p - 1 if node_in_p>0 else None
        m_in = node_in_m - 1 if node_in_m>0 else None

        if p_in is not None:
            Gx[idx, p_in] -= gain
        if m_in is not None:
            Gx[idx, m_in] += gain

        return Gx, Cx, Lx, RHSx, idx

    @staticmethod
    def cccs_stamp(G, node_out_p, node_out_m, branch_idx_in, gain):
        """
        Current-Controlled Current Source (CCCS):
          i_out = gain * i_in
        i_in is the current of some existing branch (voltage source or inductor),
        tracked at 'branch_idx_in'.
        """
        p_out = node_out_p - 1 if node_out_p>0 else None
        m_out = node_out_m - 1 if node_out_m>0 else None

        # i_out flows into node_out_p, out of node_out_m
        # => G[p_out, branch_idx_in] += gain
        # => G[m_out, branch_idx_in] -= gain
        if p_out is not None:
            G[p_out, branch_idx_in] += gain
        if m_out is not None:
            G[m_out, branch_idx_in] -= gain

        return G

    @staticmethod
    def ccvs_stamp(G, C, L, RHS, node_out_p, node_out_m, branch_idx_in, transres):
        """
        Current-Controlled Voltage Source (CCVS):
          v_out = transres * i_in
        => new branch unknown i_CCVS
        => eqn: v(node_out_p) - v(node_out_m) - transres*i_in = 0
        """
        Gx, Cx, Lx, RHSx, idx_new = CircuitStamps.expand_system_size(G, C, L, RHS)

        # KCL => i_CCVS in node_out_p, out node_out_m
        if node_out_p > 0:
            Gx[node_out_p - 1, idx_new] += 1
            Gx[idx_new, node_out_p - 1] += 1
        if node_out_m > 0:
            Gx[node_out_m - 1, idx_new] -= 1
            Gx[idx_new, node_out_m - 1] -= 1

        # eqn => v_out - transres*i_in=0 => G[idx_new, branch_idx_in] -= transres
        Gx[idx_new, branch_idx_in] -= transres

        return Gx, Cx, Lx, RHSx, idx_new

###############################################################################
#                          SOLVER and PARALLEL CODE
###############################################################################
def solve_mna(M, RHS):
    """ Solve M x = RHS """
    return np.linalg.solve(M, RHS)

def solve_frequency_wrapper(args):
    """
    For parallel computing:
    args => (freq, G, C, L, RHS)
    Build M = G + s*C + s*L, solve, compute FLOPs, return (freq, x, flops).
    """
    f, base_G, base_C, base_L, base_RHS = args

    omega = 2 * np.pi * f
    s = 1j * omega

    M = base_G + s*base_C + s*base_L

    # dimension
    n = M.shape[0]
    # approximate flops for an n×n LU => (2/3)*n^3 + 2*n^2
    flops = (2.0/3.0)*n**3 + 2.0*(n**2)

    x = solve_mna(M, base_RHS)
    return f, x, flops

def matrix_to_string(mat, name):
    """Convert a numpy matrix to a string for debug printing."""
    return f"{name} =\n{np.array_str(mat, precision=4, suppress_small=True)}\n"

def vector_to_string(vec, name):
    """Convert a numpy vector to a string for debug printing."""
    return f"{name} = {np.array_str(vec, precision=4, suppress_small=True)}\n"

###############################################################################
#                                    MAIN
###############################################################################
if __name__ == "__main__":
    # For demonstration, we'll use a 5-node circuit plus ground (node0).
    # We'll add a variety of elements to test all stamps.

    # 1) Initialize base size for 5 nodes => G, C, L are 5x5 for node voltages
    num_nodes = 5
    G = np.zeros((num_nodes, num_nodes), dtype=complex)
    C = np.zeros((num_nodes, num_nodes), dtype=complex)
    L = np.zeros((num_nodes, num_nodes), dtype=complex)

    # RHS is length 5
    RHS = np.zeros(num_nodes, dtype=complex)

    # We'll keep a list of unknown names for debugging: node voltages first
    unknowns = [f"V{i}" for i in range(1, num_nodes+1)]

    # -------------------------------------------------------------------------
    # 2) Stamp Circuit Elements
    #    We'll demonstrate each stamp at least once.
    # -------------------------------------------------------------------------

    # a) Resistors
    #    R=100Ω between node1, node2 => G=1/100=0.01
    G = CircuitStamps.conductance_stamp(G, 1, 2, 0.01)

    #    R=200Ω between node2, node3 => G=1/200=0.005
    G = CircuitStamps.conductance_stamp(G, 2, 3, 0.005)

    # b) Capacitors
    #    C=1e-6 F between node1, ground
    C = CircuitStamps.capacitance_stamp(C, 1, 0, 1e-6)

    #    C=2e-6 F between node3, node5
    C = CircuitStamps.capacitance_stamp(C, 3, 5, 2e-6)

    # c) Inductor
    #    L=1e-3 H between node2, ground => new unknown i_L
    G, C, L, RHS, idx_l = CircuitStamps.inductor_stamp(G, C, L, RHS, 2, 0, 1e-3)
    unknowns.append("I_L1")  # inductor current

    # d) Voltage Source
    #    Vs=10V from node4 to ground => new unknown i_VS
    G, C, L, RHS, idx_vs = CircuitStamps.voltage_source_stamp(G, C, L, RHS, 4, 0, 10)
    unknowns.append("I_VS1")

    # e) Current Source
    #    Is=0.02 A from node5 to node3
    RHS = CircuitStamps.current_source_stamp(RHS, 5, 3, 0.02)

    # f) Dependent Sources

    # 1) VCCS => gm=0.1 S, controlling voltage = (node1 - node3), output = node2->node4
    G = CircuitStamps.vccs_stamp(G, node_out_p=2, node_out_m=4,
                                 node_in_p=1, node_in_m=3,
                                 gm=0.1)

    # 2) VCVS => gain=2, out=(node1,node5), in=(node2,node4)
    #    => new unknown i_VCVS
    G, C, L, RHS, idx_vcvs = CircuitStamps.vcvs_stamp(G, C, L, RHS,
                                                      node_out_p=1, node_out_m=5,
                                                      node_in_p=2, node_in_m=4,
                                                      gain=2.0)
    unknowns.append("I_VCVS")

    # 3) CCCS => i_out = 1.5 * i_L1 (the inductor’s current)
    #    Output flows from node3->node4
    branch_idx_in = idx_l  # controlling current = inductor current
    G = CircuitStamps.cccs_stamp(G, node_out_p=3, node_out_m=4,
                                 branch_idx_in=branch_idx_in,
                                 gain=1.5)

    # 4) CCVS => v_out=0.5*i_VCVS, output node2->node5
    #    => new unknown i_CCVS
    branch_idx_in_vcvs = idx_vcvs
    G, C, L, RHS, idx_ccvs = CircuitStamps.ccvs_stamp(G, C, L, RHS,
                                                      node_out_p=2, node_out_m=5,
                                                      branch_idx_in=branch_idx_in_vcvs,
                                                      transres=0.5)
    unknowns.append("I_CCVS")

    # -------------------------------------------------------------------------
    # Check final matrix dimension
    n_size = G.shape[0]
    print(f"Final MNA dimension: {n_size}x{n_size}")
    print("Unknowns:", unknowns)

    # -------------------------------------------------------------------------
    # 3) Debug-Print Matrices and One Example (G + sC + sL) at f_debug=1kHz
    # -------------------------------------------------------------------------
    f_debug = 1e3
    omega_debug = 2*math.pi*f_debug
    s_debug = 1j*omega_debug
    M_debug = G + s_debug*C + s_debug*L

    print("\n--- Debug Print ---")
    print(matrix_to_string(G, "G"))
    print(matrix_to_string(C, "C"))
    print(matrix_to_string(L, "L  (inductors store -L on diag)"))
    print(vector_to_string(RHS, "RHS"))
    print(f"Unknowns = {unknowns}")
    print(f"\nM_debug at f={f_debug} Hz => (G + sC + sL):\n{np.array_str(M_debug, precision=4, suppress_small=True)}\n")

    # -------------------------------------------------------------------------
    # 4) Frequency Sweep Setup: 1 Hz -> 1 GHz (10,000 points, log-spaced)
    # -------------------------------------------------------------------------
    num_points = 10000
    freqs = np.logspace(np.log10(1), np.log10(1e9), num_points)

    # For storing solutions
    x_solutions = np.zeros((num_points, n_size), dtype=complex)

    # Collect parallel arguments
    parallel_args = [(f, G, C, L, RHS) for f in freqs]

    # -------------------------------------------------------------------------
    # 5) Solve in Parallel + Time It
    # -------------------------------------------------------------------------
    from time import time
    start_time = time()
    total_flops = 0

    # Use up to (cpu_count) processes
    # (You can adjust or cap it to fewer if desired)
    processes = min(cpu_count(), 8)

    with Pool(processes=processes) as pool:
        results_iter = pool.imap(solve_frequency_wrapper, parallel_args)
        for i, (freq_val, x_val, flops_val) in enumerate(
                tqdm(results_iter, total=num_points, desc="Frequency Sweep")
        ):
            x_solutions[i, :] = x_val
            total_flops += flops_val

    end_time = time()
    elapsed_time = end_time - start_time

    # -------------------------------------------------------------------------
    # 6) Save all results to CSV (all unknowns at each frequency)
    # -------------------------------------------------------------------------
    csv_filename = "all_solutions.csv"
    with open(csv_filename, "w") as f_out:
        # Header
        f_out.write("Frequency(Hz)")
        for name in unknowns:
            f_out.write(f",Re({name}),Im({name})")
        f_out.write("\n")
        # Data
        for i, freq_val in enumerate(freqs):
            row_str = f"{freq_val}"
            for val in x_solutions[i, :]:
                row_str += f",{val.real},{val.imag}"
            row_str += "\n"
            f_out.write(row_str)

    print(f"\nSaved all solutions in CSV: {csv_filename}")

    # -------------------------------------------------------------------------
    # 7) Plot Magnitude/Phase for Node Voltages => PDF
    #    (We usually only plot node voltages, i.e. first `num_nodes` unknowns.)
    # -------------------------------------------------------------------------
    magnitudes = np.abs(x_solutions[:, :num_nodes])  # shape [N, node_count]
    phases_deg = np.angle(x_solutions[:, :num_nodes], deg=True)

    with PdfPages("plot_results.pdf") as pdf_plots:
        # Magnitude
        plt.figure()
        for i_node in range(num_nodes):
            plt.semilogx(freqs, magnitudes[:, i_node], label=f"V{i_node+1}")
        plt.grid(True, which="both")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude (V)")
        plt.title("Node Voltage Magnitudes")
        plt.legend()
        pdf_plots.savefig()
        plt.close()

        # Phase
        plt.figure()
        for i_node in range(num_nodes):
            plt.semilogx(freqs, phases_deg[:, i_node], label=f"V{i_node+1}")
        plt.grid(True, which="both")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Phase [degrees]")
        plt.title("Node Voltage Phases")
        plt.legend()
        pdf_plots.savefig()
        plt.close()

    print("Saved plots to plot_results.pdf")

    # -------------------------------------------------------------------------
    # 8) Save Debug Info + FLOPs at each freq into another PDF
    #    We'll do it by building text pages in matplotlib.
    # -------------------------------------------------------------------------
    # We also want to store the per-frequency FLOPs. We must either store them
    # during the sweep or re-run. For demonstration, let's re-compute them quickly.
    # A real code might store them in a list during the parallel loop.

    def make_text_page(text_data, title="Debug Info"):
        """Helper to create a matplotlib figure with a text box."""
        fig = plt.figure(figsize=(8.5, 11))  # typical "letter" size
        plt.axis("off")
        plt.title(title)
        # Use plt.text with monospace
        plt.text(0.01, 0.95, text_data,
                 ha='left', va='top', wrap=True, family='monospace', fontsize=8)
        return fig

    # Re-compute FLOPs for each frequency (non-parallel) for demonstration
    # (In practice, you'd store them from the parallel pass.)
    flops_list = []
    for f_val in freqs:
        n = n_size
        flops_est = (2.0/3.0)*(n**3) + 2.0*(n**2)
        flops_list.append(flops_est)

    with PdfPages("debug_and_flops.pdf") as pdf_dbg:
        # Page 1: show G, C, L, RHS, M_debug
        page1_text = ""
        page1_text += matrix_to_string(G, "G")
        page1_text += matrix_to_string(C, "C")
        page1_text += matrix_to_string(L, "L  (storing -L on diag)")
        page1_text += vector_to_string(RHS, "RHS")
        page1_text += f"\nUnknowns = {unknowns}\n"
        page1_text += f"\nM_debug at {f_debug} Hz:\n{np.array_str(M_debug, precision=4, suppress_small=True)}\n"
        fig1 = make_text_page(page1_text, title="Debug Matrices / RHS")
        pdf_dbg.savefig(fig1)
        plt.close(fig1)

        # Page 2: show FLOPs for each frequency, plus total
        page2_text = "Frequency(Hz), FLOPs\n"
        for i, f_val in enumerate(freqs):
            page2_text += f"{f_val:.3g}, {flops_list[i]:.4e}\n"

        page2_text += f"\nTOTAL SWEEP FLOPs (approx) = {total_flops:.4e}"
        page2_text += f"\nTOTAL TIME (seconds) = {elapsed_time:.2f}\n"
        fig2 = make_text_page(page2_text, title="FLOPs & Timing")
        pdf_dbg.savefig(fig2)
        plt.close(fig2)

    print("Saved debug info and FLOPs/timing to debug_and_flops.pdf")

    # Final summary
    print("\n------------------------------------------------")
    print(f"Completed {num_points} frequency points from 1 Hz to 1 GHz.")
    print(f"Total simulation time: {elapsed_time:.2f} seconds")
    print(f"Approx total FLOPs: {total_flops:.4e}")
    print("------------------------------------------------")
