import numpy as np
import time
import math
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

def parse_netlist(netlist_file):
    # [Your existing netlist parsing code]
    # Ensure it returns elements and models correctly
    pass

def conductance_stamp_custom(G, n1, n2, val):
    if n1 > 0:
        G[n1-1, n1-1] += val
    if n2 > 0:
        G[n2-1, n2-1] += val
    if n1 > 0 and n2 > 0:
        G[n1-1, n2-1] -= val
        G[n2-1, n1-1] -= val
    return G

def capacitance_stamp_custom(C, n1, n2, val):
    if n1 > 0:
        C[n1-1, n1-1] += val
    if n2 > 0:
        C[n2-1, n2-1] += val
    if n1 > 0 and n2 > 0:
        C[n1-1, n2-1] -= val
        C[n2-1, n1-1] -= val
    return C

def inductance_stamp_custom(L, n1, n2, value, current_var_index):
    # Implement inductor stamping here
    return L

def current_source_stamp_custom(RHS, n1, n2, val):
    if n1 > 0:
        RHS[n1-1] += val
    if n2 > 0:
        RHS[n2-1] -= val
    return RHS

def solve_mna(M, RHS):
    try:
        return np.linalg.solve(M, RHS)
    except np.linalg.LinAlgError as e:
        print(f"LinAlgError during MNA solve: {e}")
        return np.full(RHS.shape, np.nan, dtype=complex)

def solve_frequency_wrapper(args):
    f, base_G, base_C, base_L, base_RHS = args
    omega = 2.0 * math.pi * f
    s = 1j * omega
    if base_G.shape == base_C.shape == base_L.shape:
        M = base_G + s * base_C + s * base_L
    else:
        raise ValueError(f"Matrix shapes mismatch: G{base_G.shape}, C{base_C.shape}, L{base_L.shape}")
    n = M.shape[0]
    flops = (2.0 / 3.0) * (n**3) + 2.0 * (n**2)
    x = solve_mna(M, base_RHS)
    return f, x, flops

def matrix_to_string(mat, name):
    return f"{name} =\n{np.array_str(mat, precision=4, suppress_small=True)}\n"

def vector_to_string(vec, name):
    return f"{name} = {np.array_str(vec, precision=4, suppress_small=True)}\n"

def make_text_page(text_data, title="Debug Info"):
    fig = plt.figure(figsize=(8.5, 11))  # Letter size
    plt.axis("off")
    plt.title(title)
    plt.text(0.01, 0.95, text_data, ha='left', va='top', wrap=True,
             family='monospace', fontsize=8)
    return fig

class SpiceSimulatorGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Smart Spice Simulator")
        self.master.geometry("700x500")
        self.netlist_file = None
        self.elements = []
        self.models = {}
        self.simulation_thread = None

        # Create menu
        self.create_menu()

        # Create main frame
        self.create_main_frame()

        # Initialize progress variables
        self.progress = 0
        self.total_flops = 0
        self.elapsed_time = 0

    def create_menu(self):
        menubar = tk.Menu(self.master)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open Netlist...", command=self.open_netlist)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.master.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        self.master.config(menu=menubar)

    def create_main_frame(self):
        frame = tk.Frame(self.master)
        frame.pack(pady=20)

        # Button to start simulation
        self.simulate_button = tk.Button(frame, text="Run Simulation", command=self.run_simulation, state=tk.DISABLED, width=20, height=2)
        self.simulate_button.pack(pady=10)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(frame, variable=self.progress_var, maximum=100, length=500)
        self.progress_bar.pack(pady=10)

        # Status label
        self.status_label = tk.Label(frame, text="Load a netlist to start.", wraplength=600, justify=tk.LEFT)
        self.status_label.pack(pady=10)

        # Text box for additional logs or messages
        self.log_text = tk.Text(frame, height=10, width=80, state=tk.DISABLED)
        self.log_text.pack(pady=10)

    def open_netlist(self):
        file_path = filedialog.askopenfilename(
            title="Select a Netlist File",
            filetypes=[("Netlist Files", "*.net *.spice *.ckt"), ("All Files", "*.*")]
        )
        if file_path:
            self.netlist_file = file_path
            self.status_label.config(text=f"Selected Netlist: {self.netlist_file}")
            self.log(f"Selected Netlist: {self.netlist_file}")
            self.load_netlist()

    def load_netlist(self):
        if not self.netlist_file:
            messagebox.showerror("Error", "No netlist file selected.")
            return

        # Parse netlist
        try:
            self.elements, self.models = parse_netlist(self.netlist_file)
            self.status_label.config(text=f"Parsed netlist successfully: {self.netlist_file}")
            self.log(f"Parsed netlist successfully: {self.netlist_file}")
            self.simulate_button.config(state=tk.NORMAL)
        except UnicodeDecodeError as e:
            messagebox.showerror("Encoding Error", f"Failed to decode the netlist file:\n{e}")
            self.simulate_button.config(state=tk.DISABLED)
            self.log(f"Encoding Error: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while parsing the netlist:\n{e}")
            self.simulate_button.config(state=tk.DISABLED)
            self.log(f"Error while parsing netlist: {e}")

    def run_simulation(self):
        if not self.elements:
            messagebox.showerror("Error", "No elements to simulate.")
            return

        # Disable buttons during simulation
        self.simulate_button.config(state=tk.DISABLED)
        self.log("Starting simulation...")

        # Start simulation in a separate thread to keep GUI responsive
        self.simulation_thread = threading.Thread(target=self.simulation_process)
        self.simulation_thread.start()

    def simulation_process(self):
        # Identify the highest node number
        max_node = 0
        for e in self.elements:
            # Identify nodes based on element type
            node_keys = []
            if e["type"] in ["R", "C", "L", "V", "I", "D"]:
                node_keys = ["n1", "n2", "n_p", "n_m"] if "n_p" in e else ["n1", "n2"]
            elif e["type"] in ["VCCS", "VCVS", "CCCS", "CCVS"]:
                node_keys = ["n_out_p", "n_out_m", "n_in_p", "n_in_m"]
            elif e["type"] in ["BJT"]:
                node_keys = ["nc", "nb", "ne"]
            elif e["type"] in ["MOS"]:
                node_keys = ["nd", "ng", "ns"]

            for key in node_keys:
                if key in e and isinstance(e[key], int):
                    if e[key] > max_node:
                        max_node = e[key]

        self.log(f"Max node identified: {max_node}")

        # Count additional variables for voltage sources and dependent sources
        voltage_sources = {}
        dependent_sources = {}
        vs_count = 0
        ds_count = 0
        for elem in self.elements:
            if elem["type"] in ["V", "VCVS", "CCVS"]:
                vs_count += 1
                voltage_sources[elem["name"]] = vs_count  # Assign an index
            if elem["type"] in ["VCCS", "VCVS", "CCCS", "CCVS"]:
                ds_count += 1
                dependent_sources[elem["name"]] = ds_count  # Assign an index

        total_unknowns = max_node + vs_count + ds_count
        self.log(f"Total unknowns (nodes + sources): {total_unknowns}")

        # Initialize MNA matrices with consistent dimensions
        G_extended = np.zeros((total_unknowns, total_unknowns), dtype=complex)
        C_extended = np.zeros((total_unknowns, total_unknowns), dtype=complex)
        L_extended = np.zeros((total_unknowns, total_unknowns), dtype=complex)
        RHS_extended = np.zeros(total_unknowns, dtype=complex)

        unknowns = [f"V{i}" for i in range(1, max_node+1)]
        for vs_name in voltage_sources:
            unknowns.append(f"I_{vs_name}")  # Current through voltage sources

        # Stamp linear elements
        for elem in self.elements:
            t = elem["type"]
            if t == "R":
                G_extended = conductance_stamp_custom(G_extended, elem["n1"], elem["n2"], 1.0 / elem["value"])
            elif t == "C":
                C_extended = capacitance_stamp_custom(C_extended, elem["n1"], elem["n2"], elem["value"])
            elif t == "L":
                # Inductor handling: Placeholder
                self.log(f"Inductor '{elem['name']}' is currently not implemented.")
                continue
            elif t == "I":
                RHS_extended = current_source_stamp_custom(RHS_extended, elem["n1"], elem["n2"], elem["value"])
            elif t == "V":
                vs_index = voltage_sources[elem["name"]]
                n1 = elem["n1"]
                n2 = elem["n2"]
                # Stamp voltage source in G_extended
                if n1 > 0:
                    G_extended[max_node + vs_index -1, n1-1] += 1
                    G_extended[n1-1, max_node + vs_index -1] += 1
                if n2 > 0:
                    G_extended[max_node + vs_index -1, n2-1] -= 1
                    G_extended[n2-1, max_node + vs_index -1] -= 1
                # Set RHS for voltage source
                RHS_extended[max_node + vs_index -1] = elem["value"]
            elif t == "VCCS":
                n_out_p = elem["n_out_p"]
                n_out_m = elem["n_out_m"]
                n_in_p = elem["n_in_p"]
                n_in_m = elem["n_in_m"]
                gm = elem["gm"]
                # Stamp VCCS conductance
                if n_in_p > 0 and n_in_m > 0:
                    G_extended[n_out_p-1, n_in_p-1] += gm
                    G_extended[n_out_p-1, n_in_m-1] -= gm
                    G_extended[n_out_m-1, n_in_p-1] -= gm
                    G_extended[n_out_m-1, n_in_m-1] += gm
            elif t == "VCVS":
                vs_index = voltage_sources[elem["name"]]
                n_out_p = elem["n_out_p"]
                n_out_m = elem["n_out_m"]
                n_in_p = elem["n_in_p"]
                n_in_m = elem["n_in_m"]
                gain = elem["gain"]
                # Stamp VCVS in G_extended
                if n_out_p > 0:
                    G_extended[max_node + vs_index -1, n_out_p-1] += 1
                    G_extended[n_out_p-1, max_node + vs_index -1] += 1
                if n_out_m > 0:
                    G_extended[max_node + vs_index -1, n_out_m-1] -= 1
                    G_extended[n_out_m-1, max_node + vs_index -1] -= 1
                # Stamp the controlling voltage
                if n_in_p > 0:
                    G_extended[max_node + vs_index -1, n_in_p-1] -= gain
                if n_in_m > 0:
                    G_extended[max_node + vs_index -1, n_in_m-1] += gain
            elif t == "CCCS":
                # Find controlling branch index
                controlling_branch = elem["branch_name"]
                # Assuming 'controlling_branch' refers to a voltage source
                if controlling_branch in voltage_sources:
                    controlling_vs_index = voltage_sources[controlling_branch]
                    gain = elem["gain"]
                    G_extended[elem["n_out_p"]-1, max_node + controlling_vs_index -1] += gain
                    G_extended[elem["n_out_m"]-1, max_node + controlling_vs_index -1] -= gain
                else:
                    self.log(f"Warning: Controlling voltage source '{controlling_branch}' for CCCS '{elem['name']}' not found.")
            elif t == "CCVS":
                # Implement CCVS handling here
                self.log(f"CCVS '{elem['name']}' is currently not implemented.")
                continue
            elif t == "D":
                # Implement Diode stamping if necessary
                self.log(f"Diode '{elem['name']}' is currently not implemented.")
                continue
            elif t in ["BJT", "MOS"]:
                # Implement BJT/MOSFET stamping if necessary
                self.log(f"Non-linear device '{elem['name']}' of type '{t}' is currently not implemented.")
                continue
            else:
                self.log(f"Skipping unhandled element '{elem['name']}' of type '{t}'.")

        # Debug: Check matrix shapes
        self.log(f"Shape of G_extended: {G_extended.shape}")
        self.log(f"Shape of C_extended: {C_extended.shape}")
        self.log(f"Shape of L_extended: {L_extended.shape}")
        self.log(f"Shape of RHS_extended: {RHS_extended.shape}")

        # Ensure matrices have consistent shapes
        if not (G_extended.shape == C_extended.shape == L_extended.shape):
            messagebox.showerror("Matrix Shape Error", f"Matrix shapes mismatch:\nG: {G_extended.shape}\nC: {C_extended.shape}\nL: {L_extended.shape}")
            self.log(f"Matrix shapes mismatch: G {G_extended.shape}, C {C_extended.shape}, L {L_extended.shape}")
            self.simulate_button.config(state=tk.NORMAL)
            return

        # Prepare frequency sweep
        num_points = 1000
        freqs = np.logspace(np.log10(1), np.log10(1e9), num_points)
        x_solutions = np.zeros((num_points, total_unknowns), dtype=complex)

        parallel_args = [(f, G_extended, C_extended, L_extended, RHS_extended) for f in freqs]

        start_time_all = time.time()
        total_flops = 0

        # Set up a pool with limited number of processes
        processes = min(cpu_count(), 8)
        with Pool(processes=processes) as pool:
            # Use tqdm for progress visualization
            results_iter = pool.imap(solve_frequency_wrapper, parallel_args)
            for i, (freq_val, x_val, flops_val) in enumerate(
                    tqdm(results_iter, total=num_points, desc="Frequency Sweep")
            ):
                x_solutions[i, :] = x_val
                total_flops += flops_val

        elapsed_time_all = time.time() - start_time_all

        # Save solutions to CSV
        csv_file = "all_solutions.csv"
        with open(csv_file, "w") as f_out:
            f_out.write("Freq(Hz)")
            for name in unknowns:
                f_out.write(f",Re({name}),Im({name})")
            f_out.write("\n")
            for i, freq_val in enumerate(freqs):
                row = f"{freq_val}"
                for val in x_solutions[i, :]:
                    row += f",{val.real},{val.imag}"
                row += "\n"
                f_out.write(row)
        self.log(f"Saved solution data to {csv_file}")

        # Plot node voltages magnitude/phase
        node_count = sum(1 for n in unknowns if n.startswith("V"))
        mag = np.abs(x_solutions[:, :node_count])
        pha = np.angle(x_solutions[:, :node_count], deg=True)

        with PdfPages("plot_results.pdf") as pdfp:
            # Magnitude
            plt.figure()
            for i_node in range(node_count):
                plt.semilogx(freqs, mag[:, i_node], label=unknowns[i_node])
            plt.grid(True, which="both")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Magnitude")
            plt.title("Node Voltages: Magnitude")
            plt.legend()
            pdfp.savefig()
            plt.close()

            # Phase
            plt.figure()
            for i_node in range(node_count):
                plt.semilogx(freqs, pha[:, i_node], label=unknowns[i_node])
            plt.grid(True, which="both")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Phase [deg]")
            plt.title("Node Voltages: Phase")
            plt.legend()
            pdfp.savefig()
            plt.close()

        self.log("Saved plots to plot_results.pdf")

        # Debug & FLOPs PDF
        with PdfPages("debug_and_flops.pdf") as pdfd:
            # Page1: G, C, L, RHS, M_debug
            page1_text = ""
            page1_text += matrix_to_string(G_extended, "G")
            page1_text += matrix_to_string(C_extended, "C")
            page1_text += matrix_to_string(L_extended, "L")
            page1_text += vector_to_string(RHS_extended, "RHS")
            page1_text += f"\nUnknowns:\n{unknowns}\n"
            # For M_debug, pick the first frequency
            if len(freqs) > 0:
                f_debug = freqs[0]
                s_debug = 1j * 2 * math.pi * f_debug
                M_debug = G_extended + s_debug * C_extended + s_debug * L_extended
                page1_text += f"\nM_debug at f={f_debug} Hz:\n{np.array_str(M_debug, precision=4, suppress_small=True)}\n"
            fig1 = make_text_page(page1_text, title="Debug Matrices / M_debug")
            pdfd.savefig(fig1)
            plt.close(fig1)

            # Page2: Per-frequency FLOPs (approx)
            page2_text = "Frequency(Hz), FLOPs\n"
            n = total_unknowns
            single_flops = (2.0 / 3.0) * (n**3) + 2.0 * (n**2)
            sample_indices = np.linspace(0, num_points-1, num=10, dtype=int)
            for i in sample_indices:
                page2_text += f"{freqs[i]:.4g}, {single_flops:.4e}\n"
            page2_text += f" ... (Remaining frequencies omitted)\n"
            page2_text += f"\nTOTAL SWEEP FLOPs = {total_flops:.4e}\n"
            page2_text += f"TOTAL TIME (sec) = {elapsed_time_all:.2f}\n"

            fig2 = make_text_page(page2_text, title="FLOPs & Timing")
            pdfd.savefig(fig2)
            plt.close(fig2)

            # Page3: Final summary
            summary_text = f"Simulation completed\nFrequency Points = {num_points}\n"
            summary_text += f"Total Time = {elapsed_time_all:.2f} s\n"
            summary_text += f"Total FLOPs = {total_flops:.4e}\n"
            fig3 = make_text_page(summary_text, title="Final Summary")
            pdfd.savefig(fig3)
            plt.close(fig3)

        self.log("Saved debug info, total time & FLOPs to debug_and_flops.pdf")

        # Re-enable simulation button
        self.simulate_button.config(state=tk.NORMAL)
        self.log("Simulation process completed.")

    def log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

# Main function to run the GUI
def main():
    root = tk.Tk()
    app = SpiceSimulatorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
