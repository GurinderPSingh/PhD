import numpy as np
import time
import math
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --- For parallel AC solving ---
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# --- For Tkinter file dialog ---
import tkinter as tk
from tkinter import filedialog

###############################################################################
#                 STEP 1: TKINTER FILE-DIALOG TO PICK NETLIST
###############################################################################
def pick_netlist_file():
    """
    Opens a file dialog allowing the user to select a netlist file.
    Returns the path to the selected file, or None if canceled.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the default Tkinter window
    file_path = filedialog.askopenfilename(
        title="Select a Netlist File",
        filetypes=[("Netlist Files", "*.net *.spice *.ckt"), ("All Files", "*.*")]
    )
    # If user canceled, file_path is empty string
    if not file_path:
        return None
    return file_path

###############################################################################
#                           NETLIST PARSER
###############################################################################
def parse_netlist(netlist_file):
    """
    Very simplistic netlist parser example.
    See comments for sample netlist lines.
    Returns a list of 'elements' and a dict of 'models'.
    """
    elements = []
    models = {}

    with open(netlist_file, "r", encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip blank or comment lines
            if not line or line.startswith("*") or line.startswith("//") or line.startswith("#"):
                continue
            # Check if it's a .model line
            if line.lower().startswith(".model"):
                # e.g. ".model diodeModel D IS=1e-14 ..."
                parts = line.split()
                if len(parts) < 3:
                    continue
                model_name = parts[1]
                model_type = parts[2].upper()  # e.g. D, BJT, MOS, etc.
                model_params = {}
                # parse the rest as key=value
                for param in parts[3:]:
                    if "=" in param:
                        k, v = param.split("=")
                        model_params[k.upper()] = float(v)
                models[model_name] = {
                    "type": model_type,
                    "params": model_params
                }
            elif line.lower().startswith(".end"):
                break
            else:
                # Otherwise parse an element
                parts = line.split()
                elem_name = parts[0]
                elem_type = elem_name[0].upper()  # first letter => R, C, L, V, I, etc.
                if elem_type == "R":
                    # R# node1 node2 value
                    node1 = int(parts[1])
                    node2 = int(parts[2])
                    value = float(parts[3])
                    elements.append({"type":"R","name":elem_name,"n1":node1,"n2":node2,"value":value})
                elif elem_type == "C":
                    node1 = int(parts[1])
                    node2 = int(parts[2])
                    value = float(parts[3])
                    elements.append({"type":"C","name":elem_name,"n1":node1,"n2":node2,"value":value})
                elif elem_type == "L":
                    node1 = int(parts[1])
                    node2 = int(parts[2])
                    value = float(parts[3])
                    elements.append({"type":"L","name":elem_name,"n1":node1,"n2":node2,"value":value})
                elif elem_type == "V":
                    node1 = int(parts[1])
                    node2 = int(parts[2])
                    value = float(parts[3])
                    elements.append({"type":"V","name":elem_name,"n1":node1,"n2":node2,"value":value})
                elif elem_type == "I":
                    node1 = int(parts[1])
                    node2 = int(parts[2])
                    value = float(parts[3])
                    elements.append({"type":"I","name":elem_name,"n1":node1,"n2":node2,"value":value})
                elif elem_type == "G":
                    # G# node_out_p node_out_m node_in_p node_in_m gm
                    node_out_p = int(parts[1])
                    node_out_m = int(parts[2])
                    node_in_p  = int(parts[3])
                    node_in_m  = int(parts[4])
                    gm = float(parts[5])
                    elements.append({"type":"VCCS","name":elem_name,
                                     "n_out_p":node_out_p,"n_out_m":node_out_m,
                                     "n_in_p":node_in_p,"n_in_m":node_in_m,
                                     "gm":gm})
                elif elem_type == "E":
                    # E# node_out_p node_out_m node_in_p node_in_m gain
                    node_out_p = int(parts[1])
                    node_out_m = int(parts[2])
                    node_in_p  = int(parts[3])
                    node_in_m  = int(parts[4])
                    gain = float(parts[5])
                    elements.append({"type":"VCVS","name":elem_name,
                                     "n_out_p":node_out_p,"n_out_m":node_out_m,
                                     "n_in_p":node_in_p,"n_in_m":node_in_m,
                                     "gain":gain})
                elif elem_type == "F":
                    # F# node_out_p node_out_m VsName gain (CCCS)
                    node_out_p = int(parts[1])
                    node_out_m = int(parts[2])
                    branch_name = parts[3]  # controlling branch
                    gain = float(parts[4])
                    elements.append({"type":"CCCS","name":elem_name,
                                     "n_out_p":node_out_p,"n_out_m":node_out_m,
                                     "branch_name":branch_name,"gain":gain})
                elif elem_type == "H":
                    # H# node_out_p node_out_m branch_name transres (CCVS)
                    node_out_p = int(parts[1])
                    node_out_m = int(parts[2])
                    branch_name = parts[3]
                    transres = float(parts[4])
                    elements.append({"type":"CCVS","name":elem_name,
                                     "n_out_p":node_out_p,"n_out_m":node_out_m,
                                     "branch_name":branch_name,"transres":transres})
                elif elem_type == "D":
                    # D# node_p node_m modelName
                    node_p = int(parts[1])
                    node_m = int(parts[2])
                    model_name = parts[3]
                    elements.append({"type":"D","name":elem_name,
                                     "n_p":node_p,"n_m":node_m,
                                     "model":model_name})
                elif elem_type == "Q":
                    # Q# collector base emitter modelName
                    n_c = int(parts[1])
                    n_b = int(parts[2])
                    n_e = int(parts[3])
                    model_name = parts[4]
                    elements.append({"type":"BJT","name":elem_name,
                                     "nc":n_c,"nb":n_b,"ne":n_e,
                                     "model":model_name})
                elif elem_type == "M":
                    # M# drain gate source modelName
                    nd = int(parts[1])
                    ng = int(parts[2])
                    ns = int(parts[3])
                    model_name = parts[4]
                    elements.append({"type":"MOS","name":elem_name,
                                     "nd":nd,"ng":ng,"ns":ns,
                                     "model":model_name})
                else:
                    print(f"Warning: unrecognized element line: {line}")
    return elements, models

###############################################################################
#                  NONLINEAR DEVICE MODELS (Newton iteration)
###############################################################################
def stamp_nonlinear_devices(G, RHS, x, elements, models, node_offset):
    """
    Stub for stamping nonlinear devices (diodes, BJTs, MOSFETs)
    using a Newton iteration linearization.
    This example shows a diode stamp.
    """
    for elem in elements:
        if elem["type"] == "D":
            diode_model = models.get(elem["model"], {})
            Is = diode_model.get("params", {}).get("IS", 1e-14)
            Vt = 0.0259  # thermal voltage approximate value
            n_p = elem["n_p"] - 1 if elem["n_p"] > 0 else None
            n_m = elem["n_m"] - 1 if elem["n_m"] > 0 else None
            Vp = x[n_p] if n_p is not None else 0
            Vm = x[n_m] if n_m is not None else 0
            Vd = Vp - Vm
            Id = Is * (np.exp(Vd/Vt) - 1)
            Gd = (Is/Vt) * np.exp(Vd/Vt)
            if n_p is not None:
                G[n_p, n_p] += Gd
            if n_m is not None:
                G[n_m, n_m] += Gd
            if (n_p is not None) and (n_m is not None):
                G[n_p, n_m] -= Gd
                G[n_m, n_p] -= Gd
            IncI = Id - Gd*Vd
            if n_p is not None:
                RHS[n_p] -= IncI
            if n_m is not None:
                RHS[n_m] += IncI
    return G, RHS

def newton_solve_nonlinear(elements, models, num_nodes, max_iter=50, tol=1e-9):
    """
    Skeleton for a Newton–Raphson DC operating point solution.
    """
    x = np.zeros(num_nodes, dtype=float)
    for it in range(max_iter):
        G = np.zeros((num_nodes, num_nodes), dtype=float)
        RHS = np.zeros(num_nodes, dtype=float)
        # (Stamp linear devices here as needed.)
        G, RHS = stamp_nonlinear_devices(G, RHS, x, elements, models, node_offset=1)
        x_new = np.linalg.solve(G, RHS)
        if np.linalg.norm(x_new - x, ord=2) < tol:
            print(f"Newton converged in {it+1} iterations.")
            break
        x = x_new
    return x

###############################################################################
#            FREQUENCY SWEEP & PDF OUTPUT (AC analysis)
###############################################################################
def solve_mna(M, RHS):
    return np.linalg.solve(M, RHS)

def solve_frequency_wrapper(args):
    f, base_G, base_C, base_L, base_RHS = args
    omega = 2 * np.pi * f
    s = 1j * omega
    M = base_G + s*base_C + s*base_L
    n = M.shape[0]
    flops = (2.0/3.0)*(n**3) + 2.0*(n**2)
    x = solve_mna(M, base_RHS)
    return f, x, flops

def matrix_to_string(mat, name):
    return f"{name} =\n{np.array_str(mat, precision=4, suppress_small=True)}\n"

def vector_to_string(vec, name):
    return f"{name} = {np.array_str(vec, precision=4, suppress_small=True)}\n"

def make_text_page(text_data, title="Debug Info"):
    fig = plt.figure(figsize=(8.5, 11))  # letter size
    plt.axis("off")
    plt.title(title)
    plt.text(0.01, 0.95, text_data, ha='left', va='top', wrap=True,
             family='monospace', fontsize=8)
    return fig

###############################################################################
#            NEW FUNCTIONS FOR DC & TRANSIENT ANALYSIS
###############################################################################
# For DC analysis, we stamp elements with capacitors open and inductors as shorts.
def voltage_source_stamp_custom_dc(G, RHS, unknowns, n1, n2, volt):
    size_old = G.shape[0]
    G_expanded = np.pad(G, ((0,1),(0,1)), mode='constant', constant_values=0)
    RHS_expanded = np.pad(RHS, (0,1), mode='constant', constant_values=0)
    idx = size_old
    if n1 > 0:
        G_expanded[n1-1, idx] += 1
        G_expanded[idx, n1-1] += 1
    if n2 > 0:
        G_expanded[n2-1, idx] -= 1
        G_expanded[idx, n2-1] -= 1
    RHS_expanded[idx] = volt
    unknowns.append(f"I_V_{idx}")
    return G_expanded, RHS_expanded, unknowns, idx

def vcvs_stamp_custom_dc(G, RHS, unknowns, n_out_p, n_out_m, n_in_p, n_in_m, gain):
    size_old = G.shape[0]
    G_expanded = np.pad(G, ((0,1),(0,1)), mode='constant', constant_values=0)
    RHS_expanded = np.pad(RHS, (0,1), mode='constant', constant_values=0)
    idx = size_old
    if n_out_p > 0:
        G_expanded[n_out_p-1, idx] += 1
        G_expanded[idx, n_out_p-1] += 1
    if n_out_m > 0:
        G_expanded[n_out_m-1, idx] -= 1
        G_expanded[idx, n_out_m-1] -= 1
    if n_in_p > 0:
        G_expanded[idx, n_in_p-1] -= gain
    if n_in_m > 0:
        G_expanded[idx, n_in_m-1] += gain
    unknowns.append(f"I_E_{idx}")
    return G_expanded, RHS_expanded, unknowns, idx

def ccvs_stamp_custom_dc(G, RHS, unknowns, n_out_p, n_out_m, branch_idx, transres):
    size_old = G.shape[0]
    G_expanded = np.pad(G, ((0,1),(0,1)), mode='constant', constant_values=0)
    RHS_expanded = np.pad(RHS, (0,1), mode='constant', constant_values=0)
    idx = size_old
    if n_out_p > 0:
        G_expanded[n_out_p-1, idx] += 1
        G_expanded[idx, n_out_p-1] += 1
    if n_out_m > 0:
        G_expanded[n_out_m-1, idx] -= 1
        G_expanded[idx, n_out_m-1] -= 1
    G_expanded[idx, branch_idx] -= transres
    unknowns.append(f"I_H_{idx}")
    return G_expanded, RHS_expanded, unknowns, idx

def stamp_dc_elements(elements, G, RHS, unknowns):
    """
    Stamp only linear elements for DC analysis.
    In DC: Capacitors are open (ignored) and inductors are shorts (large conductance).
    """
    for elem in elements:
        t = elem["type"]
        if t == "R":
            G = conductance_stamp_custom(G, elem["n1"], elem["n2"], 1.0/elem["value"])
        elif t == "C":
            # Open circuit: do nothing
            pass
        elif t == "L":
            # Short circuit: use a large conductance
            G = conductance_stamp_custom(G, elem["n1"], elem["n2"], 1e9)
        elif t == "V":
            G, RHS, unknowns, idx = voltage_source_stamp_custom_dc(G, RHS, unknowns, elem["n1"], elem["n2"], elem["value"])
            elem["branch_idx"] = idx
        elif t == "I":
            RHS = current_source_stamp_custom(RHS, elem["n1"], elem["n2"], elem["value"])
        elif t == "VCCS":
            G = vccs_stamp_custom(G,
                                  elem["n_out_p"], elem["n_out_m"],
                                  elem["n_in_p"], elem["n_in_m"],
                                  elem["gm"])
        elif t == "VCVS":
            G, RHS, unknowns, idx = vcvs_stamp_custom_dc(G, RHS, unknowns,
                                                         elem["n_out_p"], elem["n_out_m"],
                                                         elem["n_in_p"], elem["n_in_m"],
                                                         elem["gain"])
            elem["branch_idx"] = idx
        elif t == "CCCS":
            controlling_branch = None
            for e2 in elements:
                if e2.get("name") == elem["branch_name"]:
                    controlling_branch = e2.get("branch_idx", None)
            if controlling_branch is not None:
                G = cccs_stamp_custom(G, elem["n_out_p"], elem["n_out_m"],
                                      controlling_branch, elem["gain"])
            else:
                print(f"Warning: CCCS {elem['name']} controlling branch not found.")
        elif t == "CCVS":
            controlling_branch = None
            for e2 in elements:
                if e2.get("name") == elem["branch_name"]:
                    controlling_branch = e2.get("branch_idx", None)
            if controlling_branch is not None:
                G, RHS, unknowns, idx = ccvs_stamp_custom_dc(G, RHS, unknowns,
                                                             elem["n_out_p"], elem["n_out_m"],
                                                             controlling_branch,
                                                             elem["transres"])
                elem["branch_idx"] = idx
            else:
                print(f"Warning: CCVS {elem['name']} controlling branch not found.")
        else:
            # Skip nonlinear devices in DC stamping here.
            pass
    return G, RHS, unknowns

def run_dc_analysis(elements, num_nodes):
    """
    Perform a DC analysis by stamping the circuit with capacitors open and
    inductors shorted, then solving the resulting linear system.
    """
    # Initialize DC matrices for node-voltage analysis.
    G_dc = np.zeros((num_nodes, num_nodes), dtype=complex)
    RHS_dc = np.zeros(num_nodes, dtype=complex)
    unknowns_dc = [f"V{i}" for i in range(1, num_nodes+1)]
    G_dc, RHS_dc, unknowns_dc = stamp_dc_elements(elements, G_dc, RHS_dc, unknowns_dc)
    x_dc = np.linalg.solve(G_dc, RHS_dc)
    return x_dc, unknowns_dc

def run_transient_analysis(elements, tmax=1e-3, dt=1e-6):
    """
    Perform a rudimentary transient analysis using a backward Euler method.
    For simplicity, only stamp R, C, L, and I elements on nodes (ignore voltage sources).
    In transient analysis: capacitors are dynamic and inductors are treated via a companion model.
    Here, inductors are approximated as shorts.
    """
    # Determine maximum node number
    max_node = 0
    for e in elements:
        for key in ['n1','n2','n_p','n_m']:
            if key in e and isinstance(e[key], int) and e[key] > max_node:
                max_node = e[key]
    num_nodes = max_node
    # Build transient matrices: G (resistive) and C (capacitive)
    G_t = np.zeros((num_nodes, num_nodes), dtype=float)
    C_t = np.zeros((num_nodes, num_nodes), dtype=float)
    RHS_t = np.zeros(num_nodes, dtype=float)
    for e in elements:
        t = e["type"]
        if t == "R":
            G_t = conductance_stamp_custom(G_t, e["n1"], e["n2"], 1.0/float(e["value"]))
        elif t == "C":
            C_t = capacitance_stamp_custom(C_t, e["n1"], e["n2"], float(e["value"]))
        elif t == "L":
            # In transient, inductor approximated as short (large conductance)
            G_t = conductance_stamp_custom(G_t, e["n1"], e["n2"], 1e9)
        elif t == "I":
            RHS_t = current_source_stamp_custom(RHS_t, e["n1"], e["n2"], float(e["value"]))
    steps = int(tmax/dt)
    x_transient = np.zeros((steps, num_nodes))
    x_prev = np.zeros(num_nodes)
    for i in range(steps):
        A = G_t + C_t/dt
        b = RHS_t + (C_t/dt).dot(x_prev)
        x_new = np.linalg.solve(A, b)
        x_transient[i, :] = x_new
        x_prev = x_new
    time_points = np.linspace(dt, tmax, steps)
    return time_points, x_transient, [f"V{i}" for i in range(1, num_nodes+1)]

###############################################################################
#                  LINEAR ELEMENT STAMPING (AC analysis)
# (ORIGINAL CODE - DO NOT MODIFY)
###############################################################################
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

def current_source_stamp_custom(RHS, n1, n2, val):
    if n1 > 0:
        RHS[n1-1] += val
    if n2 > 0:
        RHS[n2-1] -= val
    return RHS

def voltage_source_stamp_custom(G, C, L, RHS, n1, n2, volt):
    size_old = G.shape[0]
    G_expanded = np.pad(G, ((0,1),(0,1)), mode='constant', constant_values=0)
    C_expanded = np.pad(C, ((0,1),(0,1)), mode='constant', constant_values=0)
    L_expanded = np.pad(L, ((0,1),(0,1)), mode='constant', constant_values=0)
    RHS_expanded = np.pad(RHS, (0,1), mode='constant', constant_values=0)
    idx = size_old
    if n1 > 0:
        G_expanded[n1-1, idx] += 1
        G_expanded[idx, n1-1] += 1
    if n2 > 0:
        G_expanded[n2-1, idx] -= 1
        G_expanded[idx, n2-1] -= 1
    RHS_expanded[idx] = volt
    return G_expanded, C_expanded, L_expanded, RHS_expanded, idx

def inductor_stamp_custom(G, C, L, RHS, n1, n2, val):
    size_old = G.shape[0]
    G_expanded = np.pad(G, ((0,1),(0,1)), mode='constant')
    C_expanded = np.pad(C, ((0,1),(0,1)), mode='constant')
    L_expanded = np.pad(L, ((0,1),(0,1)), mode='constant')
    RHS_expanded = np.pad(RHS, (0,1), mode='constant')
    idx = size_old
    if n1 > 0:
        G_expanded[n1-1, idx] += 1
        G_expanded[idx, n1-1] += 1
    if n2 > 0:
        G_expanded[n2-1, idx] -= 1
        G_expanded[idx, n2-1] -= 1
    L_expanded[idx, idx] = -val
    return G_expanded, C_expanded, L_expanded, RHS_expanded, idx

def vccs_stamp_custom(G, nop, nom, nip, nim, gm):
    p_out = nop-1 if nop>0 else None
    m_out = nom-1 if nom>0 else None
    p_in =  nip-1 if nip>0 else None
    m_in =  nim-1 if nim>0 else None
    if p_out is not None and p_in is not None:
        G[p_out, p_in] += gm
    if p_out is not None and m_in is not None:
        G[p_out, m_in] -= gm
    if m_out is not None and p_in is not None:
        G[m_out, p_in] -= gm
    if m_out is not None and m_in is not None:
        G[m_out, m_in] += gm
    return G

def vcvs_stamp_custom(G, C, L, RHS, nop, nom, nip, nim, gain):
    size_old = G.shape[0]
    G_expanded = np.pad(G, ((0,1),(0,1)), mode='constant')
    C_expanded = np.pad(C, ((0,1),(0,1)), mode='constant')
    L_expanded = np.pad(L, ((0,1),(0,1)), mode='constant')
    RHS_expanded = np.pad(RHS, (0,1), mode='constant')
    idx = size_old
    if nop > 0:
        G_expanded[nop-1, idx] += 1
        G_expanded[idx, nop-1] += 1
    if nom > 0:
        G_expanded[nom-1, idx] -= 1
        G_expanded[idx, nom-1] -= 1
    if nip > 0:
        G_expanded[idx, nip-1] -= gain
    if nim > 0:
        G_expanded[idx, nim-1] += gain
    return G_expanded, C_expanded, L_expanded, RHS_expanded, idx

def cccs_stamp_custom(G, nop, nom, branch_idx, gain):
    p_out = nop-1 if nop>0 else None
    m_out = nom-1 if nom>0 else None
    if p_out is not None:
        G[p_out, branch_idx] += gain
    if m_out is not None:
        G[m_out, branch_idx] -= gain
    return G

def ccvs_stamp_custom(G, C, L, RHS, nop, nom, branch_idx, transres):
    size_old = G.shape[0]
    G_expanded = np.pad(G, ((0,1),(0,1)), mode='constant')
    C_expanded = np.pad(C, ((0,1),(0,1)), mode='constant')
    L_expanded = np.pad(L, ((0,1),(0,1)), mode='constant')
    RHS_expanded = np.pad(RHS, (0,1), mode='constant')
    idx = size_old
    if nop > 0:
        G_expanded[nop-1, idx] += 1
        G_expanded[idx, nop-1] += 1
    if nom > 0:
        G_expanded[nom-1, idx] -= 1
        G_expanded[idx, nom-1] -= 1
    G_expanded[idx, branch_idx] -= transres
    return G_expanded, C_expanded, L_expanded, RHS_expanded, idx

###############################################################################
#                                   MAIN
###############################################################################
if __name__ == "__main__":
    # 1) Ask user to pick netlist file via Tkinter
    netlist_file = pick_netlist_file()
    if not netlist_file:
        print("No netlist file selected. Exiting.")
        sys.exit(0)

    print(f"Loading netlist from: {netlist_file}")
    if not os.path.isfile(netlist_file):
        print(f"Error: netlist file '{netlist_file}' not found.")
        sys.exit(1)

    print(f"Loading netlist: {netlist_file}")
    elements, models = parse_netlist(netlist_file)

    # -------------------------------------------------------------------------
    # 2) Determine the highest node number from the elements
    # -------------------------------------------------------------------------
    max_node = 0
    for e in elements:
        # Check common node keys
        for key in e:
            if key.startswith("n") and isinstance(e[key], int):
                if e[key] > max_node:
                    max_node = e[key]
        # Check special names
        for key in ["n_out_p", "n_out_m", "n_in_p", "n_in_m"]:
            if key in e and e[key] > max_node:
                max_node = e[key]

    num_nodes = max_node
    # Initialize MNA matrices for AC analysis (as in original)
    G = np.zeros((num_nodes, num_nodes), dtype=complex)
    C = np.zeros((num_nodes, num_nodes), dtype=complex)
    L = np.zeros((num_nodes, num_nodes), dtype=complex)
    RHS = np.zeros(num_nodes, dtype=complex)
    unknowns = [f"V{i}" for i in range(1, num_nodes+1)]

    # Stamp linear elements (AC stamping as originally implemented)
    def stamp_linear_elements(elements, G, C, L, RHS, unknowns):
        for elem in elements:
            t = elem["type"]
            if t == "R":
                G = conductance_stamp_custom(G, elem["n1"], elem["n2"], 1.0/elem["value"])
            elif t == "C":
                C = capacitance_stamp_custom(C, elem["n1"], elem["n2"], elem["value"])
            elif t == "L":
                G, C, L, RHS, idx_l = inductor_stamp_custom(G, C, L, RHS, elem["n1"], elem["n2"], elem["value"])
                unknowns.append(elem["name"]+"_I")
                elem["branch_idx"] = idx_l
            elif t == "V":
                G, C, L, RHS, idx_vs = voltage_source_stamp_custom(G, C, L, RHS, elem["n1"], elem["n2"], elem["value"])
                unknowns.append(elem["name"]+"_I")
                elem["branch_idx"] = idx_vs
            elif t == "I":
                RHS = current_source_stamp_custom(RHS, elem["n1"], elem["n2"], elem["value"])
            elif t == "VCCS":
                G = vccs_stamp_custom(G, elem["n_out_p"], elem["n_out_m"], elem["n_in_p"], elem["n_in_m"], elem["gm"])
            elif t == "VCVS":
                G, C, L, RHS, idx_vcvs = vcvs_stamp_custom(G, C, L, RHS, elem["n_out_p"], elem["n_out_m"], elem["n_in_p"], elem["n_in_m"], elem["gain"])
                unknowns.append(elem["name"]+"_I")
                elem["branch_idx"] = idx_vcvs
            elif t == "CCCS":
                controlling_branch = None
                for e2 in elements:
                    if e2.get("name") == elem["branch_name"]:
                        controlling_branch = e2.get("branch_idx", None)
                if controlling_branch is not None:
                    G = cccs_stamp_custom(G, elem["n_out_p"], elem["n_out_m"], controlling_branch, elem["gain"])
                else:
                    print(f"Warning: CCCS {elem['name']} controlling branch not found.")
            elif t == "CCVS":
                controlling_branch = None
                for e2 in elements:
                    if e2.get("name") == elem["branch_name"]:
                        controlling_branch = e2.get("branch_idx", None)
                if controlling_branch is not None:
                    G, C, L, RHS, idx_ccvs = ccvs_stamp_custom(G, C, L, RHS, elem["n_out_p"], elem["n_out_m"], controlling_branch, elem["transres"])
                    unknowns.append(elem["name"]+"_I")
                    elem["branch_idx"] = idx_ccvs
                else:
                    print(f"Warning: CCVS {elem['name']} controlling branch not found.")
            else:
                print(f"Skipping unhandled element '{elem['name']}' of type '{t}'.")
        return G, C, L, RHS, unknowns

    # Stamp for AC analysis (this does not stamp nonlinear devices)
    G, C, L, RHS, unknowns = stamp_linear_elements(elements, G, C, L, RHS, unknowns)

    # -------------------------------------------------------------------------
    # 3) Ask user to select analysis type
    # -------------------------------------------------------------------------
    print("\nSelect analysis type:")
    print("1) DC Analysis")
    print("2) AC Analysis")
    print("3) Transient Analysis")
    analysis_choice = input("Enter your choice (1/2/3): ").strip()

    if analysis_choice == "1":
        # ------------------- DC Analysis -------------------
        print("\n--- Running DC Analysis ---")
        # For DC analysis, we build new matrices (capacitors open, inductors shorted)
        G_dc = np.zeros((num_nodes, num_nodes), dtype=complex)
        RHS_dc = np.zeros(num_nodes, dtype=complex)
        unknowns_dc = [f"V{i}" for i in range(1, num_nodes+1)]
        G_dc, RHS_dc, unknowns_dc = stamp_dc_elements(elements, G_dc, RHS_dc, unknowns_dc)
        x_dc = np.linalg.solve(G_dc, RHS_dc)
        # Save DC solution to CSV
        dc_csv = "dc_solution.csv"
        with open(dc_csv, "w") as f_out:
            f_out.write("Unknown,Voltage\n")
            for i, val in enumerate(x_dc):
                f_out.write(f"{unknowns_dc[i]},{val.real}\n")
        print(f"DC solution saved to {dc_csv}")
        print("DC Node Voltages:")
        for name, voltage in zip(unknowns_dc, x_dc):
            print(f"  {name} = {voltage.real:.4g} V")
    elif analysis_choice == "2":
        # ------------------- AC Analysis -------------------
        print("\n--- Running AC Analysis ---")
        # Ask user for number of frequency points
        num_points_input = input("Enter number of frequency points (e.g., 1000): ").strip()
        try:
            num_points = int(num_points_input)
        except:
            print("Invalid number entered. Using default 1000 frequency points.")
            num_points = 1000

        # Debug print of matrices at a chosen frequency
        f_debug = 1e3
        s_debug = 1j * 2 * math.pi * f_debug
        M_debug = G + s_debug * C + s_debug * L
        print("\n--- Debug Print ---")
        print(matrix_to_string(G, "G"))
        print(matrix_to_string(C, "C"))
        print(matrix_to_string(L, "L"))
        print(vector_to_string(RHS, "RHS"))
        print(f"Unknowns = {unknowns}")
        print(f"\nM_debug at f={f_debug} Hz:\n{M_debug}\n")

        freqs = np.logspace(np.log10(1), np.log10(1e9), num_points)
        n_size = G.shape[0]
        x_solutions = np.zeros((num_points, n_size), dtype=complex)
        parallel_args = [(f, G, C, L, RHS) for f in freqs]
        start_time_all = time.time()
        total_flops = 0
        processes = min(cpu_count(), 8)
        with Pool(processes=processes) as pool:
            results_iter = pool.imap(solve_frequency_wrapper, parallel_args)
            for i, (freq_val, x_val, flops_val) in enumerate(tqdm(results_iter, total=num_points, desc="Frequency Sweep")):
                x_solutions[i, :] = x_val
                total_flops += flops_val
        elapsed_time_all = time.time() - start_time_all

        # Save AC solutions to CSV
        ac_csv = "ac_solution.csv"
        with open(ac_csv, "w") as f_out:
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
        print(f"AC solution saved to {ac_csv}")

        # Plot node voltages magnitude/phase for actual nodes only
        node_count = sum(1 for n in unknowns if n.startswith("V"))
        mag = np.abs(x_solutions[:, :node_count])
        pha = np.angle(x_solutions[:, :node_count], deg=True)
        with PdfPages("ac_plots.pdf") as pdfp:
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
        print("AC plots saved to ac_plots.pdf")

        # Generate debug PDF
        with PdfPages("debug_and_flops.pdf") as pdfd:
            page1_text = ""
            page1_text += matrix_to_string(G, "G")
            page1_text += matrix_to_string(C, "C")
            page1_text += matrix_to_string(L, "L")
            page1_text += vector_to_string(RHS, "RHS")
            page1_text += f"\nUnknowns:\n{unknowns}\n"
            page1_text += f"\nM_debug at f={f_debug} Hz:\n{np.array_str(M_debug, precision=4, suppress_small=True)}\n"
            fig1 = make_text_page(page1_text, title="Debug Matrices / M_debug")
            pdfd.savefig(fig1)
            plt.close(fig1)
            page2_text = "Frequency(Hz), FLOPs\n"
            n = n_size
            single_flops = (2.0/3.0)*(n**3) + 2.0*(n**2)
            for i in range(min(10, num_points)):
                page2_text += f"{freqs[i]:.4g}, {single_flops:.4e}\n"
            page2_text += f" ... (omitted many)\n"
            page2_text += f"\nTOTAL SWEEP FLOPs = {total_flops:.4e}\n"
            page2_text += f"TOTAL TIME (sec) = {elapsed_time_all:.2f}\n"
            fig2 = make_text_page(page2_text, title="FLOPs & Timing")
            pdfd.savefig(fig2)
            plt.close(fig2)
            summary_text = f"AC Simulation completed\nFrequency Points = {num_points}\n"
            summary_text += f"Total Time = {elapsed_time_all:.2f} s\n"
            summary_text += f"Total FLOPs = {total_flops:.4e}\n"
            fig3 = make_text_page(summary_text, title="Final Summary")
            pdfd.savefig(fig3)
            plt.close(fig3)
        print("Saved debug info, total time & FLOPs to debug_and_flops.pdf")
    elif analysis_choice == "3":
        # ------------------- Transient Analysis -------------------
        print("\n--- Running Transient Analysis ---")
        # Ask user for tmax and dt with examples:
        print("Please enter the total simulation time (tmax) and time step (dt) in seconds.")
        print("For example, enter '1e-3' for tmax (which means 0.001 s) and '1e-6' for dt (0.000001 s).")
        tmax_input = input("Enter tmax (e.g., 1e-3): ").strip()
        dt_input = input("Enter dt (e.g., 1e-6): ").strip()
        try:
            tmax = float(tmax_input)
            dt = float(dt_input)
        except:
            print("Invalid input for tmax or dt. Using default values: tmax=1e-3, dt=1e-6.")
            tmax = 1e-3
            dt = 1e-6

        t, x_transient, unknowns_trans = run_transient_analysis(elements, tmax=tmax, dt=dt)
        # Save transient solution to CSV
        trans_csv = "transient_solution.csv"
        with open(trans_csv, "w") as f_out:
            header = "Time"
            for name in unknowns_trans:
                header += f",{name}"
            f_out.write(header + "\n")
            for i, time_val in enumerate(t):
                row = f"{time_val}"
                for val in x_transient[i, :]:
                    row += f",{val}"
                row += "\n"
                f_out.write(row)
        print(f"Transient solution saved to {trans_csv}")
        # Plot transient node voltages
        node_count = sum(1 for n in unknowns_trans if n.startswith("V"))
        plt.figure()
        for i in range(node_count):
            plt.plot(t, x_transient[:, i], label=unknowns_trans[i])
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.title("Transient Node Voltages")
        plt.grid(True)
        plt.legend()
        plt.savefig("transient_plot.pdf")
        plt.close()
        print("Transient plot saved to transient_plot.pdf")
    else:
        print("Invalid analysis type selected. Exiting.")
        sys.exit(1)

    print("\n=== Simulation Complete ===")
