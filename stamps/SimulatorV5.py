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

# --- For iterative and sparse solvers ---
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg, gmres, bicgstab, minres, spilu, spsolve, LinearOperator

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
            if not line or line.startswith("*") or line.startswith("//") or line.startswith("#"):
                continue
            if line.lower().startswith(".model"):
                parts = line.split()
                if len(parts) < 3:
                    continue
                model_name = parts[1]
                model_type = parts[2].upper()
                model_params = {}
                for param in parts[3:]:
                    if "=" in param:
                        k, v = param.split("=")
                        model_params[k.upper()] = float(v)
                models[model_name] = {"type": model_type, "params": model_params}
            elif line.lower().startswith(".end"):
                break
            else:
                parts = line.split()
                elem_name = parts[0]
                elem_type = elem_name[0].upper()
                if elem_type == "R":
                    node1 = int(parts[1]); node2 = int(parts[2]); value = float(parts[3])
                    elements.append({"type": "R", "name": elem_name, "n1": node1, "n2": node2, "value": value})
                elif elem_type == "C":
                    node1 = int(parts[1]); node2 = int(parts[2]); value = float(parts[3])
                    elements.append({"type": "C", "name": elem_name, "n1": node1, "n2": node2, "value": value})
                elif elem_type == "L":
                    node1 = int(parts[1]); node2 = int(parts[2]); value = float(parts[3])
                    elements.append({"type": "L", "name": elem_name, "n1": node1, "n2": node2, "value": value})
                elif elem_type == "V":
                    node1 = int(parts[1]); node2 = int(parts[2]); value = float(parts[3])
                    elements.append({"type": "V", "name": elem_name, "n1": node1, "n2": node2, "value": value})
                elif elem_type == "I":
                    node1 = int(parts[1]); node2 = int(parts[2]); value = float(parts[3])
                    elements.append({"type": "I", "name": elem_name, "n1": node1, "n2": node2, "value": value})
                elif elem_type == "G":
                    node_out_p = int(parts[1]); node_out_m = int(parts[2])
                    node_in_p  = int(parts[3]); node_in_m  = int(parts[4]); gm = float(parts[5])
                    elements.append({"type": "VCCS", "name": elem_name,
                                     "n_out_p": node_out_p, "n_out_m": node_out_m,
                                     "n_in_p": node_in_p, "n_in_m": node_in_m, "gm": gm})
                elif elem_type == "E":
                    node_out_p = int(parts[1]); node_out_m = int(parts[2])
                    node_in_p  = int(parts[3]); node_in_m  = int(parts[4]); gain = float(parts[5])
                    elements.append({"type": "VCVS", "name": elem_name,
                                     "n_out_p": node_out_p, "n_out_m": node_out_m,
                                     "n_in_p": node_in_p, "n_in_m": node_in_m, "gain": gain})
                elif elem_type == "F":
                    node_out_p = int(parts[1]); node_out_m = int(parts[2])
                    branch_name = parts[3]; gain = float(parts[4])
                    elements.append({"type": "CCCS", "name": elem_name,
                                     "n_out_p": node_out_p, "n_out_m": node_out_m,
                                     "branch_name": branch_name, "gain": gain})
                elif elem_type == "H":
                    node_out_p = int(parts[1]); node_out_m = int(parts[2])
                    branch_name = parts[3]; transres = float(parts[4])
                    elements.append({"type": "CCVS", "name": elem_name,
                                     "n_out_p": node_out_p, "n_out_m": node_out_m,
                                     "branch_name": branch_name, "transres": transres})
                elif elem_type == "D":
                    node_p = int(parts[1]); node_m = int(parts[2]); model_name = parts[3]
                    elements.append({"type": "D", "name": elem_name,
                                     "n_p": node_p, "n_m": node_m, "model": model_name})
                elif elem_type == "Q":
                    n_c = int(parts[1]); n_b = int(parts[2]); n_e = int(parts[3]); model_name = parts[4]
                    elements.append({"type": "BJT", "name": elem_name,
                                     "nc": n_c, "nb": n_b, "ne": n_e, "model": model_name})
                elif elem_type == "M":
                    nd = int(parts[1]); ng = int(parts[2]); ns = int(parts[3]); model_name = parts[4]
                    elements.append({"type": "MOS", "name": elem_name,
                                     "nd": nd, "ng": ng, "ns": ns, "model": model_name})
                else:
                    print(f"Warning: unrecognized element line: {line}")
    return elements, models

###############################################################################
#       NONLINEAR DEVICE MODELS (Newton iteration) - Updated stamps
###############################################################################
def stamp_nonlinear_devices(G, RHS, x, elements, models, node_offset):
    for elem in elements:
        if elem["type"] == "D":
            # Existing diode stamping
            diode_model = models.get(elem["model"], {})
            Is = diode_model.get("params", {}).get("IS", 1e-14)
            Vt = 0.0259
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
        elif elem["type"] == "BJT":
            G, RHS = stamp_nonlinear_bjt(G, RHS, x, elem, models, node_offset)
        elif elem["type"] == "MOS":
            G, RHS = stamp_nonlinear_mos(G, RHS, x, elem, models, node_offset)
    return G, RHS

def stamp_nonlinear_bjt(G, RHS, x, elem, models, node_offset):
    """
    Simplified nonlinear stamp for an NPN BJT.
    Assumes nodes: nc (collector), nb (base), ne (emitter)
    and a simple diode model for the base-emitter junction.
    The collector current is approximated as I_C = alpha_F * I_BE.
    """
    nc = elem["nc"] - 1
    nb = elem["nb"] - 1
    ne = elem["ne"] - 1
    model = models.get(elem["model"], {})
    params = model.get("params", {})
    Is = params.get("IS", 1e-15)
    Vt = 0.0259
    alpha_F = params.get("ALPHA_F", 0.99)
    # Base-emitter junction (diode)
    V_BE = x[nb] - x[ne]
    I_BE = Is * (np.exp(V_BE/Vt) - 1)
    G_BE = (Is/Vt) * np.exp(V_BE/Vt)
    if nb is not None:
        G[nb, nb] += G_BE
    if ne is not None:
        G[ne, ne] += G_BE
    if nb is not None and ne is not None:
        G[nb, ne] -= G_BE
        G[ne, nb] -= G_BE
    mismatch_BE = I_BE - G_BE*V_BE
    if nb is not None:
        RHS[nb] -= mismatch_BE
    if ne is not None:
        RHS[ne] += mismatch_BE
    # Controlled collector current: I_C = alpha_F * I_BE (from collector to emitter)
    I_C = alpha_F * I_BE
    G_C = alpha_F * G_BE
    if nc is not None:
        RHS[nc] -= I_C
    if ne is not None:
        RHS[ne] += I_C
    # Stamp the linearized effect of V_BE on collector current
    if nc is not None and nb is not None:
        G[nc, nb] += G_C
    if nc is not None and ne is not None:
        G[nc, ne] -= G_C
    if ne is not None and nb is not None:
        G[ne, nb] -= G_C
    if ne is not None and ne is not None:
        G[ne, ne] += G_C
    return G, RHS

def stamp_nonlinear_mos(G, RHS, x, elem, models, node_offset):
    """
    Simplified nonlinear stamp for a MOSFET (assumed n-channel by default).
    Uses a square-law saturation model:
      I_D = 0.5 * k * (V_GS - Vth)^2 for V_GS > Vth, else I_D = 0.
    Current flows from drain (nd) to source (ns). V_GS = V(ng) - V(ns).
    """
    nd = elem["nd"] - 1
    ng = elem["ng"] - 1
    ns = elem["ns"] - 1
    model = models.get(elem["model"], {})
    params = model.get("params", {})
    Vth = params.get("VTH", 0.7)
    k = params.get("K", 1e-3)
    V_GS = x[ng] - x[ns]
    if V_GS <= Vth:
        I_D = 0.0
        G_m = 0.0
    else:
        I_D = 0.5 * k * (V_GS - Vth)**2
        G_m = k * (V_GS - Vth)
    if nd is not None:
        RHS[nd] -= I_D
    if ns is not None:
        RHS[ns] += I_D
    # Linearize with respect to V_GS: V_GS = V(ng) - V(ns)
    if ng is not None:
        G[nd, ng] -= G_m
        G[ns, ng] += G_m
    if ns is not None:
        G[nd, ns] += G_m
        G[ns, ns] -= G_m
    return G, RHS

###############################################################################
#                  NONLINEAR SOLVER (Newton iteration)
###############################################################################
def newton_solve_nonlinear(elements, models, num_nodes, max_iter=50, tol=1e-9):
    x = np.zeros(num_nodes, dtype=float)
    for it in range(max_iter):
        G = np.zeros((num_nodes, num_nodes), dtype=float)
        RHS = np.zeros(num_nodes, dtype=float)
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
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis("off")
    plt.title(title)
    plt.text(0.01, 0.95, text_data, ha='left', va='top', wrap=True,
             family='monospace', fontsize=8)
    return fig

###############################################################################
#            NEW FUNCTIONS FOR DC & TRANSIENT ANALYSIS
###############################################################################
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
    for elem in elements:
        t = elem["type"]
        if t == "R":
            G = conductance_stamp_custom(G, elem["n1"], elem["n2"], 1.0/elem["value"])
        elif t == "C":
            pass
        elif t == "L":
            G = conductance_stamp_custom(G, elem["n1"], elem["n2"], 1e9)
        elif t == "V":
            G, RHS, unknowns, idx = voltage_source_stamp_custom_dc(G, RHS, unknowns, elem["n1"], elem["n2"], elem["value"])
            elem["branch_idx"] = idx
        elif t == "I":
            RHS = current_source_stamp_custom(RHS, elem["n1"], elem["n2"], elem["value"])
        elif t == "VCCS":
            G = vccs_stamp_custom(G, elem["n_out_p"], elem["n_out_m"], elem["n_in_p"], elem["n_in_m"], elem["gm"])
        elif t == "VCVS":
            G, RHS, unknowns, idx = vcvs_stamp_custom_dc(G, RHS, unknowns, elem["n_out_p"], elem["n_out_m"], elem["n_in_p"], elem["n_in_m"], elem["gain"])
            elem["branch_idx"] = idx
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
                G, RHS, unknowns, idx = ccvs_stamp_custom_dc(G, RHS, unknowns, elem["n_out_p"], elem["n_out_m"], controlling_branch, elem["transres"])
                elem["branch_idx"] = idx
            else:
                print(f"Warning: CCVS {elem['name']} controlling branch not found.")
        else:
            pass
    return G, RHS, unknowns

def run_dc_analysis(elements, num_nodes):
    G_dc = np.zeros((num_nodes, num_nodes), dtype=complex)
    RHS_dc = np.zeros(num_nodes, dtype=complex)
    unknowns_dc = [f"V{i}" for i in range(1, num_nodes+1)]
    G_dc, RHS_dc, unknowns_dc = stamp_dc_elements(elements, G_dc, RHS_dc, unknowns_dc)
    x_dc = np.linalg.solve(G_dc, RHS_dc)
    return x_dc, unknowns_dc

# ------------------- UPDATED TRANSIENT ANALYSIS -------------------
def run_transient_analysis(elements, tmax=1e-3, dt=1e-6):
    """
    Perform a transient analysis using backward Euler.
    This updated version stamps voltage sources (using the DC stamping routine)
    so that the forcing voltage is included in the nodal equations.
    """
    # Determine maximum node number from all elements.
    max_node = 0
    for e in elements:
        for key in ['n1','n2','n_p','n_m']:
            if key in e and isinstance(e[key], int) and e[key] > max_node:
                max_node = e[key]
    num_nodes = max_node
    # Build transient matrices for resistive (G_t) and capacitive (C_t) components,
    # and initialize RHS.
    G_t = np.zeros((num_nodes, num_nodes), dtype=float)
    C_t = np.zeros((num_nodes, num_nodes), dtype=float)
    RHS_t = np.zeros(num_nodes, dtype=float)
    unknowns = [f"V{i}" for i in range(1, num_nodes+1)]
    # Stamp each element.
    for e in elements:
        t = e["type"]
        if t == "R":
            G_t = conductance_stamp_custom(G_t, e["n1"], e["n2"], 1.0/float(e["value"]))
        elif t == "C":
            C_t = capacitance_stamp_custom(C_t, e["n1"], e["n2"], float(e["value"]))
        elif t == "L":
            G_t = conductance_stamp_custom(G_t, e["n1"], e["n2"], 1e9)
        elif t == "I":
            RHS_t = current_source_stamp_custom(RHS_t, e["n1"], e["n2"], float(e["value"]))
        elif t == "V":
            G_t, RHS_t, unknowns, idx = voltage_source_stamp_custom_dc(G_t, RHS_t, unknowns, e["n1"], e["n2"], e["value"])
            C_t = np.pad(C_t, ((0,1),(0,1)), mode='constant', constant_values=0)
            e["branch_idx"] = idx
        # Dependent sources and nonlinear devices are not handled here.
    new_dim = G_t.shape[0]
    steps = int(tmax/dt)
    x_transient = np.zeros((steps, new_dim))
    x_prev = np.zeros(new_dim)
    for i in range(steps):
        A = G_t + C_t/dt
        b = RHS_t + (C_t/dt).dot(x_prev)
        x_new = np.linalg.solve(A, b)
        x_transient[i, :] = x_new
        x_prev = x_new
    time_points = np.linspace(dt, tmax, steps)
    return time_points, x_transient, unknowns

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
    m_out = nom-1 if nop>0 else None
    p_in = nip-1 if nip>0 else None
    m_in = nim-1 if nim>0 else None
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
    m_out = nom-1 if nop>0 else None
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
#            ITERATIVE SOLVER METHODS (New additions)
###############################################################################
class IterationCounter:
    def __init__(self):
        self.niter = 0
    def __call__(self, xk):
        self.niter += 1

def iterative_cg(A, b, tol=1e-9, maxiter=1000):
    counter = IterationCounter()
    x, info = cg(A, b, tol=tol, maxiter=maxiter, callback=counter)
    return x, info, counter.niter

def iterative_gmres(A, b, tol=1e-9, maxiter=1000):
    counter = IterationCounter()
    x, info = gmres(A, b, tol=tol, maxiter=maxiter, callback=counter)
    return x, info, counter.niter

def iterative_bicgstab(A, b, tol=1e-9, maxiter=1000):
    counter = IterationCounter()
    x, info = bicgstab(A, b, tol=tol, maxiter=maxiter, callback=counter)
    return x, info, counter.niter

def iterative_minres(A, b, tol=1e-9, maxiter=1000):
    counter = IterationCounter()
    x, info = minres(A, b, tol=tol, maxiter=maxiter, callback=counter)
    return x, info, counter.niter

def iterative_jacobi(A, b, tol=1e-9, maxiter=1000):
    n = A.shape[0]
    x = np.zeros(n)
    D = np.diag(np.diag(A))
    R = A - D
    D_inv = np.diag(1/np.diag(A))
    iter_count = 0
    for i in range(maxiter):
        x_new = D_inv.dot(b - R.dot(x))
        iter_count += 1
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    return x, 0, iter_count

def iterative_sor(A, b, tol=1e-9, maxiter=1000, omega=1.5):
    n = A.shape[0]
    x = np.zeros(n, dtype=A.dtype)
    iter_count = 0
    for i in range(maxiter):
        x_new = np.copy(x)
        for j in range(n):
            sigma = 0
            for k in range(n):
                if k != j:
                    sigma += A[j, k] * (x_new[k] if k < j else x[k])
            x_new[j] = (1 - omega) * x[j] + (omega / A[j, j]) * (b[j] - sigma)
        iter_count += 1
        if np.linalg.norm(x_new - x) < tol:
            return x_new, 0, iter_count
        x = x_new
    return x, 1, iter_count

def iterative_multigrid(A, b, tol=1e-9, maxiter=100, pre_smooth=3, post_smooth=3):
    # Simplified placeholder: using Jacobi as a surrogate multigrid.
    x, info, niter = iterative_jacobi(A, b, tol=tol, maxiter=maxiter)
    return x, info, niter

def iterative_ilu_gmres(A, b, tol=1e-9, maxiter=1000):
    A_sparse = csr_matrix(A)
    ilu = spilu(A_sparse)
    Mx = lambda x: ilu.solve(x)
    M = LinearOperator(A.shape, Mx)
    counter = IterationCounter()
    x, info = gmres(A, b, tol=tol, maxiter=maxiter, M=M, callback=counter)
    return x, info, counter.niter

def sparse_direct_solver(A, b):
    A_sparse = csr_matrix(A)
    x = spsolve(A_sparse, b)
    return x

def compute_memory_usage_dense(A, b):
    return A.nbytes + b.nbytes

def iterative_solver_analysis(method_name, solver_function, A, b, x_direct, tol=1e-9, maxiter=1000):
    start = time.time()
    x_iter, info, niter = solver_function(A, b, tol=tol, maxiter=maxiter)
    end = time.time()
    time_taken = end - start
    rel_error = np.linalg.norm(x_iter - x_direct) / np.linalg.norm(x_direct)
    mem_usage = compute_memory_usage_dense(A, b)
    if method_name.upper() == "CG":
        complexity = "CG: O(n) per iteration for sparse mat-vec products (only for SPD matrices)."
    elif method_name.upper() == "GMRES":
        complexity = "GMRES: Builds a Krylov subspace; may require restart to limit memory."
    elif method_name.upper() == "BICGSTAB":
        complexity = "BiCGSTAB: Stabilized variant of BiCG; effective for non-symmetric systems."
    elif method_name.upper() == "MINRES":
        complexity = "MINRES: Suitable for symmetric (possibly indefinite) matrices."
    elif method_name.upper() == "JACOBI":
        complexity = "Jacobi: Simple iteration; generally slower without acceleration."
    elif method_name.upper() == "SOR":
        complexity = "SOR: Extension of Gauss-Seidel with relaxation; sensitive to omega."
    elif method_name.upper() == "MULTIGRID":
        complexity = "Multigrid: Hierarchical method accelerating error reduction; here a simplified placeholder."
    elif method_name.upper() == "ILU-GMRES":
        complexity = "ILU-GMRES: GMRES with ILU preconditioning; effective for sparse systems."
    elif method_name.upper() == "SPARSEDIRECT":
        complexity = "Sparse Direct: Exploits CSR/CSC formats for efficiency (e.g., UMFPACK)."
    else:
        complexity = "N/A"
    analysis_text = f"Iterative Solver Analysis: {method_name}\n"
    analysis_text += f"--------------------------------------\n"
    analysis_text += f"Time taken: {time_taken:.4e} seconds\n"
    analysis_text += f"Iterations: {niter}\n"
    analysis_text += f"Solver info flag: {info}\n"
    analysis_text += f"Relative error (vs direct solve): {rel_error:.4e}\n"
    analysis_text += f"Estimated memory usage (A and b): {mem_usage} bytes\n"
    analysis_text += f"Implementation Complexity: {complexity}\n"
    return analysis_text

###############################################################################
#                                   MAIN
###############################################################################
if __name__ == "__main__":
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
        for key in e:
            if key.startswith("n") and isinstance(e[key], int):
                if e[key] > max_node:
                    max_node = e[key]
        for key in ["n_out_p", "n_out_m", "n_in_p", "n_in_m"]:
            if key in e and e[key] > max_node:
                max_node = e[key]
    num_nodes = max_node

    # Initialize matrices for AC analysis (as in original)
    G = np.zeros((num_nodes, num_nodes), dtype=complex)
    C = np.zeros((num_nodes, num_nodes), dtype=complex)
    L = np.zeros((num_nodes, num_nodes), dtype=complex)
    RHS = np.zeros(num_nodes, dtype=complex)
    unknowns = [f"V{i}" for i in range(1, num_nodes+1)]

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

    G, C, L, RHS, unknowns = stamp_linear_elements(elements, G, C, L, RHS, unknowns)

    print("\nSelect analysis type:")
    print("1) DC Analysis")
    print("2) AC Analysis")
    print("3) Transient Analysis")
    print("4) Iterative Solver Analysis")
    analysis_choice = input("Enter your choice (1/2/3/4): ").strip()

    if analysis_choice == "1":
        print("\n--- Running DC Analysis ---")
        G_dc = np.zeros((num_nodes, num_nodes), dtype=complex)
        RHS_dc = np.zeros(num_nodes, dtype=complex)
        unknowns_dc = [f"V{i}" for i in range(1, num_nodes+1)]
        G_dc, RHS_dc, unknowns_dc = stamp_dc_elements(elements, G_dc, RHS_dc, unknowns_dc)
        x_dc = np.linalg.solve(G_dc, RHS_dc)
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
        print("\n--- Running AC Analysis ---")
        num_points_input = input("Enter number of frequency points (e.g., 1000): ").strip()
        try:
            num_points = int(num_points_input)
        except:
            print("Invalid number entered. Using default 1000 frequency points.")
            num_points = 1000
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
        print("\n--- Running Transient Analysis ---")
        print("Please enter the total simulation time (tmax) and time step (dt) in seconds.")
        print("For example, enter '1e1' for tmax (which means 10 s) and '1e-2' for dt (0.01 s).")
        tmax_input = input("Enter tmax (e.g., 1e1): ").strip()
        dt_input = input("Enter dt (e.g., 1e-2): ").strip()
        try:
            tmax = float(tmax_input)
            dt = float(dt_input)
        except:
            print("Invalid input for tmax or dt. Using default values: tmax=1e1, dt=1e-2.")
            tmax = 1e1
            dt = 1e-2
        t, x_transient, unknowns_trans = run_transient_analysis(elements, tmax=tmax, dt=dt)
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
    elif analysis_choice == "4":
        print("\n--- Running Iterative Solver Analysis (on DC system) ---")
        G_dc = np.zeros((num_nodes, num_nodes), dtype=complex)
        RHS_dc = np.zeros(num_nodes, dtype=complex)
        unknowns_dc = [f"V{i}" for i in range(1, num_nodes+1)]
        G_dc, RHS_dc, unknowns_dc = stamp_dc_elements(elements, G_dc, RHS_dc, unknowns_dc)
        x_direct = np.linalg.solve(G_dc, RHS_dc)
        print("Select iterative method:")
        print("1) Conjugate Gradient (CG)")
        print("2) GMRES")
        print("3) BiCGSTAB")
        print("4) MINRES")
        print("5) Jacobi")
        print("6) Successive Over-Relaxation (SOR)")
        print("7) Multigrid (simplified placeholder)")
        print("8) ILU-preconditioned GMRES")
        print("9) Sparse Direct Solver (spsolve)")
        iter_choice = input("Enter your choice (1-9): ").strip()
        A = G_dc
        b = RHS_dc
        if iter_choice == "1":
            method_name = "CG"
            solver = iterative_cg
        elif iter_choice == "2":
            method_name = "GMRES"
            solver = iterative_gmres
        elif iter_choice == "3":
            method_name = "BiCGSTAB"
            solver = iterative_bicgstab
        elif iter_choice == "4":
            method_name = "MINRES"
            solver = iterative_minres
        elif iter_choice == "5":
            method_name = "Jacobi"
            solver = iterative_jacobi
        elif iter_choice == "6":
            method_name = "SOR"
            solver = lambda A, b, tol=1e-9, maxiter=1000: iterative_sor(A, b, tol=tol, maxiter=maxiter, omega=1.5)
        elif iter_choice == "7":
            method_name = "Multigrid"
            solver = iterative_multigrid
        elif iter_choice == "8":
            method_name = "ILU-GMRES"
            solver = iterative_ilu_gmres
        elif iter_choice == "9":
            method_name = "SparseDirect"
            solver = lambda A, b, tol=1e-9, maxiter=1000: (sparse_direct_solver(A, b), 0, 0)
        else:
            print("Invalid iterative method selected. Exiting.")
            sys.exit(1)
        analysis_text = iterative_solver_analysis(method_name, solver, A, b, x_direct)
        pdf_filename = f"{method_name.lower()}_analysis.pdf"
        with PdfPages(pdf_filename) as pdf_out:
            fig_analysis = make_text_page(analysis_text, title=f"{method_name} Analysis")
            pdf_out.savefig(fig_analysis)
            plt.close(fig_analysis)
        print(f"Iterative solver analysis for {method_name} saved to {pdf_filename}")
        print("Analysis Summary:")
        print(analysis_text)
    else:
        print("Invalid analysis type selected. Exiting.")
        sys.exit(1)

    print("\n=== Simulation Complete ===")
