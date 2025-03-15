import numpy as np
import time, math, sys, os
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
# STEP 1: TKINTER FILE-DIALOG TO PICK NETLIST
###############################################################################
def pick_netlist_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a Netlist File",
        filetypes=[("Netlist Files", "*.net *.spice *.ckt"), ("All Files", "*.*")]
    )
    return file_path if file_path else None

###############################################################################
# NETLIST PARSER
###############################################################################
def parse_netlist(netlist_file):
    elements = []
    models = {}
    with open(netlist_file, "r", encoding="utf-8") as f:
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
                    elements.append({"type": "R", "name": elem_name,
                                     "n1": int(parts[1]), "n2": int(parts[2]), "value": float(parts[3])})
                elif elem_type == "C":
                    elements.append({"type": "C", "name": elem_name,
                                     "n1": int(parts[1]), "n2": int(parts[2]), "value": float(parts[3])})
                elif elem_type == "L":
                    elements.append({"type": "L", "name": elem_name,
                                     "n1": int(parts[1]), "n2": int(parts[2]), "value": float(parts[3])})
                elif elem_type == "V":
                    elements.append({"type": "V", "name": elem_name,
                                     "n1": int(parts[1]), "n2": int(parts[2]), "value": float(parts[3])})
                elif elem_type == "I":
                    elements.append({"type": "I", "name": elem_name,
                                     "n1": int(parts[1]), "n2": int(parts[2]), "value": float(parts[3])})
                elif elem_type == "G":
                    elements.append({"type": "VCCS", "name": elem_name,
                                     "n_out_p": int(parts[1]), "n_out_m": int(parts[2]),
                                     "n_in_p": int(parts[3]), "n_in_m": int(parts[4]), "gm": float(parts[5])})
                elif elem_type == "E":
                    elements.append({"type": "VCVS", "name": elem_name,
                                     "n_out_p": int(parts[1]), "n_out_m": int(parts[2]),
                                     "n_in_p": int(parts[3]), "n_in_m": int(parts[4]), "gain": float(parts[5])})
                elif elem_type == "F":
                    elements.append({"type": "CCCS", "name": elem_name,
                                     "n_out_p": int(parts[1]), "n_out_m": int(parts[2]),
                                     "branch_name": parts[3], "gain": float(parts[4])})
                elif elem_type == "H":
                    elements.append({"type": "CCVS", "name": elem_name,
                                     "n_out_p": int(parts[1]), "n_out_m": int(parts[2]),
                                     "branch_name": parts[3], "transres": float(parts[4])})
                elif elem_type == "D":
                    elements.append({"type": "D", "name": elem_name,
                                     "n_p": int(parts[1]), "n_m": int(parts[2]), "model": parts[3]})
                elif elem_type == "Q":
                    elements.append({"type": "BJT", "name": elem_name,
                                     "nc": int(parts[1]), "nb": int(parts[2]), "ne": int(parts[3]), "model": parts[4]})
                elif elem_type == "M":
                    elements.append({"type": "MOS", "name": elem_name,
                                     "nd": int(parts[1]), "ng": int(parts[2]), "ns": int(parts[3]), "model": parts[4]})
                else:
                    print(f"Warning: unrecognized element line: {line}")
    return elements, models

###############################################################################
# LINEAR STAMPING FUNCTIONS (for DC/AC/Transient direct analysis)
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
    G_expanded = np.pad(G, ((0,1),(0,1)), mode="constant", constant_values=0)
    C_expanded = np.pad(C, ((0,1),(0,1)), mode="constant", constant_values=0)
    L_expanded = np.pad(L, ((0,1),(0,1)), mode="constant", constant_values=0)
    RHS_expanded = np.pad(RHS, (0,1), mode="constant", constant_values=0)
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
    G_expanded = np.pad(G, ((0,1),(0,1)), mode="constant")
    C_expanded = np.pad(C, ((0,1),(0,1)), mode="constant")
    L_expanded = np.pad(L, ((0,1),(0,1)), mode="constant")
    RHS_expanded = np.pad(RHS, (0,1), mode="constant")
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
    p_out = nop-1 if nop > 0 else None
    m_out = nom-1 if nop > 0 else None
    p_in = nip-1 if nip > 0 else None
    m_in = nim-1 if nim > 0 else None
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
    G_expanded = np.pad(G, ((0,1),(0,1)), mode="constant")
    C_expanded = np.pad(C, ((0,1),(0,1)), mode="constant")
    L_expanded = np.pad(L, ((0,1),(0,1)), mode="constant")
    RHS_expanded = np.pad(RHS, (0,1), mode="constant")
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
    p_out = nop-1 if nop > 0 else None
    m_out = nom-1 if nop > 0 else None
    if p_out is not None:
        G[p_out, branch_idx] += gain
    if m_out is not None:
        G[m_out, branch_idx] -= gain
    return G

def ccvs_stamp_custom(G, C, L, RHS, nop, nom, branch_idx, transres):
    size_old = G.shape[0]
    G_expanded = np.pad(G, ((0,1),(0,1)), mode="constant")
    C_expanded = np.pad(C, ((0,1),(0,1)), mode="constant")
    L_expanded = np.pad(L, ((0,1),(0,1)), mode="constant")
    RHS_expanded = np.pad(RHS, (0,1), mode="constant")
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
# NONLINEAR DEVICE STAMPS (Diode, BJT, MOS)
###############################################################################
def stamp_nonlinear_devices(G, RHS, x, elements, models, node_offset):
    for elem in elements:
        if elem["type"] == "D":
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
            if n_p is not None and n_m is not None:
                G[n_p, n_m] -= Gd; G[n_m, n_p] -= Gd
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
    nc = elem["nc"] - 1
    nb = elem["nb"] - 1
    ne = elem["ne"] - 1
    model = models.get(elem["model"], {})
    params = model.get("params", {})
    Is = params.get("IS", 1e-15)
    Vt = 0.0259
    alpha_F = params.get("ALPHA_F", 0.99)
    V_BE = x[nb] - x[ne]
    I_BE = Is * (np.exp(V_BE/Vt) - 1)
    G_BE = (Is/Vt) * np.exp(V_BE/Vt)
    if nb is not None:
        G[nb, nb] += G_BE
    if ne is not None:
        G[ne, ne] += G_BE
    if nb is not None and ne is not None:
        G[nb, ne] -= G_BE; G[ne, nb] -= G_BE
    mismatch_BE = I_BE - G_BE*V_BE
    if nb is not None:
        RHS[nb] -= mismatch_BE
    if ne is not None:
        RHS[ne] += mismatch_BE
    I_C = alpha_F * I_BE
    G_C = alpha_F * G_BE
    if nc is not None:
        RHS[nc] -= I_C
    if ne is not None:
        RHS[ne] += I_C
    if nc is not None and nb is not None:
        G[nc, nb] += G_C
    if nc is not None and ne is not None:
        G[nc, ne] -= G_C
    if ne is not None and nb is not None:
        G[ne, nb] -= G_C
    if ne is not None:
        G[ne, ne] += G_C
    return G, RHS

def stamp_nonlinear_mos(G, RHS, x, elem, models, node_offset):
    nd = elem["nd"] - 1
    ng = elem["ng"] - 1
    ns = elem["ns"] - 1
    model = models.get(elem["model"], {})
    params = model.get("params", {})
    Vth = params.get("VTH", 0.7)
    k = params.get("K", 1e-3)
    V_GS = x[ng] - x[ns]
    if V_GS <= Vth:
        I_D = 0.0; G_m = 0.0
    else:
        I_D = 0.5 * k * (V_GS - Vth)**2
        G_m = k * (V_GS - Vth)
    if nd is not None:
        RHS[nd] -= I_D
    if ns is not None:
        RHS[ns] += I_D
    if ng is not None:
        G[nd, ng] -= G_m; G[ns, ng] += G_m
    if ns is not None:
        G[nd, ns] += G_m; G[ns, ns] -= G_m
    return G, RHS

###############################################################################
# NONLINEAR SOLVER (Newton iteration for DC analysis)
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
# FREQUENCY SWEEP & PDF OUTPUT (AC analysis)
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
    plt.text(0.01, 0.95, text_data, ha="left", va="top", wrap=True,
             family="monospace", fontsize=8)
    return fig

###############################################################################
# NEW FUNCTIONS FOR DC & TRANSIENT ANALYSIS (Direct solver version)
###############################################################################
def voltage_source_stamp_custom_dc(G, RHS, unknowns, n1, n2, volt):
    size_old = G.shape[0]
    G_expanded = np.pad(G, ((0,1),(0,1)), mode="constant", constant_values=0)
    RHS_expanded = np.pad(RHS, (0,1), mode="constant", constant_values=0)
    idx = size_old
    if n1 > 0:
        G_expanded[n1-1, idx] += 1; G_expanded[idx, n1-1] += 1
    if n2 > 0:
        G_expanded[n2-1, idx] -= 1; G_expanded[idx, n2-1] -= 1
    RHS_expanded[idx] = volt
    unknowns.append(f"I_V_{idx}")
    return G_expanded, RHS_expanded, unknowns, idx

def vcvs_stamp_custom_dc(G, RHS, unknowns, n_out_p, n_out_m, n_in_p, n_in_m, gain):
    size_old = G.shape[0]
    G_expanded = np.pad(G, ((0,1),(0,1)), mode="constant", constant_values=0)
    RHS_expanded = np.pad(RHS, (0,1), mode="constant", constant_values=0)
    idx = size_old
    if n_out_p > 0:
        G_expanded[n_out_p-1, idx] += 1; G_expanded[idx, n_out_p-1] += 1
    if n_out_m > 0:
        G_expanded[n_out_m-1, idx] -= 1; G_expanded[idx, n_out_m-1] -= 1
    if n_in_p > 0:
        G_expanded[idx, n_in_p-1] -= gain
    if n_in_m > 0:
        G_expanded[idx, n_in_m-1] += gain
    unknowns.append(f"I_E_{idx}")
    return G_expanded, RHS_expanded, unknowns, idx

def ccvs_stamp_custom_dc(G, RHS, unknowns, n_out_p, n_out_m, branch_idx, transres):
    size_old = G.shape[0]
    G_expanded = np.pad(G, ((0,1),(0,1)), mode="constant", constant_values=0)
    RHS_expanded = np.pad(RHS, (0,1), mode="constant", constant_values=0)
    idx = size_old
    if n_out_p > 0:
        G_expanded[n_out_p-1, idx] += 1; G_expanded[idx, n_out_p-1] += 1
    if n_out_m > 0:
        G_expanded[n_out_m-1, idx] -= 1; G_expanded[idx, n_out_m-1] -= 1
    G_expanded[idx, branch_idx] -= transres
    unknowns.append(f"I_H_{idx}")
    return G_expanded, RHS_expanded, unknowns, idx

def stamp_dc_elements(elements, G, RHS, unknowns):
    for elem in elements:
        t = elem["type"]
        # Only stamp linear devices here
        if t == "R":
            G = conductance_stamp_custom(G, elem["n1"], elem["n2"], 1.0/elem["value"])
        elif t == "C":
            # Nonlinear; skip in linear stamping
            print(f"Note: Nonlinear element {elem['name']} of type 'C' skipped in DC linear stamping.")
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
        elif t in ["CCCS", "CCVS", "D", "BJT", "MOS"]:
            print(f"Note: Nonlinear or dependent element {elem['name']} of type '{t}' skipped in DC linear stamping. Use Nonlinear DC Analysis (option 5) instead.")
        else:
            pass
    return G, RHS, unknowns

###############################################################################
# TRANSIENT ANALYSIS (Dense Direct Solver)
###############################################################################
def run_transient_analysis(elements, tmax=1e-3, dt=1e-6):
    max_node = 0
    for e in elements:
        for key in ['n1','n2','n_p','n_m']:
            if key in e and isinstance(e[key], int):
                max_node = max(max_node, e[key])
    num_nodes = max_node
    G_t = np.zeros((num_nodes, num_nodes), dtype=float)
    C_t = np.zeros((num_nodes, num_nodes), dtype=float)
    RHS_t = np.zeros(num_nodes, dtype=float)
    unknowns = [f"V{i}" for i in range(1, num_nodes+1)]
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
            C_t = np.pad(C_t, ((0,1),(0,1)), mode="constant", constant_values=0)
            e["branch_idx"] = idx
        # Dependent and nonlinear elements are skipped here.
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
    t_points = np.linspace(dt, tmax, steps)
    return t_points, x_transient, unknowns

###############################################################################
# LINEAR STAMPING (AC analysis) – Original (Dense)
###############################################################################
# (conductance_stamp_custom, capacitance_stamp_custom, current_source_stamp_custom,
#  voltage_source_stamp_custom, inductor_stamp_custom, vccs_stamp_custom, vcvs_stamp_custom,
#  cccs_stamp_custom, ccvs_stamp_custom are defined above)

###############################################################################
# ITERATIVE SOLVER METHODS
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
    # Simplified placeholder: use Jacobi as a surrogate.
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
        complexity = "Multigrid: Hierarchical method; here a simplified placeholder using Jacobi."
    elif method_name.upper() == "ILU-GMRES":
        complexity = "ILU-GMRES: GMRES with ILU preconditioning; effective for sparse systems."
    elif method_name.upper() == "SPARSEDIRECT":
        complexity = "Sparse Direct: Uses CSR/CSC formats with spsolve."
    else:
        complexity = "N/A"
    analysis_text = f"Iterative Solver Analysis: {method_name}\n"
    analysis_text += "--------------------------------------\n"
    analysis_text += f"Time taken: {time_taken:.4e} seconds\n"
    analysis_text += f"Iterations: {niter}\n"
    analysis_text += f"Solver info flag: {info}\n"
    analysis_text += f"Relative error (vs direct solve): {rel_error:.4e}\n"
    analysis_text += f"Estimated memory usage (A and b): {mem_usage} bytes\n"
    analysis_text += f"Implementation Complexity: {complexity}\n"
    return analysis_text

###############################################################################
# MAIN MENU
###############################################################################
def main():
    netlist_file = pick_netlist_file()
    if not netlist_file:
        print("No netlist file selected. Exiting.")
        sys.exit(0)
    if not os.path.isfile(netlist_file):
        print(f"Error: netlist file '{netlist_file}' not found.")
        sys.exit(1)
    print(f"Loading netlist: {netlist_file}")
    elements, models = parse_netlist(netlist_file)
    # Determine highest node number
    max_node = 0
    for e in elements:
        for key in e:
            if key.startswith("n") and isinstance(e[key], int):
                max_node = max(max_node, e[key])
        for key in ["n_out_p", "n_out_m", "n_in_p", "n_in_m"]:
            if key in e:
                max_node = max(max_node, e[key])
    num_nodes = max_node

    # Initialize matrices for AC analysis
    G = np.zeros((num_nodes, num_nodes), dtype=complex)
    C = np.zeros((num_nodes, num_nodes), dtype=complex)
    L = np.zeros((num_nodes, num_nodes), dtype=complex)
    RHS = np.zeros(num_nodes, dtype=complex)
    unknowns = [f"V{i}" for i in range(1, num_nodes+1)]

    # Stamp linear elements (only linear devices are stamped here)
    def stamp_linear_elements_wrapper(elements, G, C, L, RHS, unknowns):
        # For linear stamping, skip nonlinear devices and dependent sources that require special handling.
        new_unknowns = unknowns.copy()
        for elem in elements:
            t = elem["type"]
            if t in ["D", "BJT", "MOS"]:
                print(f"Note: Nonlinear element {elem['name']} of type '{t}' skipped in linear stamping. Use Nonlinear DC Analysis (option 5).")
                continue
            # Otherwise, stamp as usual:
            if t == "R":
                G = conductance_stamp_custom(G, elem["n1"], elem["n2"], 1.0/elem["value"])
            elif t == "C":
                C = capacitance_stamp_custom(C, elem["n1"], elem["n2"], elem["value"])
            elif t == "L":
                G, C, L, RHS, idx = inductor_stamp_custom(G, C, L, RHS, elem["n1"], elem["n2"], elem["value"])
                new_unknowns.append(elem["name"]+"_I")
                elem["branch_idx"] = idx
            elif t == "V":
                G, C, L, RHS, idx = voltage_source_stamp_custom(G, C, L, RHS, elem["n1"], elem["n2"], elem["value"])
                new_unknowns.append(elem["name"]+"_I")
                elem["branch_idx"] = idx
            elif t == "I":
                RHS = current_source_stamp_custom(RHS, elem["n1"], elem["n2"], elem["value"])
            elif t == "VCCS":
                G = vccs_stamp_custom(G, elem["n_out_p"], elem["n_out_m"], elem["n_in_p"], elem["n_in_m"], elem["gm"])
            elif t == "VCVS":
                G, C, L, RHS, idx = vcvs_stamp_custom(G, C, L, RHS, elem["n_out_p"], elem["n_out_m"], elem["n_in_p"], elem["n_in_m"], elem["gain"])
                new_unknowns.append(elem["name"]+"_I")
                elem["branch_idx"] = idx
            elif t in ["CCCS", "CCVS"]:
                # Dependent sources: if controlling branch is not found, print a warning.
                controlling_branch = None
                for e2 in elements:
                    if e2.get("name") == elem.get("branch_name"):
                        controlling_branch = e2.get("branch_idx", None)
                if controlling_branch is not None:
                    if t == "CCCS":
                        G = cccs_stamp_custom(G, elem["n_out_p"], elem["n_out_m"], controlling_branch, elem["gain"])
                    else:
                        G, C, L, RHS, idx = ccvs_stamp_custom(G, C, L, RHS, elem["n_out_p"], elem["n_out_m"], controlling_branch, elem["transres"])
                        new_unknowns.append(elem["name"]+"_I")
                        elem["branch_idx"] = idx
                else:
                    print(f"Warning: {t} {elem['name']} controlling branch not found.")
            else:
                print(f"Skipping unhandled element '{elem['name']}' of type '{t}'.")
        return G, C, L, RHS, new_unknowns

    G, C, L, RHS, unknowns = stamp_linear_elements_wrapper(elements, G, C, L, RHS, unknowns)

    print("\nSelect analysis type:")
    print("1) DC Analysis (Direct Linear)")
    print("2) AC Analysis")
    print("3) Transient Analysis")
    print("4) Iterative Solver Analysis (on DC linear system)")
    print("5) Nonlinear DC Analysis (Newton iteration)")
    analysis_choice = input("Enter your choice (1/2/3/4/5): ").strip()

    if analysis_choice == "1":
        print("\n--- Running DC Analysis (Direct Linear) ---")
        G_dc = np.zeros((num_nodes, num_nodes), dtype=complex)
        RHS_dc = np.zeros(num_nodes, dtype=complex)
        unknowns_dc = [f"V{i}" for i in range(1, num_nodes+1)]
        G_dc, RHS_dc, unknowns_dc = stamp_dc_elements(elements, G_dc, RHS_dc, unknowns_dc)
        x_dc = np.linalg.solve(G_dc, RHS_dc)
        with open("dc_solution.csv", "w") as f_out:
            f_out.write("Unknown,Voltage\n")
            for i, val in enumerate(x_dc):
                f_out.write(f"{unknowns_dc[i]},{val.real}\n")
        print("DC solution saved to dc_solution.csv")
    elif analysis_choice == "2":
        print("\n--- Running AC Analysis ---")
        num_points = int(input("Enter number of frequency points (e.g., 1000): ").strip())
        complex_choice = input("Is your AC circuit complex (use iterative solver)? (yes/no): ").strip().lower()
        freqs = np.logspace(np.log10(1), np.log10(1e9), num_points)
        n_size = G.shape[0]
        x_solutions = np.zeros((num_points, n_size), dtype=complex)
        total_flops = 0
        start_time_all = time.time()
        if complex_choice.startswith("y"):
            # Convert matrices to sparse
            G_sparse = csr_matrix(G)
            C_sparse = csr_matrix(C)
            L_sparse = csr_matrix(L)
            print("Select iterative method for AC analysis:")
            print("1) CG, 2) GMRES, 3) BiCGSTAB, 4) MINRES, 5) Jacobi, 6) SOR, 7) Multigrid, 8) ILU-GMRES, 9) Sparse Direct")
            iter_choice = input("Enter your choice (1-9): ").strip()
            methods = {
                "1": iterative_cg,
                "2": iterative_gmres,
                "3": iterative_bicgstab,
                "4": iterative_minres,
                "5": iterative_jacobi,
                "6": lambda A, b, **kw: iterative_sor(A, b, omega=1.5, **kw),
                "7": iterative_multigrid,
                "8": iterative_ilu_gmres,
                "9": lambda A, b, **kw: (sparse_direct_solver(A, b), 0, 0)
            }
            solver = methods.get(iter_choice, None)
            if solver is None:
                print("Invalid choice; using direct solver.")
            for i, f in enumerate(tqdm(freqs, desc="AC Frequency Sweep")):
                omega = 2 * np.pi * f
                s = 1j * omega
                M = G + s * C + s * L
                if solver:
                    A_sparse = csr_matrix(M)
                    b = np.array(RHS)
                    x, info, niter = solver(A_sparse, b)
                else:
                    x = np.linalg.solve(M, RHS)
                x_solutions[i, :] = x
                total_flops += (2.0/3.0)*(n_size**3) + 2.0*(n_size**2)
        else:
            for i, f in enumerate(tqdm(freqs, desc="AC Frequency Sweep")):
                omega = 2 * np.pi * f
                s = 1j * omega
                M = G + s * C + s * L
                x = np.linalg.solve(M, RHS)
                x_solutions[i, :] = x
                total_flops += (2.0/3.0)*(n_size**3) + 2.0*(n_size**2)
        elapsed_time_all = time.time() - start_time_all
        with open("ac_solution.csv", "w") as f_out:
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
        print("AC solution saved to ac_solution.csv")
        node_count = sum(1 for n in unknowns if n.startswith("V"))
        mag = np.abs(x_solutions[:, :node_count])
        pha = np.angle(x_solutions[:, :node_count], deg=True)
        with PdfPages("ac_plots.pdf") as pdfp:
            plt.figure()
            for i in range(node_count):
                plt.semilogx(freqs, mag[:, i], label=unknowns[i])
            plt.xlabel("Frequency [Hz]"); plt.ylabel("Magnitude"); plt.title("AC Magnitude")
            plt.legend(); plt.grid(True, which="both")
            pdfp.savefig(); plt.close()
            plt.figure()
            for i in range(node_count):
                plt.semilogx(freqs, pha[:, i], label=unknowns[i])
            plt.xlabel("Frequency [Hz]"); plt.ylabel("Phase [deg]"); plt.title("AC Phase")
            plt.legend(); plt.grid(True, which="both")
            pdfp.savefig(); plt.close()
        with PdfPages("debug_and_flops.pdf") as pdfd:
            page1_text = matrix_to_string(G, "G") + matrix_to_string(C, "C") + matrix_to_string(L, "L") + vector_to_string(RHS, "RHS")
            page1_text += f"\nUnknowns:\n{unknowns}\n"
            page1_text += f"\nM_debug at f={f_debug} Hz:\n{np.array_str(M_debug, precision=4, suppress_small=True)}\n"
            fig1 = make_text_page(page1_text, "AC Debug Matrices")
            pdfd.savefig(fig1); plt.close(fig1)
            page2_text = f"Total FLOPs: {(total_flops):.4e}\nElapsed Time: {elapsed_time_all:.2f} sec"
            fig2 = make_text_page(page2_text, "AC Timing & FLOPs")
            pdfd.savefig(fig2); plt.close(fig2)
        print("AC debug info saved to debug_and_flops.pdf")
    elif analysis_choice == "3":
        print("\n--- Running Transient Analysis ---")
        tmax = float(input("Enter tmax (e.g., 1e1): ").strip())
        dt = float(input("Enter dt (e.g., 1e-2): ").strip())
        t, x_transient, unknowns_trans = run_transient_analysis(elements, tmax=tmax, dt=dt)
        use_iterative = input("Use iterative solver with warm starting for transient analysis? (yes/no): ").strip().lower()
        if use_iterative.startswith("y"):
            print("Select iterative method for Transient Analysis:")
            print("1) GMRES, 2) BiCGSTAB, 3) Jacobi, 4) SOR, 5) ILU-GMRES, 6) Sparse Direct")
            iter_choice = input("Enter your choice (1-6): ").strip()
            transient_methods = {
                "1": iterative_gmres,
                "2": iterative_bicgstab,
                "3": iterative_jacobi,
                "4": lambda A, b, **kw: iterative_sor(A, b, omega=1.5, **kw),
                "5": iterative_ilu_gmres,
                "6": lambda A, b, **kw: (sparse_direct_solver(A, b), 0, 0)
            }
            solver_trans = transient_methods.get(iter_choice, None)
            if solver_trans is None:
                print("Invalid choice; using direct solver.")
            new_dim = x_transient.shape[1]
            x_transient_iter = np.zeros((len(t), new_dim))
            # Rebuild system matrices for transient (same as in run_transient_analysis)
            G_t = np.zeros((num_nodes, num_nodes), dtype=float)
            C_t = np.zeros((num_nodes, num_nodes), dtype=float)
            RHS_t = np.zeros((num_nodes), dtype=float)
            unknowns_temp = [f"V{i}" for i in range(1, num_nodes+1)]
            for e in elements:
                t_e = e["type"]
                if t_e == "R":
                    G_t = conductance_stamp_custom(G_t, e["n1"], e["n2"], 1.0/float(e["value"]))
                elif t_e == "C":
                    C_t = capacitance_stamp_custom(C_t, e["n1"], e["n2"], float(e["value"]))
                elif t_e == "L":
                    G_t = conductance_stamp_custom(G_t, e["n1"], e["n2"], 1e9)
                elif t_e == "I":
                    RHS_t = current_source_stamp_custom(RHS_t, e["n1"], e["n2"], float(e["value"]))
                elif t_e == "V":
                    G_t, RHS_t, unknowns_temp, idx = voltage_source_stamp_custom_dc(G_t, RHS_t, unknowns_temp, e["n1"], e["n2"], e["value"])
                    C_t = np.pad(C_t, ((0,1),(0,1)), mode="constant", constant_values=0)
                    e["branch_idx"] = idx
            new_dim = G_t.shape[0]
            x_prev = np.zeros(new_dim)
            stable = True
            for i, t_val in enumerate(t):
                A = G_t + C_t/dt
                b = RHS_t + (C_t/dt).dot(x_prev)
                x_iter, info, niter = solver_trans(A, b, tol=1e-9, maxiter=1000)
                if info != 0:
                    stable = False
                x_transient_iter[i, :] = x_iter
                x_prev = x_iter
            x_transient = x_transient_iter
            stability_note = "Stability issues detected during iterative transient analysis." if not stable else "No stability issues detected."
            report_text = f"Transient Analysis Report\n tmax: {tmax}, dt: {dt}\n{stability_note}"
        else:
            report_text = "Transient Analysis using direct dense solver."
        with open("transient_solution.csv", "w") as f_out:
            header = "Time," + ",".join(unknowns_trans)
            f_out.write(header+"\n")
            for i, t_val in enumerate(t):
                row = f"{t_val}," + ",".join(str(val) for val in x_transient[i, :])
                f_out.write(row+"\n")
        plt.figure()
        for i in range(sum(1 for n in unknowns_trans if n.startswith("V"))):
            plt.plot(t, x_transient[:, i], label=unknowns_trans[i])
        plt.xlabel("Time (s)"); plt.ylabel("Voltage (V)")
        plt.title("Transient Node Voltages")
        plt.grid(True); plt.legend()
        plt.savefig("transient_plot.pdf"); plt.close()
        with PdfPages("transient_analysis_report.pdf") as pdf_t:
            fig_report = make_text_page(report_text, "Transient Analysis Report")
            pdf_t.savefig(fig_report); plt.close(fig_report)
        print("Transient solution and report saved.")
    elif analysis_choice == "4":
        print("\n--- Running Iterative Solver Analysis on DC system ---")
        G_dc = np.zeros((num_nodes, num_nodes), dtype=complex)
        RHS_dc = np.zeros(num_nodes, dtype=complex)
        unknowns_dc = [f"V{i}" for i in range(1, num_nodes+1)]
        G_dc, RHS_dc, unknowns_dc = stamp_dc_elements(elements, G_dc, RHS_dc, unknowns_dc)
        x_direct = np.linalg.solve(G_dc, RHS_dc)
        print("Select iterative method for DC system:")
        print("1) CG, 2) GMRES, 3) BiCGSTAB, 4) MINRES, 5) Jacobi, 6) SOR, 7) Multigrid, 8) ILU-GMRES, 9) Sparse Direct")
        iter_choice = input("Enter your choice (1-9): ").strip()
        methods = {
            "1": iterative_cg,
            "2": iterative_gmres,
            "3": iterative_bicgstab,
            "4": iterative_minres,
            "5": iterative_jacobi,
            "6": lambda A, b, **kw: iterative_sor(A, b, tol=kw.get('tol',1e-9), maxiter=kw.get('maxiter',1000), omega=1.5),
            "7": iterative_multigrid,
            "8": iterative_ilu_gmres,
            "9": lambda A, b, **kw: (sparse_direct_solver(A, b), 0, 0)
        }
        solver = methods.get(iter_choice, None)
        if solver is None:
            print("Invalid iterative method selected. Exiting.")
            sys.exit(1)
        A = G_dc; b = RHS_dc
        analysis_text = iterative_solver_analysis(iter_choice, solver, A, b, x_direct)
        pdf_filename = f"dc_iterative_analysis_{iter_choice}.pdf"
        with PdfPages(pdf_filename) as pdf_out:
            fig_analysis = make_text_page(analysis_text, title=f"DC Iterative Analysis Method {iter_choice}")
            pdf_out.savefig(fig_analysis); plt.close(fig_analysis)
        print(f"DC iterative analysis saved to {pdf_filename}")
        print("Analysis Summary:")
        print(analysis_text)
    elif analysis_choice == "5":
        print("\n--- Running Nonlinear DC Analysis (Newton iteration) ---")
        # Nonlinear stamping: use all nonlinear devices
        x_nl = newton_solve_nonlinear(elements, models, num_nodes)
        with open("nonlinear_dc_solution.csv", "w") as f_out:
            f_out.write("Node,Voltage\n")
            for i, v in enumerate(x_nl):
                f_out.write(f"V{i+1},{v}\n")
        print("Nonlinear DC solution saved to nonlinear_dc_solution.csv")
    else:
        print("Invalid analysis type selected. Exiting.")
        sys.exit(1)

    print("\n=== Simulation Complete ===")

if __name__ == "__main__":
    main()
