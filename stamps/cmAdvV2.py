import numpy as np
import time
import math
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

###############################################################################
#                           NETLIST PARSER
###############################################################################
def parse_netlist(netlist_file):
    """
    Very simplistic netlist parser example:

    Sample netlist lines might look like:
      R1 1 2 100
      C1 2 0 1e-6
      L1 2 0 1e-3
      V1 4 0 10
      I1 5 3 0.02
      G1 2 4 1 3 0.1   (VCCS => node_out_p=2, node_out_m=4, node_in_p=1, node_in_m=3, gm=0.1)
      E1 1 5 2 4 2.0   (VCVS => out=(1,5), in=(2,4), gain=2.0)
      F1 3 4 VS1 1.5   (CCCS => i_out=1.5*i(VS1). We'll need to store references.)
      H1 2 5 F1 0.5    (CCVS => out=(2,5), controlling current=F1, transres=0.5)
      D1 3 2 diodeModel
      Q1 c b e NPN
      M1 d g s MOSModel
      .model diodeModel D IS=1e-14 ...
      .model NPN BJT ...
      .model MOSModel MOS ...
      .end

    We'll store a list of 'elements' and a dict of 'models'.
    """
    elements = []
    models = {}

    with open(netlist_file, "r") as f:
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
                # Heuristics:
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
                    # H# node_out_p node_out_m branch_name transres
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
                    # unknown element
                    print(f"Warning: unrecognized element line: {line}")

    return elements, models

###############################################################################
#                  NONLINEAR DEVICE MODELS (Newton iteration)
###############################################################################
def stamp_nonlinear_devices(G, RHS, x, elements, models, node_offset):
    """
    For DC or large-signal MNA:
      1) x includes node voltages + branch currents for linear elements.
      2) We find node voltages and update the diode/BJT/MOS equations and partial derivatives.
      3) We do an incremental stamp: G, RHS changed based on the linearization.
         G, RHS are typically "reset" each Newton iteration to partial derivatives.

    This is just a stub showing how you might handle diodes, BJTs, MOSFETs.
    - 'node_offset' is 1 if 'x[0]' = voltage at node1, etc.
    """
    # For each device, we compute i_device and conductance dI/dV for each node pin.
    # Then we add i_device - g_device*(v_pin) to RHS and G.

    # Example diode eqn (shockley): I = Is*(exp(Vd/Vt)-1). We approximate around Vd0:
    #   I_lin = I(Vd0) + Gd*(Vd - Vd0),  Gd = dI/dVd at Vd0
    #   stamp +Gd in G[node_p,node_p], +Gd in G[node_m,node_m], -Gd in cross terms.
    #   and fix RHS so total eqn is KCL => i_d flows from p->m.

    # The details for BJT, MOS are more complicated and typically require additional
    # internal node references or specialized stamping. This is just a minimal example.

    # For brevity, not implementing a full solution.
    # We'll just do an empty function or a simple diode example.

    for elem in elements:
        if elem["type"] == "D":
            # Suppose the diode model has "IS" from models
            diode_model = models.get(elem["model"], {})
            Is = diode_model.get("params", {}).get("IS", 1e-14)
            Vt = 0.0259  # thermal voltage approx

            # node indices
            n_p = elem["n_p"] - 1 if elem["n_p"] > 0 else None
            n_m = elem["n_m"] - 1 if elem["n_m"] > 0 else None

            # diode voltage Vd = v_p - v_m
            Vp = x[n_p] if n_p is not None else 0
            Vm = x[n_m] if n_m is not None else 0
            Vd = Vp - Vm

            # diode I, dI/dV
            Id = Is * (np.exp(Vd / Vt) - 1)
            Gd = (Is / Vt) * np.exp(Vd / Vt)

            # KCL => current flows from p->m
            # So node_p has +Id, node_m has -Id
            # The linear form => +Gd on p->p, +Gd on m->m, -Gd cross terms
            # RHS => diode current contribution offset
            # We'll do incremental stamp
            if n_p is not None:
                G[n_p, n_p] += Gd
            if n_m is not None:
                G[n_m, n_m] += Gd
            if (n_p is not None) and (n_m is not None):
                G[n_p, n_m] -= Gd
                G[n_m, n_p] -= Gd

            # Now the RHS offset => i - Gd*Vd0
            # i.e. i(Vd0) - Gd*Vd0 => we add that to node_p (and subtract from node_m)
            # so that KCL is balanced with the linearized current
            # incremental current = Id - Gd*Vd
            IncI = Id - Gd*Vd
            if n_p is not None:
                RHS[n_p] -= IncI
            if n_m is not None:
                RHS[n_m] += IncI

        elif elem["type"] == "BJT":
            # ... analogously handle BJT
            pass

        elif elem["type"] == "MOS":
            # ... handle MOS
            pass

    return G, RHS

def newton_solve_nonlinear(elements, models, num_nodes, max_iter=50, tol=1e-9):
    """
    Pseudo-code for DC operating point using Newton iteration.
    For each iteration:
      1) Start with guess x (node voltages, etc.).
      2) Clear G, RHS to zero, stamp linear elements (R, V, I, etc.) as usual.
      3) Call stamp_nonlinear_devices(...) to incorporate diode/BJT/MOS partial derivatives.
      4) Solve G x_new = RHS for x_new
      5) Check if ||x_new - x|| < tol => done
      6) Else repeat
    This is a skeleton, does not handle inductors, etc. for DC, or special pivoting, etc.
    """
    # For simplicity, let's say we only keep node voltages in DC
    # (voltage sources add current unknowns, etc.)
    # In a real solver, we must handle those expansions, but let's keep it short.

    x = np.zeros(num_nodes, dtype=float)  # initial guess 0 V
    for it in range(max_iter):
        # Build G, RHS from linear stamps
        G = np.zeros((num_nodes, num_nodes), dtype=float)
        RHS = np.zeros(num_nodes, dtype=float)

        # Stamp linear elements (resistors, voltage/current sources, etc.) for DC
        # (We skip inductors or treat them as short in DC, capacitors open, etc.)
        # The user might define that approach.
        # For brevity, we assume a function stamp_linear_DC(...) not shown here.

        # Stamp the nonlinear devices
        G, RHS = stamp_nonlinear_devices(G, RHS, x, elements, models, node_offset=1)

        # Solve
        x_new = np.linalg.solve(G, RHS)
        diff = np.linalg.norm(x_new - x, ord=2)
        x = x_new
        if diff < tol:
            print(f"Newton converged in {it+1} iterations, diff={diff}")
            break
    return x

###############################################################################
#            FREQUENCY SWEEP & PDF OUTPUT (including final summary)
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
    """Helper to create a matplotlib figure with a text box."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8.5, 11))  # letter size
    plt.axis("off")
    plt.title(title)
    # Use monospace text
    plt.text(0.01, 0.95, text_data, ha='left', va='top', wrap=True,
             family='monospace', fontsize=8)
    return fig

###############################################################################
#                                   MAIN
###############################################################################
if __name__ == "__main__":
    import sys
    import os
    import matplotlib.pyplot as plt

    # -------------------------------------------------------------------------
    # 1) Let the user choose a netlist file from the command line or some GUI
    #    For example: python my_smart_spice.py my_circuit.net
    # -------------------------------------------------------------------------
    if len(sys.argv) < 2:
        print("Usage: python my_smart_spice.py <netlist_file>")
        sys.exit(1)

    netlist_file = sys.argv[1]
    if not os.path.isfile(netlist_file):
        print(f"Error: netlist file '{netlist_file}' not found.")
        sys.exit(1)

    print(f"Loading netlist: {netlist_file}")
    elements, models = parse_netlist(netlist_file)

    # -------------------------------------------------------------------------
    # 2) Determine the highest node number from the elements
    #    For a real parse, also track the new current variables from Vs, etc.
    # -------------------------------------------------------------------------
    max_node = 0
    for e in elements:
        # We just check all node references. For linear elements:
        # R => n1,n2; C => n1,n2; L => n1,n2; V => n1,n2; ...
        # We'll skip the complicated expansions. This is a skeleton.
        for key, val in e.items():
            if key.startswith("n") and isinstance(val, int):
                if val > max_node:
                    max_node = val
        # Some elements (VCVS, etc.) have node_out_p, node_out_m, node_in_p, node_in_m...
        # We do the same approach:
        if "n_out_p" in e:
            if e["n_out_p"] > max_node:
                max_node = e["n_out_p"]
        if "n_out_m" in e:
            if e["n_out_m"] > max_node:
                max_node = e["n_out_m"]
        if "n_in_p" in e:
            if e["n_in_p"] > max_node:
                max_node = e["n_in_p"]
        if "n_in_m" in e:
            if e["n_in_m"] > max_node:
                max_node = e["n_in_m"]
        # etc.

    # Basic node-voltage dimension:
    num_nodes = max_node
    # We will have expansions for voltage sources, etc.
    # Let's do a naive approach: We'll stamp everything in a single pass,
    # capturing expansions automatically as in previous code.

    # Initialize G, C, L for node-voltage dimension
    G = np.zeros((num_nodes, num_nodes), dtype=complex)
    C = np.zeros((num_nodes, num_nodes), dtype=complex)
    L = np.zeros((num_nodes, num_nodes), dtype=complex)
    RHS = np.zeros(num_nodes, dtype=complex)

    unknowns = [f"V{i}" for i in range(1, num_nodes+1)]

    # -------------------------------------------------------------------------
    # 3) Stamp the linear elements. For nonlinear, we'll do a DC solve first.
    #    Then for AC, we'll linearize the devices around that DC operating point.
    # -------------------------------------------------------------------------
    # We'll implement a function stamp_linear_elements(...) for brevity:
    def stamp_linear_elements(elements, G, C, L, RHS, unknowns):
        """
        Stamp only linear parts (R, C, L, V, I, etc.),
        including expansions for V, L in classical MNA.
        Return updated G, C, L, RHS, unknowns.
        """
        for elem in elements:
            t = elem["type"]
            if t == "R":
                # G-stamp
                G = conductance_stamp_custom(G, elem["n1"], elem["n2"], 1.0/elem["value"])
            elif t == "C":
                C = capacitance_stamp_custom(C, elem["n1"], elem["n2"], elem["value"])
            elif t == "L":
                G, C, L, RHS, idx_l = inductor_stamp_custom(G, C, L, RHS,
                                                            elem["n1"], elem["n2"],
                                                            elem["value"])
                unknowns.append(elem["name"]+"_I")
                # store index for controlling current references if needed
                elem["branch_idx"] = idx_l
            elif t == "V":
                G, C, L, RHS, idx_vs = voltage_source_stamp_custom(G, C, L, RHS,
                                                                   elem["n1"], elem["n2"],
                                                                   elem["value"])
                unknowns.append(elem["name"]+"_I")
                elem["branch_idx"] = idx_vs
            elif t == "I":
                RHS = current_source_stamp_custom(RHS, elem["n1"], elem["n2"], elem["value"])
            elif t == "VCCS":
                G = vccs_stamp_custom(G,
                                      elem["n_out_p"], elem["n_out_m"],
                                      elem["n_in_p"], elem["n_in_m"],
                                      elem["gm"])
            elif t == "VCVS":
                G, C, L, RHS, idx_vcvs = vcvs_stamp_custom(G, C, L, RHS,
                                                           elem["n_out_p"], elem["n_out_m"],
                                                           elem["n_in_p"], elem["n_in_m"],
                                                           elem["gain"])
                unknowns.append(elem["name"]+"_I")
                elem["branch_idx"] = idx_vcvs
            elif t == "CCCS":
                # Must find controlling branch index from the netlist
                # This is a minimal approach: user might have named it "Vx"
                # We'll do a pass to see if there's an element with that name
                controlling_branch = None
                for e2 in elements:
                    if e2.get("name", "") == elem["branch_name"]:
                        controlling_branch = e2.get("branch_idx", None)
                if controlling_branch is None:
                    print(f"CCCS {elem['name']} could not find controlling branch {elem['branch_name']}")
                    continue
                G = cccs_stamp_custom(G, elem["n_out_p"], elem["n_out_m"],
                                      controlling_branch, elem["gain"])
            elif t == "CCVS":
                controlling_branch = None
                for e2 in elements:
                    if e2.get("name", "") == elem["branch_name"]:
                        controlling_branch = e2.get("branch_idx", None)
                if controlling_branch is None:
                    print(f"CCVS {elem['name']} could not find controlling branch {elem['branch_name']}")
                    continue
                G, C, L, RHS, idx_ccvs = ccvs_stamp_custom(G, C, L, RHS,
                                                           elem["n_out_p"], elem["n_out_m"],
                                                           controlling_branch,
                                                           elem["transres"])
                unknowns.append(elem["name"]+"_I")
                elem["branch_idx"] = idx_ccvs
            else:
                # Nonlinear => skip in linear stamping
                pass
        return G, C, L, RHS, unknowns

    # We'll just reuse from the code above (CircuitStamps) but rename them here:
    def conductance_stamp_custom(G, n1, n2, val):
        if n1>0:
            G[n1-1,n1-1]+=val
        if n2>0:
            G[n2-1,n2-1]+=val
        if n1>0 and n2>0:
            G[n1-1,n2-1]-=val
            G[n2-1,n1-1]-=val
        return G

    def capacitance_stamp_custom(C, n1, n2, val):
        if n1>0:
            C[n1-1,n1-1]+=val
        if n2>0:
            C[n2-1,n2-1]+=val
        if n1>0 and n2>0:
            C[n1-1,n2-1]-=val
            C[n2-1,n1-1]-=val
        return C

    from copy import deepcopy
    # We need the expansions from our main code:
    from_num = G.shape[0]
    def voltage_source_stamp_custom(G, C, L, RHS, n1, n2, volt):
        # replicate the logic from CircuitStamps...
        size_old = G.shape[0]
        G_expanded = np.pad(G, ((0,1),(0,1)), mode='constant', constant_values=0)
        C_expanded = np.pad(C, ((0,1),(0,1)), mode='constant', constant_values=0)
        L_expanded = np.pad(L, ((0,1),(0,1)), mode='constant', constant_values=0)
        RHS_expanded = np.pad(RHS, (0,1), mode='constant', constant_values=0)
        idx = size_old
        if n1>0:
            G_expanded[n1-1, idx]+=1
            G_expanded[idx, n1-1]+=1
        if n2>0:
            G_expanded[n2-1, idx]-=1
            G_expanded[idx, n2-1]-=1
        RHS_expanded[idx] = volt
        return G_expanded, C_expanded, L_expanded, RHS_expanded, idx

    def current_source_stamp_custom(RHS, n1, n2, val):
        if n1>0:
            RHS[n1-1]+=val
        if n2>0:
            RHS[n2-1]-=val
        return RHS

    def inductor_stamp_custom(G, C, L, RHS, n1, n2, val):
        size_old = G.shape[0]
        G_expanded = np.pad(G, ((0,1),(0,1)), mode='constant')
        C_expanded = np.pad(C, ((0,1),(0,1)), mode='constant')
        L_expanded = np.pad(L, ((0,1),(0,1)), mode='constant')
        RHS_expanded = np.pad(RHS, (0,1), mode='constant')
        idx = size_old
        if n1>0:
            G_expanded[n1-1, idx]+=1
            G_expanded[idx, n1-1]+=1
        if n2>0:
            G_expanded[n2-1, idx]-=1
            G_expanded[idx, n2-1]-=1
        # store -L
        L_expanded[idx, idx] = -val
        return G_expanded, C_expanded, L_expanded, RHS_expanded, idx

    def vccs_stamp_custom(G, nop, nom, nip, nim, gm):
        p_out = nop-1 if nop>0 else None
        m_out = nom-1 if nom>0 else None
        p_in =  nip-1 if nip>0 else None
        m_in =  nim-1 if nim>0 else None
        if p_out is not None and p_in is not None:
            G[p_out, p_in]+=gm
        if p_out is not None and m_in is not None:
            G[p_out, m_in]-=gm
        if m_out is not None and p_in is not None:
            G[m_out, p_in]-=gm
        if m_out is not None and m_in is not None:
            G[m_out, m_in]+=gm
        return G

    def vcvs_stamp_custom(G, C, L, RHS, nop, nom, nip, nim, gain):
        size_old = G.shape[0]
        G_expanded = np.pad(G, ((0,1),(0,1)), mode='constant')
        C_expanded = np.pad(C, ((0,1),(0,1)), mode='constant')
        L_expanded = np.pad(L, ((0,1),(0,1)), mode='constant')
        RHS_expanded = np.pad(RHS, (0,1), mode='constant')
        idx = size_old
        if nop>0:
            G_expanded[nop-1, idx]+=1
            G_expanded[idx, nop-1]+=1
        if nom>0:
            G_expanded[nom-1, idx]-=1
            G_expanded[idx, nom-1]-=1
        p_in = nip-1 if nip>0 else None
        m_in = nim-1 if nim>0 else None
        if p_in is not None:
            G_expanded[idx, p_in] -= gain
        if m_in is not None:
            G_expanded[idx, m_in] += gain
        return G_expanded, C_expanded, L_expanded, RHS_expanded, idx

    def cccs_stamp_custom(G, nop, nom, branch_idx, gain):
        p_out = nop-1 if nop>0 else None
        m_out = nom-1 if nom>0 else None
        if p_out is not None:
            G[p_out, branch_idx]+=gain
        if m_out is not None:
            G[m_out, branch_idx]-=gain
        return G

    def ccvs_stamp_custom(G, C, L, RHS, nop, nom, branch_idx_in, transres):
        size_old = G.shape[0]
        G_expanded = np.pad(G, ((0,1),(0,1)), mode='constant')
        C_expanded = np.pad(C, ((0,1),(0,1)), mode='constant')
        L_expanded = np.pad(L, ((0,1),(0,1)), mode='constant')
        RHS_expanded = np.pad(RHS, (0,1), mode='constant')
        idx = size_old
        if nop>0:
            G_expanded[nop-1, idx]+=1
            G_expanded[idx, nop-1]+=1
        if nom>0:
            G_expanded[nom-1, idx]-=1
            G_expanded[idx, nom-1]-=1
        G_expanded[idx, branch_idx_in] -= transres
        return G_expanded, C_expanded, L_expanded, RHS_expanded, idx

    # Stamp linear parts
    G, C, L, RHS, unknowns = stamp_linear_elements(elements, G, C, L, RHS, unknowns)

    # If we do a DC operating point (Newton) for nonlinear, we do it here.
    # => newton_solve_nonlinear(elements, models, <some dimension>)

    # For AC small-signal, we assume the linearization around DC is already done
    # and is represented in G, C, L. (This is a simplified approach.)

    # -------------------------------------------------------------------------
    # 4) FREQUENCY SWEEP (like before), PARALLEL
    # -------------------------------------------------------------------------
    from functools import partial
    n_size = G.shape[0]
    print(f"Final MNA dimension: {n_size}x{n_size}")
    print("Unknowns:", unknowns)

    # For debug, pick 1 frequency to show M
    f_debug = 1e3
    s_debug = 1j*2*math.pi*f_debug
    M_debug = G + s_debug*C + s_debug*L

    print("\n--- Debug Print ---")
    print(matrix_to_string(G, "G"))
    print(matrix_to_string(C, "C"))
    print(matrix_to_string(L, "L (inductor -L on diag)"))
    print(vector_to_string(RHS, "RHS"))
    print(f"Unknowns = {unknowns}")
    print(f"\nM_debug at f={f_debug} =>\n{M_debug}\n")

    # Sweep
    num_points = 10000
    freqs = np.logspace(np.log10(1), np.log10(1e9), num_points)
    x_solutions = np.zeros((num_points, n_size), dtype=complex)

    parallel_args = [(f, G, C, L, RHS) for f in freqs]

    start_time_all = time.time()
    total_flops = 0

    processes = min(cpu_count(), 8)
    with Pool(processes=processes) as pool:
        results_iter = pool.imap(solve_frequency_wrapper, parallel_args)
        for i, (freq_val, x_val, flops_val) in enumerate(
                tqdm(results_iter, total=num_points, desc="Frequency Sweep")
        ):
            x_solutions[i,:] = x_val
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
            for val in x_solutions[i,:]:
                row += f",{val.real},{val.imag}"
            row += "\n"
            f_out.write(row)
    print(f"Saved solution data to {csv_file}")

    # Plot node voltages magnitude/phase => only first 'num_nodes' are actual node voltages
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

    print("Saved plots to plot_results.pdf")

    # Debug & FLOPs PDF
    with PdfPages("debug_and_flops.pdf") as pdfd:
        # Page1: G, C, L, RHS, M_debug
        page1_text = ""
        page1_text += matrix_to_string(G, "G")
        page1_text += matrix_to_string(C, "C")
        page1_text += matrix_to_string(L, "L")
        page1_text += vector_to_string(RHS, "RHS")
        page1_text += f"\nUnknowns:\n{unknowns}\n"
        page1_text += f"\nM_debug at f={f_debug}:\n{np.array_str(M_debug, precision=4, suppress_small=True)}\n"
        fig1 = make_text_page(page1_text, title="Debug Matrices / M_debug")
        pdfd.savefig(fig1)
        plt.close(fig1)

        # Page2: Per-frequency FLOPs (approx)
        page2_text = "Frequency(Hz), FLOPs\n"
        # We didn't store flops for each freq individually (just total), so let's do a quick re-calc:
        n = n_size
        single_flops = (2.0/3.0)*(n**3) + 2.0*(n**2)
        # For large # of points, listing all 10k frequencies is too big,
        # but let's list maybe the first 10
        for i in range(min(10, num_points)):
            page2_text += f"{freqs[i]:.4g}, {single_flops:.4e}\n"
        page2_text += f" ... (omitted many)\n"
        page2_text += f"\nTOTAL SWEEP FLOPs = {total_flops:.4e}\n"
        page2_text += f"TOTAL TIME (sec) = {elapsed_time_all:.2f}\n"

        fig2 = make_text_page(page2_text, title="FLOPs & Timing")
        pdfd.savefig(fig2)
        plt.close(fig2)

        # A final summary page
        summary_text = f"Simulation completed\nFrequency Points = {num_points}\n"
        summary_text += f"Total Time = {elapsed_time_all:.2f} s\n"
        summary_text += f"Total FLOPs = {total_flops:.4e}\n"
        fig3 = make_text_page(summary_text, title="Final Summary")
        pdfd.savefig(fig3)
        plt.close(fig3)

    print("Saved debug info, total time & FLOPs to debug_and_flops.pdf")

    print("\n=== Simulation Complete ===")
    print(f"Total time: {elapsed_time_all:.2f} s")
    print(f"Total FLOPs: {total_flops:.4e}")
    print("===========================")
