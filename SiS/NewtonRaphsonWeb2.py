import numpy as np

def newton_raphson(func, dfunc, initial_guess, tolerance=1e-6, max_iterations=100):
    x = initial_guess
    for _ in range(max_iterations):
        fx = func(x)
        dfx = dfunc(x)
        if abs(fx) < tolerance:
            return x
        if dfx == 0:
            return None  # Derivative zero, no solution
        x -= fx / dfx
    return None  # Max iterations exceeded

def simulate_diode(ID_target, initial_guess=0.7):
    IS = 1e-12
    VT = 26e-3
    n = 1

    def diode_func(VD):
        return IS * (np.exp(VD / (n * VT)) - 1) - ID_target

    def diode_dfunc(VD):
        return (IS / (n * VT)) * np.exp(VD / (n * VT))

    VD = newton_raphson(diode_func, diode_dfunc, initial_guess)
    VD_range = np.linspace(0, 1, 500)
    ID = IS * (np.exp(VD_range / (n * VT)) - 1)
    return VD, VD_range, ID

def simulate_diode_with_resistor(VS, R, initial_guess=0.7):
    IS = 1e-12
    VT = 26e-3
    n = 1

    def diode_resistor_func(VD):
        return IS * (np.exp(VD / (n * VT)) - 1) - (VS - VD) / R

    def diode_resistor_dfunc(VD):
        return (IS / (n * VT)) * np.exp(VD / (n * VT)) + 1 / R

    VD = newton_raphson(diode_resistor_func, diode_resistor_dfunc, initial_guess)
    VD_range = np.linspace(0, VS, 500)
    ID_diode = IS * (np.exp(VD_range / (n * VT)) - 1)
    I_R = (VS - VD_range) / R
    return VD, VD_range, ID_diode, I_R
