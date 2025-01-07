import numpy as np
import matplotlib.pyplot as plt

def newton_raphson(func, dfunc, initial_guess, tolerance=1e-6, max_iterations=100):
    """
    Implements the Newton-Raphson method for root finding.

    Parameters:
    - func: The function whose root is to be found.
    - dfunc: The derivative of the function.
    - initial_guess: Starting point for the iterations.
    - tolerance: Convergence tolerance.
    - max_iterations: Maximum number of iterations allowed.

    Returns:
    - root: The root of the function.
    - iterations: Number of iterations taken.
    """
    x = initial_guess
    for i in range(max_iterations):
        fx = func(x)
        dfx = dfunc(x)
        if abs(fx) < tolerance:
            return x, i  # Converged to the root
        if dfx == 0:
            raise ValueError("Derivative is zero. Newton-Raphson method fails.")
        x = x - fx / dfx
    raise ValueError("Max iterations exceeded. Method did not converge.")

# Constants for the diode
IS = 1e-12  # Reverse saturation current (A)
VT = 26e-3  # Thermal voltage (V)
n = 1       # Ideality factor

# Example 1: Single Diode
ID_target = 1e-3  # Target current (A)

# Define the diode equation and its derivative
def diode_func(VD):
    return IS * (np.exp(VD / (n * VT)) - 1) - ID_target

def diode_dfunc(VD):
    return (IS / (n * VT)) * np.exp(VD / (n * VT))

# Solve for VD using Newton-Raphson
VD_initial_guess = 0.7  # Initial guess for VD
VD_single, iterations_single = newton_raphson(diode_func, diode_dfunc, VD_initial_guess)

# Example 2: Diode in Series with Resistor
R = 1000  # Resistor value (Ohms)
VS = 5    # Source voltage (V)

# Define the combined equation for the diode and resistor
def diode_resistor_func(VD):
    return IS * (np.exp(VD / (n * VT)) - 1) - (VS - VD) / R

def diode_resistor_dfunc(VD):
    return (IS / (n * VT)) * np.exp(VD / (n * VT)) + 1 / R

# Solve for VD using Newton-Raphson
VD_series, iterations_series = newton_raphson(diode_resistor_func, diode_resistor_dfunc, VD_initial_guess)

# Numerical Results
print("Single Diode:")
print(f"Voltage across diode (VD): {VD_single:.6f} V")
print(f"Iterations: {iterations_single}")

print("\nDiode in Series with Resistor:")
print(f"Voltage across diode (VD): {VD_series:.6f} V")
print(f"Iterations: {iterations_series}")

# Graphical Results
VD_range = np.linspace(0, 1, 500)  # Voltage range for plotting
ID_single = IS * (np.exp(VD_range / (n * VT)) - 1)
ID_series = IS * (np.exp(VD_range / (n * VT)) - 1)
I_R = (VS - VD_range) / R  # Current through the resistor

plt.figure(figsize=(10, 6))
# Single Diode IV Curve
plt.subplot(2, 1, 1)
plt.plot(VD_range, ID_single * 1e3, label="Diode IV Curve")
plt.axvline(VD_single, color='r', linestyle='--', label=f"VD = {VD_single:.3f} V")
plt.axhline(ID_target * 1e3, color='g', linestyle='--', label=f"ID = {ID_target * 1e3:.1f} mA")
plt.xlabel("Voltage Across Diode (V)")
plt.ylabel("Current Through Diode (mA)")
plt.title("Single Diode IV Curve")
plt.legend()
plt.grid()

# Diode + Resistor IV Curve
plt.subplot(2, 1, 2)
plt.plot(VD_range, ID_series * 1e3, label="Diode IV Curve")
plt.plot(VD_range, I_R * 1e3, label="Resistor Current (I_R)")
plt.axvline(VD_series, color='r', linestyle='--', label=f"VD = {VD_series:.3f} V")
plt.axhline((VS - VD_series) / R * 1e3, color='g', linestyle='--', label=f"ID = {(VS - VD_series) / R * 1e3:.1f} mA")
plt.xlabel("Voltage Across Diode (V)")
plt.ylabel("Current (mA)")
plt.title("Diode in Series with Resistor")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
