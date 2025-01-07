import numpy as np
import matplotlib.pyplot as plt
import time

# Parameters
R = 1000  # Resistance (Ohms)
C = 1e-6  # Capacitance (Farads)
t_pulse = 0.0001  # Pulse duration (seconds)
t_start = 0.002  # Start time of the impulse (seconds)
V_impulse = 1 / t_pulse  # Impulse height (to maintain area = 1)
time_steps = 10_000  # Increased time steps
t = np.linspace(0, 0.01, time_steps)  # Time array
h = t[1] - t[0]  # Time step size

# Delayed impulse input
V_source = np.zeros_like(t)
start_index = int(t_start / h)  # Start index for the impulse
end_index = start_index + int(t_pulse / h)  # End index for the impulse
V_source[start_index:end_index] = V_impulse  # Define impulse within specified range

# Initial conditions
V_euler = np.zeros_like(t)
V_rk4 = np.zeros_like(t)
V_bdf = np.zeros_like(t)
V_euler[0] = V_rk4[0] = V_bdf[0] = 0  # Initial voltage across the capacitor

# Euler's Method
start_time_euler = time.time()
for i in range(1, len(t)):
    dVdt = (V_source[i] - V_euler[i-1]) / (R * C)
    V_euler[i] = V_euler[i-1] + h * dVdt
end_time_euler = time.time()

# Runge-Kutta Method (RK4)
start_time_rk4 = time.time()
for i in range(1, len(t)):
    k1 = (V_source[i] - V_rk4[i-1]) / (R * C)
    k2 = (V_source[i] - (V_rk4[i-1] + 0.5 * h * k1)) / (R * C)
    k3 = (V_source[i] - (V_rk4[i-1] + 0.5 * h * k2)) / (R * C)
    k4 = (V_source[i] - (V_rk4[i-1] + h * k3)) / (R * C)
    V_rk4[i] = V_rk4[i-1] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
end_time_rk4 = time.time()

# Backward Differentiation Formula (BDF-1)
start_time_bdf = time.time()
for i in range(1, len(t)):
    V_bdf[i] = (V_bdf[i-1] + h * V_source[i] / (R * C)) / (1 + h / (R * C))
end_time_bdf = time.time()

# Measure simulation times
time_euler = end_time_euler - start_time_euler
time_rk4 = end_time_rk4 - start_time_rk4
time_bdf = end_time_bdf - start_time_bdf

# Plot Results
plt.figure(figsize=(10, 6))
plt.plot(t, V_source, 'k--', label="Input Voltage (Delayed Impulse)")
plt.plot(t, V_euler, label=f"Euler's Method (Time: {time_euler:.6f} s)", linewidth=1.5)
plt.plot(t, V_rk4, label=f"Runge-Kutta (RK4) (Time: {time_rk4:.6f} s)", linestyle='--', linewidth=1.5)
plt.plot(t, V_bdf, label=f"BDF-1 (Time: {time_bdf:.6f} s)", linestyle=':', linewidth=1.5)
plt.title("Transient Response of RC Circuit with Delayed Impulse Input")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Print Simulation Times
print(f"Euler's Method Simulation Time: {time_euler:.6f} seconds")
print(f"Runge-Kutta (RK4) Simulation Time: {time_rk4:.6f} seconds")
print(f"BDF-1 Simulation Time: {time_bdf:.6f} seconds")
