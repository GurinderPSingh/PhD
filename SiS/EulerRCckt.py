import numpy as np
import time

def simulate_euler_rc_circuit(input_type="dc", params=None):
    """
    Simulates an RC circuit using various numerical methods (Euler, RK4, BDF-1) for different input types.

    :param input_type: Type of input ('dc', 'step', 'impulse', 'delayed_impulse', 'ramp').
    :param params: Dictionary containing input parameters for the simulation.
    :return: A dictionary containing time array, input voltage, and simulation results for Euler, RK4, and BDF-1.
    """
    if params is None:
        params = {}

    # Parameters
    R = 1000  # Resistance (Ohms)
    C = 1e-6  # Capacitance (Farads)
    time_steps = 100_000  # Increased time steps
    t = np.linspace(0, 0.01, time_steps)  # Time array
    h = t[1] - t[0]  # Time step size

    # Configure V_source based on input_type
    if input_type == "dc":
        V_source = np.full_like(t, params.get("dc_voltage", 5))  # DC voltage (default: 5V)

    elif input_type == "step":
        V_initial = params.get("v_initial", 0)  # Initial voltage
        V_final = params.get("v_final", 5)  # Final voltage
        t_step = params.get("t_step", 0.005)  # Step time
        V_source = np.where(t < t_step, V_initial, V_final)

    elif input_type == "impulse":
        t_pulse = params.get("t_pulse", 0.0001)  # Pulse duration
        V_impulse = params.get("v_impulse", 1 / t_pulse)  # Impulse height (area = 1)
        V_source = np.zeros_like(t)
        pulse_end_index = int(t_pulse / h)
        V_source[:pulse_end_index] = V_impulse

    elif input_type == "delayed_impulse":
        t_pulse = params.get("t_pulse", 0.0001)  # Pulse duration
        t_start = params.get("t_start", 0.002)  # Start time of impulse
        V_impulse = params.get("v_impulse", 1 / t_pulse)
        V_source = np.zeros_like(t)
        start_index = int(t_start / h)
        end_index = start_index + int(t_pulse / h)
        V_source[start_index:end_index] = V_impulse

    elif input_type == "ramp":
        t_start = params.get("t_start", 0.002)  # Ramp starts
        slope = params.get("slope", 1000)  # Ramp slope
        V_source = np.where(t < t_start, 0, slope * (t - t_start))

    else:
        raise ValueError("Invalid input_type. Choose from: dc, step, impulse, delayed_impulse, ramp.")

    # Initialize voltage arrays
    V_euler = np.zeros_like(t)
    V_rk4 = np.zeros_like(t)
    V_bdf = np.zeros_like(t)

    # Numerical methods
    start_time_euler = time.time()
    for i in range(1, len(t)):
        dVdt = (V_source[i] - V_euler[i-1]) / (R * C)
        V_euler[i] = V_euler[i-1] + h * dVdt
    time_euler = time.time() - start_time_euler

    start_time_rk4 = time.time()
    for i in range(1, len(t)):
        k1 = (V_source[i] - V_rk4[i-1]) / (R * C)
        k2 = (V_source[i] - (V_rk4[i-1] + 0.5 * h * k1)) / (R * C)
        k3 = (V_source[i] - (V_rk4[i-1] + 0.5 * h * k2)) / (R * C)
        k4 = (V_source[i] - (V_rk4[i-1] + h * k3)) / (R * C)
        V_rk4[i] = V_rk4[i-1] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    time_rk4 = time.time() - start_time_rk4

    start_time_bdf = time.time()
    for i in range(1, len(t)):
        V_bdf[i] = (V_bdf[i-1] + h * V_source[i] / (R * C)) / (1 + h / (R * C))
    time_bdf = time.time() - start_time_bdf

    return {
        "time": t,
        "input_voltage": V_source,
        "euler": V_euler,
        "rk4": V_rk4,
        "bdf": V_bdf,
        "times": {"euler": time_euler, "rk4": time_rk4, "bdf": time_bdf}
    }
