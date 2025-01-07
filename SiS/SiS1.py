import numpy as np
import matplotlib.pyplot as plt

# Constants
scaling_factors = np.array([1, 2, 4, 8])  # Scaling factors
frequency = np.linspace(1e6, 10e9, 500)  # Frequency range (1 MHz to 10 GHz)
velocity = 3e8 / np.sqrt(4.5)  # Signal velocity in medium (assuming ε_r = 4.5)
wavelength = velocity / frequency  # Wavelength for each frequency

# RC Delay Parameters
base_resistance = 100  # Ohms per unit length
base_capacitance = 2e-12  # Farads per unit length
length = 1  # Interconnect length (meters)

# Effects of Scaling
resistance = base_resistance * scaling_factors**2  # Resistance per unit length scales with S^2
capacitance = base_capacitance * np.ones_like(scaling_factors)  # Capacitance remains constant
rc_delay = np.array([r * c * length**2 for r, c in zip(resistance, capacitance)])  # RC delay scales

# Rise Time and Harmonics
rise_time = np.array([1e-9, 2e-9, 4e-9])  # Rise times (1 ns, 2 ns, 4 ns)
harmonic_frequencies = np.linspace(1, 10, 10)  # 1st to 10th harmonics

# Plot RC Delay vs Scaling Factor
plt.figure(figsize=(10, 6))
plt.plot(scaling_factors, rc_delay, marker='o', label="RC Delay")
plt.xlabel("Scaling Factor (S)")
plt.ylabel("RC Delay (seconds)")
plt.title("Impact of Scaling on RC Delay")
plt.grid()
plt.legend()
plt.show()

# Frequency vs. Wavelength
plt.figure(figsize=(10, 6))
for s in scaling_factors:
    plt.plot(frequency / 1e9, wavelength / s, label=f"Scaling Factor S={s}")

plt.xlabel("Frequency (GHz)")
plt.ylabel("Wavelength (m)")
plt.title("Frequency vs. Wavelength for Different Scaling Factors")
plt.grid()
plt.legend()
plt.show()

# Delay vs Frequency (for one scaling factor)
selected_scaling = 4
selected_resistance = base_resistance * selected_scaling**2
selected_capacitance = base_capacitance
selected_rc_delay = selected_resistance * selected_capacitance * length**2

delay_vs_freq = selected_rc_delay / frequency

plt.figure(figsize=(10, 6))
plt.plot(frequency / 1e9, delay_vs_freq * 1e12, label=f"Scaling Factor S={selected_scaling}")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Delay (ps)")
plt.title("RC Delay vs. Frequency for Selected Scaling Factor")
plt.grid()
plt.legend()
plt.show()

# Harmonic Content and Rise Time
plt.figure(figsize=(10, 6))
for tr in rise_time:
    harmonics = 1 / (tr * harmonic_frequencies)
    plt.plot(harmonic_frequencies, harmonics / 1e9, label=f"Rise Time = {tr*1e9:.1f} ns")

plt.xlabel("Harmonic Number")
plt.ylabel("Frequency (GHz)")
plt.title("Harmonic Frequency vs. Rise Time")
plt.grid()
plt.legend()
plt.show()
