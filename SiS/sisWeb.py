import numpy as np

def simulate_scaling():
    scaling_factors = np.array([1, 2, 4, 8])
    base_resistance = 100
    base_capacitance = 2e-12
    length = 1

    rc_delay = np.array([base_resistance * s**2 * base_capacitance * length**2 for s in scaling_factors])
    return scaling_factors, rc_delay

def simulate_frequency_vs_wavelength():
    frequency = np.linspace(1e6, 10e9, 500)
    velocity = 3e8 / np.sqrt(4.5)
    wavelength = velocity / frequency
    return frequency, wavelength

def simulate_delay_vs_frequency():
    selected_scaling = 4
    base_resistance = 100
    base_capacitance = 2e-12
    length = 1

    resistance = base_resistance * selected_scaling**2
    capacitance = base_capacitance
    selected_rc_delay = resistance * capacitance * length**2

    frequency = np.linspace(1e6, 10e9, 500)
    delay_vs_freq = selected_rc_delay / frequency
    return frequency, delay_vs_freq

def simulate_harmonic_content():
    rise_time = np.array([1e-9, 2e-9, 4e-9])
    harmonic_frequencies = np.linspace(1, 10, 10)

    results = {}
    for tr in rise_time:
        harmonics = 1 / (tr * harmonic_frequencies)
        results[f"Rise Time = {tr * 1e9:.1f} ns"] = harmonics
    return harmonic_frequencies, results
