from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import io
import base64
from sisWeb import (simulate_scaling, simulate_frequency_vs_wavelength,
                    simulate_delay_vs_frequency, simulate_harmonic_content)
from NewtonRaphsonWeb2 import simulate_diode, simulate_diode_with_resistor
from EulerRCckt import simulate_euler_rc_circuit
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    graph = None
    if request.method == 'POST':
        simulation_type = request.form['simulation']



        # Numerical Methods for Differential Equations
        if simulation_type == 'numerical_methods':
            method = request.form['method']
            try:
                params = {
                    "dc_voltage": float(request.form.get("dc_voltage", 5) or 5),  # Default to 5V for DC
                    "v_initial": float(request.form.get("v_initial", 0) or 0),  # Default to 0V
                    "v_final": float(request.form.get("v_final", 5) or 5),  # Default to 5V
                    "t_step": float(request.form.get("t_step", 0.005) or 0.005),  # Default to 0.005s
                    "t_pulse": float(request.form.get("t_pulse", 0.0001) or 0.001),  # Default to 0.0001s
                    "t_start": float(request.form.get("t_start", 0.002) or 0.002),  # Default to 0.002s
                    "slope": float(request.form.get("slope", 1000) or 1000),  # Default slope for ramp
                    "v_impulse": float(request.form.get("v_impulse", 10) or 10)  # Default impulse height
                }
            except ValueError as e:
                result = f"Error: Invalid input value. {str(e)}"
                return render_template('index.html', result=result)

            # Run the simulation
            data = simulate_euler_rc_circuit(input_type=method, params=params)


    # Extract data for plotting
            t = data["time"]
            V_source = data["input_voltage"]
            V_euler = data["euler"]
            V_rk4 = data["rk4"]
            V_bdf = data["bdf"]
            times = data["times"]



            # Generate plot
            plt.figure(figsize=(10, 6))
            plt.plot(t, V_source, 'k--', label="Input Voltage")
            plt.plot(t, V_euler, label=f"Euler's Method (Time: {times['euler']:.6f} s)", linewidth=1.5)
            plt.plot(t, V_rk4, label=f"Runge-Kutta (RK4) (Time: {times['rk4']:.6f} s)", linestyle='--', linewidth=1.5)
            plt.plot(t, V_bdf, label=f"BDF-1 (Time: {times['bdf']:.6f} s)", linestyle=':', linewidth=1.5)
            plt.title(f"Transient Response of RC Circuit ({method.capitalize()} Input)")
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage (V)")
            plt.grid()
            plt.legend()
            plt.tight_layout()

            # Save the plot as base64
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            graph = base64.b64encode(img.getvalue()).decode()

            # Result message
            result = f"Simulation completed for {method.capitalize()} input."



    # Other simulation types remain unchanged
        elif simulation_type == 'diode':
            ID_target = float(request.form['ID_target'])
            VD, VD_range, ID = simulate_diode(ID_target)
            result = f"Voltage across diode (VD): {VD:.6f} V"
            plt.figure()
            plt.plot(VD_range, ID * 1e3, label="Diode IV Curve")
            plt.axvline(VD, color='r', linestyle='--', label=f"VD = {VD:.3f} V")
            plt.axhline(ID_target * 1e3, color='g', linestyle='--', label=f"ID = {ID_target * 1e3:.1f} mA")
            plt.xlabel("Voltage Across Diode (V)")
            plt.ylabel("Current Through Diode (mA)")
            plt.title("Single Diode IV Curve")
            plt.legend()
            plt.grid()
            plt.tight_layout()

        elif simulation_type == 'diode_resistor':
            # Diode + Resistor Simulation
            VS = float(request.form['VS'])
            R = float(request.form['R'])
            VD, VD_range, ID_diode, I_R = simulate_diode_with_resistor(VS, R)

            result = f"Voltage across diode with resistor (VD): {VD:.6f} V"
            plt.figure()
            plt.plot(VD_range, ID_diode * 1e3, label="Diode IV Curve")
            plt.plot(VD_range, I_R * 1e3, label="Resistor Current (I_R)")
            plt.axvline(VD, color='r', linestyle='--', label=f"VD = {VD:.3f} V")
            plt.xlabel("Voltage Across Diode (V)")
            plt.ylabel("Current (mA)")
            plt.title("Diode + Resistor IV Curve")
            plt.legend()
            plt.grid()
            plt.tight_layout()

        elif simulation_type == 'scaling':
            # Scaling Analysis
            scaling_factors, rc_delay = simulate_scaling()
            plt.figure()
            plt.plot(scaling_factors, rc_delay, marker='o', label="RC Delay")
            plt.xlabel("Scaling Factor (S)")
            plt.ylabel("RC Delay (seconds)")
            plt.title("Impact of Scaling on RC Delay")
            plt.grid()
            plt.legend()
            plt.tight_layout()

        elif simulation_type == 'frequency_wavelength':
            # Frequency vs Wavelength
            frequency, wavelength = simulate_frequency_vs_wavelength()
            plt.figure()
            plt.plot(frequency / 1e9, wavelength, label="Wavelength")
            plt.xlabel("Frequency (GHz)")
            plt.ylabel("Wavelength (m)")
            plt.title("Frequency vs. Wavelength")
            plt.grid()
            plt.legend()
            plt.tight_layout()

        elif simulation_type == 'delay_frequency':
            # Delay vs Frequency
            frequency, delay_vs_freq = simulate_delay_vs_frequency()
            plt.figure()
            plt.plot(frequency / 1e9, delay_vs_freq * 1e12, label="RC Delay")
            plt.xlabel("Frequency (GHz)")
            plt.ylabel("Delay (ps)")
            plt.title("RC Delay vs. Frequency")
            plt.grid()
            plt.legend()
            plt.tight_layout()

        elif simulation_type == 'harmonics':
            # Harmonic Content
            harmonic_frequencies, results = simulate_harmonic_content()
            plt.figure()
            for label, harmonics in results.items():
                plt.plot(harmonic_frequencies, harmonics / 1e9, label=label)
            plt.xlabel("Harmonic Number")
            plt.ylabel("Frequency (GHz)")
            plt.title("Harmonic Frequencies for Rise Times")
            plt.grid()
            plt.legend()
            plt.tight_layout()

        # Convert plot to image for display
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph = base64.b64encode(img.getvalue()).decode()

    return render_template('index.html', result=result, graph=graph)

if __name__ == '__main__':
    app.run(debug=True)
