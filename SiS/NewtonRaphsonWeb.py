from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Newton-Raphson for Diode Simulation
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
    return VD

# Scaling Analysis
def simulate_scaling():
    scaling_factors = np.array([1, 2, 4, 8])
    base_resistance = 100
    base_capacitance = 2e-12
    length = 1
    rc_delay = np.array([base_resistance * s**2 * base_capacitance * length**2 for s in scaling_factors])

    return scaling_factors, rc_delay

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    graph = None
    if request.method == 'POST':
        simulation_type = request.form['simulation']
        if simulation_type == 'diode':
            ID_target = float(request.form['ID_target'])
            VD = simulate_diode(ID_target)
            result = f"Voltage across diode (VD): {VD:.6f} V"

        elif simulation_type == 'scaling':
            scaling_factors, rc_delay = simulate_scaling()
            plt.figure()
            plt.plot(scaling_factors, rc_delay, marker='o', label="RC Delay")
            plt.xlabel("Scaling Factor (S)")
            plt.ylabel("RC Delay (seconds)")
            plt.title("Impact of Scaling on RC Delay")
            plt.grid()
            plt.legend()
            plt.tight_layout()

            # Save the plot to a PNG image and encode it to base64
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            graph = base64.b64encode(img.getvalue()).decode()

    return render_template('index.html', result=result, graph=graph)

if __name__ == '__main__':
    app.run(debug=True)
