import pybamm
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

bounds = {
    'Positive electrode diffusivity [m2.s-1]': (5, 6),
    'Negative electrode diffusivity [m2.s-1]': (4, 5),
    'Positive particle radius [m]': (4, 5),
    'Negative particle radius [m]': (4, 5),
    'Negative electrode active material volume fraction': (6, 6.5),
    'Positive electrode active material volume fraction': (5, 5.8)
}


def apply_bounds(particle):
    particle[0] = np.clip(particle[0], bounds['Positive electrode diffusivity [m2.s-1]'][0], bounds['Positive electrode diffusivity [m2.s-1]'][1])
    particle[1] = np.clip(particle[1], bounds['Negative electrode diffusivity [m2.s-1]'][0], bounds['Negative electrode diffusivity [m2.s-1]'][1])
    particle[2] = np.clip(particle[2], bounds['Positive particle radius [m]'][0], bounds['Positive particle radius [m]'][1])
    particle[3] = np.clip(particle[3], bounds['Negative particle radius [m]'][0], bounds['Negative particle radius [m]'][1])
    particle[4] = np.clip(particle[4], bounds['Negative electrode active material volume fraction'][0], bounds['Negative electrode active material volume fraction'][1])
    particle[5] = np.clip(particle[5], bounds['Positive electrode active material volume fraction'][0], bounds['Positive electrode active material volume fraction'][1])
    return particle

def load_data():
    U_begin = np.loadtxt("image/data_1C_begin")
    T_begin = np.loadtxt("image/temp_data 1C begin")
    Time_begin = np.arange(0, U_begin.size)

    U_middle = np.loadtxt("image/data_1C_middle")
    T_middle = np.loadtxt("image/temp_data 1C middle")
    Time_middle = np.arange(0, U_middle.size)

    U_age = np.loadtxt("image/data_1C_age")
    T_age = np.loadtxt("image/temp_data 1C age")
    Time_age = np.arange(0, U_age.size)

    return (Time_begin, U_begin, T_begin), (Time_middle, U_middle, T_middle), (Time_age, U_age, T_age)

def run_battery_model(params, current):

    # pybamm.set_logging_level("INFO")
    model = pybamm.lithium_ion.SPMe({"thermal": "lumped"}, name="lumped thermal model")
    geometry = model.default_geometry
    param = pybamm.ParameterValues("Chen2020")

    # Update parameters
    param.update({
        "Positive electrode diffusivity [m2.s-1]": params[0] * 1e-15,
        "Negative electrode diffusivity [m2.s-1]": params[1] * 1e-14,
        "Positive particle radius [m]": params[2] * 1e-6,
        "Negative particle radius [m]": params[3] * 1e-6,
        'Negative electrode active material volume fraction': params[4] * 0.1,
        'Positive electrode active material volume fraction': params[5] * 0.1,
        "Current function [A]": current
    })

    param.process_geometry(geometry)
    param.process_model(model)
    # 网格设置
    var = pybamm.standard_spatial_vars
    var_pts = {var.x_n: 30, var.x_s: 30, var.x_p: 30, var.r_n: 10, var.r_p: 10}
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

    t_eval = np.linspace(0, 7050, 7051)

    solution = pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-3).solve(model, t_eval,
                                                                            inputs={
                                                                                    "Ambient temperature [K]": 298.15,
                                                                                    "Initial temperature [K]": 298.15})
    voltage = solution["Voltage [V]"].entries
    time = solution["Time [s]"].entries

    return time, voltage


# Loss function (mean square error)
def cost_function(params, experimental_data):
    total_loss = 0
    for current, (time_exp, U_exp, _) in experimental_data:
        time_sim, U_sim = run_battery_model(params, current)
        number = min(len(U_exp), len(U_sim))

        data1_trimmed = U_exp[:number]
        data2_trimmed = U_sim[:number]
        loss = np.sqrt(np.mean((data1_trimmed - data2_trimmed) ** 2))

        total_loss += loss

    return total_loss

def PSO(num_particles, num_iterations, experimental_data, initial_values):
    # Initialize particles and velocities

    w_max = 0.9
    w_min = 0.4
    c1_initial, c2_initial = 2.5, 0.5
    c1_final, c2_final = 0.5, 2.5
    initial_loss = cost_function(initial_values, experimental_data)
    print(initial_loss)
    particles = np.array([initial_values + 0.05 * np.random.randn(6) for _ in range(num_particles)])
    velocities = np.random.rand(num_particles, 6) * 0.03
    personal_best_positions = np.copy(particles)
    personal_best_scores = np.array([cost_function(p, experimental_data) for p in particles])

    # Initialize global best
    global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)

    # PSO main loop
    for iteration in range(num_iterations):
        w = w_max - (w_max - w_min) * ((iteration + 1) / num_iterations)
        c1 = c1_initial - (c1_initial - c1_final) * (iteration + 1 / num_iterations)
        c2 = c2_initial + (c2_final - c2_initial) * (iteration + 1 / num_iterations)

        for i in range(num_particles):

            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = w * velocities[i] + c1 * r1 * (personal_best_positions[i] - particles[i]) + c2 * r2 * (
                        global_best_position - particles[i])

            particles[i] += velocities[i]
            particles[i] = apply_bounds(particles[i])

            score = cost_function(particles[i], experimental_data)

            if score < personal_best_scores[i]:
                personal_best_positions[i] = particles[i]
                personal_best_scores[i] = score

            # Update global best
            if score < global_best_score:
                global_best_position = particles[i]
                global_best_score = score

            if iteration % 10 == 0:

                diversity_threshold = 1e-3
                if np.std(personal_best_scores) < diversity_threshold:
                    for j in range(num_particles // 5):
                        particles[j] = initial_values + 0.05 * np.random.randn(6)
                        velocities[j] = np.random.rand(6) * 0.03

                for dim in range(6):
                    perturbation = np.random.uniform(-0.03, 0.03)
                    temp_particle = np.copy(global_best_position)
                    temp_particle[dim] += perturbation
                    temp_particle = apply_bounds(temp_particle)
                    new_score = cost_function(temp_particle, experimental_data)

                    if new_score < global_best_score:
                        global_best_position = temp_particle
                        global_best_score = new_score

        print(f"Iteration {iteration + 1}/{num_iterations}, Best score: {global_best_score}")
        print("Best Parameters:", global_best_position)

    return global_best_position, global_best_score


# Load experimental data
data_begin, data_middle, data_age = load_data()
experimental_data = [(5, data_age)]

initial_values = [
    5.77,
    4.30,
    4.06,
    4.16,
    5.58,
    5.17
]

best_params, best_score = PSO(num_particles=30, num_iterations=100, experimental_data=experimental_data, initial_values=initial_values)
