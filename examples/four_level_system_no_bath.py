#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import json

sys.path.insert(0, "../")

import oqupy

if __name__ == "__main__":
    # define constants and system parameters
    # \hbar == k_b == 1
    parameters_file = open("parameters.json")
    parameters = json.load(parameters_file)
    sqrt_s = float(sys.argv[1]) if len(sys.argv) >= 2 else parameters["sqrt_s"]  # Huang-Rhys parameter (strength of light-matter coupling)
    nu_c = float(sys.argv[2]) if len(sys.argv) >= 3 else parameters["nu_c"]  # environment cutoff frequency
    omega_v = float(sys.argv[3]) if len(sys.argv) >= 4 else parameters["omega_v"]  # vibrational system frequency
    omega_c = float(sys.argv[4]) if len(sys.argv) >= 5 else parameters["omega_c"]  # cavity frequency
    alpha = parameters["alpha"]  # coupling strength
    temperature = parameters["temperature"]  # environment temperature
    omega_0 = parameters["omega_0"]  # two-level system frequency (set to 0)
    gn = parameters["gn"]  # Twice Rabi splitting (light-matter coupling)
    kappa = parameters["kappa"]  # field decay
    Gamma_up = parameters["Gamma_up"]  # incoherent gain (electronic gain)
    Gamma_down = parameters["Gamma_down"]  # incoherent loss (electronic dissipation)
    v_gamma_up = 1.63e-06  # from calc_gamma.py
    v_gamma_down = 0.0244

    # computational parameters
    dt = parameters["dt"]
    dkmax = parameters["dkmax"]
    epsrel = 10 ** parameters["epsrel"]
    parameters_file.close()
    parameters_file.close()

    # compute operators
    I_2 = oqupy.operators.identity(2)

    sigma_plus = np.kron(oqupy.operators.sigma("+"), I_2)
    sigma_minus = np.kron(oqupy.operators.sigma("-"), I_2)
    sigma_up_dm = np.matmul(sigma_plus, sigma_minus)

    tau_plus = np.kron(I_2, oqupy.operators.sigma("+"))
    tau_minus = np.kron(I_2, oqupy.operators.sigma("-"))
    tau_up_dm = np.matmul(tau_plus, tau_minus)
    tau_x = tau_plus + tau_minus

    # compute dynamics
    initial_field = np.sqrt(0.05)
    initial_state = np.matmul(np.matmul(sigma_minus, sigma_plus),
                              np.matmul(tau_minus, tau_plus))  # electronic and vibrational unexcited


    def Hamiltonian(_, a):
        return omega_0 * sigma_up_dm + gn * (
                a * sigma_plus + np.conj(a) * sigma_minus) + omega_v * (
                       tau_up_dm + sqrt_s * np.matmul((sigma_plus + sigma_minus), tau_x))


    def field_EOM(_, state, a):
        expectation_value_sigma_minus = np.matmul(sigma_minus, state).trace()
        return -(1j * omega_c + kappa) * a - 0.5j * gn * expectation_value_sigma_minus


    gammas = [lambda t: Gamma_down, lambda t: Gamma_up, lambda t: v_gamma_down, lambda t: v_gamma_up]
    lindblad_operators = [lambda t: sigma_minus, lambda t: sigma_plus,
                          lambda t: np.sqrt(v_gamma_down) * (tau_minus + sqrt_s * np.matmul(sigma_plus, sigma_minus)),
                          lambda t: np.sqrt(v_gamma_up) * (tau_plus + sqrt_s * np.matmul(sigma_plus, sigma_minus))]

    td_system = oqupy.TimeDependentSystemWithField(Hamiltonian, field_EOM, gammas=gammas,
                                                   lindblad_operators=lindblad_operators)
    correlations = oqupy.PowerLawSD(alpha=alpha, zeta=1, cutoff=nu_c, cutoff_type="gaussian",
                                    temperature=temperature)
    bath = oqupy.Bath(oqupy.operators.identity(4), correlations)
    tempo_parameters = oqupy.TempoParameters(dt=0.5, dkmax=20, epsrel=10 ** (-4))
    # process_tensor = oqupy.pt_tempo_compute(bath=bath, start_time=0.0, end_time=end_time, parameters=pt_tempo_parameters)

    # dynamics = oqupy.compute_dynamics_with_field(
    #     system=td_system,
    #     process_tensor=process_tensor,
    #     control=None,
    #     start_time=0.0,
    #     initial_state=initial_state,
    #     initial_field=initial_field)

    tempo_sm = oqupy.TempoWithField(system=td_system,
                                    bath=bath,
                                    initial_field=initial_field,
                                    initial_state=initial_state,
                                    start_time=0.0,
                                    parameters=tempo_parameters)
    dynamics = tempo_sm.compute(end_time=1500)

    times = dynamics.times
    trace_delta = np.abs(np.trace(dynamics.states[-1]) - np.trace(dynamics.states[0]))

    fig, axes = plt.subplots(4, figsize=(10, 14))
    fig.suptitle(f"Four level system no bath: sqrt_s={sqrt_s}, $\\nu_c$={nu_c}, $\\omega_v$={omega_v}, $\\omega_c$={omega_c}, trace delta = {trace_delta:.3f}")
    axes[0].plot(times, dynamics.expectations(np.kron(oqupy.operators.sigma("z"), I_2), real=True)[1])
    axes[1].plot(times, dynamics.expectations(np.matmul(tau_plus, tau_minus), real=True)[1])
    axes[2].plot(times, dynamics.expectations(tau_minus + tau_plus, real=True)[1])
    axes[3].plot(times, np.abs(dynamics.field_expectations()[1]) ** 2)
    axes[0].set_ylabel("$\\langle S_z\\rangle$")
    axes[1].set_ylabel("$\\langle\\tau^\\dagger \\tau\\rangle$")
    axes[2].set_ylabel("$\\langle\\tau+\\tau^\\dagger\\rangle$")
    axes[3].set_ylabel("$n$")
    axes[3].set_xlabel("$t$")
    plt.tight_layout()

    fig.savefig(os.path.join("figures", f"4_level_system_no_bath_sqrt_s{sqrt_s}_nu_c{nu_c}_omega_v{omega_v}_omega_c{omega_c}.png"), bbox_inches="tight")
    plt.show()
