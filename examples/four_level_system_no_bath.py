#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import json

sys.path.insert(0, "../")

import oqupy


def four_level_system_dynamics(end_time=10.0):
    # define constants and system parameters
    # \hbar == k_b == 1
    parameters_file = open("parameters.json")
    parameters = json.load(parameters_file)
    alpha = parameters["alpha"]  # coupling strength
    temperature = parameters["temperature"]  # environment temperature
    nu_c = parameters["nu_c"]  # environment cutoff frequency
    omega_0 = parameters["omega_0"]  # two-level system frequency (set to 0)
    gn = parameters["gn"]  # Twice Rabi splitting (light-matter coupling)
    sqrt_s = parameters["sqrt_s"]  # Huang-Rhys parameter (strength of light-matter coupling)
    omega_c = parameters["omega_c"]  # cavity frequency
    kappa = parameters["kappa"]  # field decay
    Gamma_up = parameters["Gamma_up"]  # incoherent gain (electronic gain)
    Gamma_down = parameters["Gamma_down"]  # incoherent loss (electronic dissipation)
    omega_v = parameters["omega_v"]  # vibrational system frequency
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
    dynamics = tempo_sm.compute(end_time=tf)

    return {
        "times": dynamics.times,
        "expectations": {
            "sigma_z": dynamics.expectations(np.kron(oqupy.operators.sigma("z"), I_2), real=True)[1],
            "tau_dagger_tau": dynamics.expectations(np.matmul(tau_plus, tau_minus), real=True)[1],
            "tau+tau_dagger": dynamics.expectations(tau_minus + tau_plus, real=True)[1],
            "field": dynamics.field_expectations()[1]
        }
    }


if __name__ == "__main__":
    system = four_level_system_dynamics(end_time=1500)
    times = system["times"]
    sigma_z = system["expectations"]["sigma_z"]
    tau_dagger_tau = system["expectations"]["tau_dagger_tau"]
    tau_p_tau_dagger = system["expectations"]["tau+tau_dagger"]
    field = system["expectations"]["field"]

    fig, axes = plt.subplots(4, figsize=(10, 14))
    axes[0].plot(times, sigma_z)
    axes[1].plot(times, tau_dagger_tau)
    axes[2].plot(times, tau_p_tau_dagger)
    axes[3].plot(times, np.abs(field) ** 2)
    axes[0].set_ylabel("$\\langle S_z\\rangle$")
    axes[1].set_ylabel("$\\tau^\\dagger \\tau$")
    axes[2].set_ylabel("$\\tau+\\tau^\\dagger$")
    axes[3].set_ylabel("$n$")
    axes[3].set_xlabel("$t$")
    plt.tight_layout()

    fig.savefig(os.path.join("figures", "4_level_system_no_bath.png"), bbox_inches="tight")
    plt.show()
