#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import json

sys.path.insert(0, '../')  # Make OQuPy accessible

import oqupy


def n_level_system_dynamics(n, end_time=10.0):
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
    v_gamma_up = parameters["v_gamma_up"]
    v_gamma_down = parameters["v_gamma_down"]
    parameters_file.close()
    
    # compute operators
    I_2D = oqupy.operators.identity(2)
    I_ND = oqupy.operators.identity(n)

    sigma_plus = np.kron(oqupy.operators.sigma("+"), I_ND)
    sigma_minus = np.kron(oqupy.operators.sigma("-"), I_ND)
    sigma_up_dm = np.matmul(sigma_plus, sigma_minus)

    B_plus = np.kron(I_2D, oqupy.operators.create(n))
    B_minus = np.kron(I_2D, oqupy.operators.destroy(n))
    B_up_dm = np.matmul(B_plus, B_minus)

    # compute dynamics
    initial_field = np.sqrt(0.05)
    initial_state = np.matmul(np.matmul(sigma_minus, sigma_plus),
                              np.matmul(B_minus, B_plus))  # electronic and vibrational unexcited

    def Hamiltonian(_, a):
        return omega_0 * sigma_up_dm + gn * (
                a * sigma_plus + np.conj(a) * sigma_minus) + omega_v * (
                       B_up_dm + sqrt_s * np.matmul((B_minus + B_plus), sigma_up_dm))

    def field_EOM(_, state, a):
        expectation_value_sigma_minus = np.matmul(sigma_minus, state).trace()
        return -(1j * omega_c + kappa) * a - 0.5j * gn * expectation_value_sigma_minus

    gammas = [lambda t: Gamma_down, lambda t: Gamma_up, lambda t: v_gamma_down,
              lambda t: v_gamma_up]
    lindblad_operators = [lambda t: sigma_minus, lambda t: sigma_plus, lambda t: B_minus, lambda t: B_plus]

    td_system = oqupy.TimeDependentSystemWithField(Hamiltonian, field_EOM, gammas=gammas,
                                                   lindblad_operators=lindblad_operators)
    correlations = oqupy.PowerLawSD(alpha=alpha, zeta=1, cutoff=nu_c, cutoff_type="gaussian",
                                    temperature=temperature)
    bath = oqupy.Bath(np.kron(oqupy.operators.sigma("z"), I_ND) / 2.0, correlations)
    pt_tempo_parameters = oqupy.TempoParameters(dt=0.5, dkmax=20, epsrel=10 ** (-4))
    process_tensor = oqupy.pt_tempo_compute(bath=bath, start_time=0.0, end_time=end_time,
                                            parameters=pt_tempo_parameters)

    dynamics = oqupy.compute_dynamics_with_field(
        system=td_system,
        process_tensor=process_tensor,
        control=None,
        start_time=0.0,
        initial_state=initial_state,
        initial_field=initial_field)

    return {
        "times": dynamics.times,
        "expectations": {
            "sigma_z": dynamics.expectations(np.kron(oqupy.operators.sigma("z"), I_ND), real=True)[1],
            "B_dagger_B": dynamics.expectations(np.matmul(B_plus, B_minus), real=True)[1],
            "B+B_dagger": dynamics.expectations(B_minus + B_plus, real=True)[1],
            "field": dynamics.field_expectations()[1]
        }
    }


if __name__ == "__main__":
    system = n_level_system_dynamics(2, end_time=1500)
    times = system["times"]
    sigma_z = system["expectations"]["sigma_z"]
    B_dagger_B = system["expectations"]["B_dagger_B"]
    BpB_dagger = system["expectations"]["B+B_dagger"]
    field = system["expectations"]["field"]

    fig, axes = plt.subplots(4, figsize=(10, 14))
    axes[0].plot(times, sigma_z)
    axes[1].plot(times, B_dagger_B)
    axes[2].plot(times, BpB_dagger)
    axes[3].plot(times, np.abs(field) ** 2)
    axes[0].set_ylabel("$\\langle S_z\\rangle$")
    axes[1].set_ylabel("$B^\\dagger B$")
    axes[2].set_ylabel("$B+B^\\dagger$")
    axes[3].set_ylabel("$n$")
    axes[3].set_xlabel("$t$")
    plt.tight_layout()

    fig.savefig(os.path.join("figures", "n_level_system.png"), bbox_inches="tight")
    plt.show()
