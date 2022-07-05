#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, ".")

import oqupy


def four_level_system_dynamics(end_time=10.0):
    # define constants and system parameters
    # \hbar == k_b == 1
    end_time = end_time
    alpha = 0.25  # coupling strength
    temperature = 0.026  # environment temperature
    nu_c = 0.15  # environment cutoff frequency
    omega_0 = 0.0  # two-level system frequency (set to 0)
    gn = 0.2  # Twice Rabi splitting (light-matter coupling)
    sqrt_s = 1.0  # Huang-Rhys parameter (strength of light-matter coupling)
    omega_c = 0.0  # cavity frequency
    kappa = 0.01  # field decay
    Gamma_up = 0.01  # incoherent gain (electronic gain)
    Gamma_down = 0.01  # incoherent loss (electronic dissipation)
    omega_v = 1.0  # vibrational system frequency
    
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
    initial_state = np.matmul(np.matmul(sigma_minus, sigma_plus), np.matmul(tau_minus, tau_plus))  # electronic and vibrational unexcited

    def Hamiltonian(_, a):
        return omega_0 * sigma_up_dm + gn * (
                a * sigma_plus + np.conj(a) * sigma_minus) + omega_v * (
                       tau_up_dm + sqrt_s * np.matmul((sigma_plus + sigma_minus), tau_x))

    def field_EOM(_, state, a):
        expectation_value_sigma_minus = np.matmul(sigma_minus, state).trace()
        return -(1j * omega_c + kappa) * a - 0.5j * gn * expectation_value_sigma_minus

    gammas = [lambda t: Gamma_down, lambda t: Gamma_up]
    lindblad_operators = [lambda t: sigma_minus, lambda t: sigma_plus]

    td_system = oqupy.TimeDependentSystemWithField(Hamiltonian, field_EOM, gammas=gammas,
                                                   lindblad_operators=lindblad_operators)
    correlations = oqupy.PowerLawSD(alpha=alpha, zeta=1, cutoff=nu_c, cutoff_type="gaussian",
                                    temperature=temperature)
    bath = oqupy.Bath(1/np.sqrt(2) * tau_x, correlations)
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

    fig.savefig(os.path.join("figures", "4_level_system_SR_coupling.png"), bbox_inches="tight")
    plt.show()
