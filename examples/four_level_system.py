#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import oqupy

# define constants and system parameters
# \hbar == k_b == 1
# alpha = 0.25  # coupling strength (dimensionless)
# nu_c = 227.9  # environment cutoff frequency (1/ps)
# T = 39.3  # environment temperature (1/ps) == 300K
omega_0 = 0.0  # two-level system frequency (set to 0)
# omega_c = -30.4  # cavity frequency (1/ps)
# Omega = 303.9  # light-matter coupling (1/ps)
sqrt_s = np.sqrt(1)  # Huang-Rhys parameter (strength of light-matter coupling)

# kappa = 15.2  # field decay rate (1/ps)
# Gamma_down = 30.4  # incoherent loss (electronic dissipation) (1/ps)
# Gamma_up = 0.8 * Gamma_down  # # incoherent gain (electronic gain) (1/ps)
v_gamma_down = 0.1
v_gamma_up = 0.1

omega_v = 1  # vibrational system frequency
gn = 0.2  # Twice Rabi splitting (light-matter coupling)

# omega_cutoff = 5.0  # environment cutoff frequency
alpha = 0.1  # coupling strength
T = 0.1  # environment temperature
# initial_state = oqupy.operators.spin_dm("z-")
initial_field = 1.0 + 1.0j
Gamma_down = 0.01
Gamma_up = 0.01
omega_c = 0.0  # cavity frequency
kappa = 0.1  # field decay
# Sx = np.array([[0, 0.5], [0.5, 0]])

sigma_plus_2D = oqupy.operators.sigma("-")
sigma_minus_2D = oqupy.operators.sigma("+")
I_2D = oqupy.operators.identity(2)

sigma_plus = np.kron(sigma_plus_2D, I_2D)
sigma_minus = np.kron(sigma_minus_2D, I_2D)
sigma_up_dm = np.matmul(sigma_plus, sigma_minus)
B_plus = np.kron(I_2D, sigma_plus_2D)
B_minus = np.kron(I_2D, sigma_minus_2D)
B_up_dm = np.matmul(B_plus, B_minus)

end_time = 10
# initial_field = np.sqrt(0.01)
initial_state = np.matmul(sigma_minus, sigma_plus)  # spin down


def Hamiltonian(_, field):
    return omega_0 * sigma_up_dm + gn * (field * sigma_plus + np.conj(field) * sigma_minus) + omega_v * (B_up_dm + sqrt_s * np.matmul((B_minus + B_plus), sigma_up_dm))


def field_EOM(_, state, field):
    expectation_value_sigma_minus = np.matmul(sigma_minus, state).trace().real
    return -(1j * omega_c + kappa) * field - 0.5j * gn * expectation_value_sigma_minus


gammas = [lambda _: Gamma_down, lambda _: Gamma_up, lambda _: v_gamma_down, lambda _: v_gamma_up]
lindblad_operators = [lambda _: sigma_minus, lambda _: sigma_plus, lambda _: B_minus, lambda _: B_plus]

system = oqupy.TimeDependentSystemWithField(Hamiltonian, field_EOM, gammas=gammas, lindblad_operators=lindblad_operators)

omega_cutoff = 5.0  # environment cutoff frequency
a = 0.1  # coupling strength
temperature = 0.1  # environment temperature
correlations = oqupy.PowerLawSD(alpha=a, zeta=1, cutoff=omega_cutoff, cutoff_type="gaussian", temperature=temperature)

bath = oqupy.Bath(np.kron(oqupy.operators.sigma("z"), I_2D) / 2.0, correlations)

pt_tempo_parameters = oqupy.TempoParameters(dt=0.01, dkmax=20, epsrel=10 ** (-7))

process_tensor = oqupy.pt_tempo_compute(bath=bath, start_time=0.0, end_time=1, parameters=pt_tempo_parameters)

control = None  # oqupy.Control(2)

dynamics = oqupy.compute_dynamics_with_field(
    system=system,
    process_tensor=process_tensor,
    initial_field=initial_field,
    control=control,
    start_time=0.0,
    initial_state=initial_state)

print(f"The the final time t = {dynamics.times[-1]:.1f} the field is {dynamics.fields[-1]:.8g} and the state is: {dynamics.states[-1]}")

t, s_x = dynamics.expectations(np.kron(oqupy.operators.sigma("x"), I_2D) / 2, real=True)
_, s_z = dynamics.expectations(np.kron(oqupy.operators.sigma("z"), I_2D) / 2, real=True)
_, fields = dynamics.field_expectations()
fig, axes = plt.subplots(2, figsize=(9, 10))
axes[0].plot(t, s_z)
axes[1].plot(t, np.abs(fields) ** 2)
axes[1].set_xlabel("t")
axes[0].set_ylabel("<S_z>")
axes[1].set_ylabel("n")
plt.tight_layout()

plt.show()
