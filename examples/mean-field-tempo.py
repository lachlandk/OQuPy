#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.insert(0,'.')

import numpy as np
import matplotlib.pyplot as plt

import oqupy
import oqupy.operators as op

# ----------------------------------------------------------------------------

omega_cutoff = 5.0
a = 0.1
temperature = 0.1
initial_state = oqupy.operators.spin_dm("z-")
initial_field = 1.0 + 1.0j

# System parameters
gn = 0.2    # Twice Rabi splitting
gam_down = 0.01 # incoherent loss
gam_up   = 0.01 # incoherent gain
Sx=np.array([[0,0.5],[0.5,0]])
wc=0.0
kappa=0.1
end_time=1
def field_eom(t, state, field):
    sx_exp = np.matmul(Sx, state).trace().real
    return -(1j*wc+kappa)*field - 1j*gn*sx_exp
def H(t, field):
    #print(t, field)
    return 2.0 * gn * np.abs(field) * Sx

system = oqupy.TimeDependentSystemWithField(H,
        field_eom,
        #gammas = [gam_down, gam_up],
        #lindblad_operators = [oqupy.operators.sigma("-"), tempo.operators.sigma("+")]
        )
correlations = oqupy.PowerLawSD(alpha=a,
                                zeta=1,
                                cutoff=omega_cutoff,
                                cutoff_type='gaussian',
                                temperature=temperature)
bath = oqupy.Bath(oqupy.operators.sigma("z")/2.0, correlations)


tempo_parameters = oqupy.TempoParameters(dt=0.1, dkmax=20, epsrel=10**(-7))

tempo_sys = oqupy.TempoWithField(system=system,
                        bath=bath,
                        initial_state=initial_state,
                        initial_field=initial_field,
                        start_time=0.0,
                        parameters=tempo_parameters)
dynamics = tempo_sys.compute(end_time=end_time)

print(f"The the final time t = {dynamics.times[-1]:.1f} " \
      + f"the field is {dynamics.fields[-1]:.8g} and the state is:")
print(dynamics.states[-1])

t, s_x = dynamics.expectations(oqupy.operators.sigma("x")/2, real=True)
t, s_z = dynamics.expectations(oqupy.operators.sigma("z")/2, real=True)
t, fields = dynamics.field_expectations()
fig, axes = plt.subplots(2, figsize=(9,10))
axes[0].plot(t, s_z)
axes[1].plot(t, np.abs(fields)**2)
axes[1].set_xlabel('t')
axes[0].set_ylabel('<Sz>')
axes[1].set_ylabel('n')
plt.tight_layout()

plt.show()
