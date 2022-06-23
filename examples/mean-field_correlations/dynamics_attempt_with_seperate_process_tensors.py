#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.insert(0,'../../') # Make OQuPy accessible

import numpy as np
import matplotlib.pyplot as plt

import oqupy
import oqupy.operators as op

# Output files
dynamics_plotf = 'dynamics_plot.pdf'
spectrum_plotf = 'spectrum_plot.pdf'


# Spin operators
sigma_z = oqupy.operators.sigma('z')
sigma_p = oqupy.operators.sigma('+')
sigma_m = oqupy.operators.sigma('-')

# Bath parameters
nu_c = 0.15
a = 0.25
T = 0.026

# System parameters
dim = 2
wc = 0.0
gn = 0.2    
gam_down = 0.01 
gam_up   = 0.01
kappa = 0.01

w0 = 0.0

# Initial conditions
initial_state = oqupy.operators.spin_dm('z-')
initial_field = np.sqrt(0.05)

# Computational parameters
ts = 10 # steady-state time
tf = 20 # final time
dt = 0.05
dkmax = 20
epsrel = 10**(-7)

# Measure period of oscillation and adjust global variable wc

# Functions passed to tempo
def field_eom(t, state, a):
    if t > ts:
        # In steady-state, move to rotating frame and stop evolving field
        #move_to_rotating_frame()
        return 0.0
    expect_val = np.matmul(sigma_m, state).trace()
    return -(1j * wc + kappa) * a - 0.5j * gn * expect_val
def H_MF(t, a):
    return 0.5 * w0 * sigma_z +\
        0.5 * gn * (a * sigma_p + np.conj(a) * sigma_m)

# Control object for two-time correlator measurement
control = oqupy.Control(dim)
control.add_single(float(ts), op.left_super(sigma_m))
# N.B. ts must be a float otherwise (if int) interpreted as timestep 

system = oqupy.TimeDependentSystemWithField(H_MF,
        field_eom,
        gammas = [lambda t: gam_down, lambda t: gam_up],
        lindblad_operators = [lambda t: sigma_m, lambda t: sigma_p]
        )
correlations = oqupy.PowerLawSD(alpha=a,
                                zeta=1,
                                cutoff=nu_c,
                                cutoff_type='gaussian',
                                temperature=T)
bath = oqupy.Bath(0.5 * sigma_z, correlations)

pt_tempo_parameters = oqupy.TempoParameters(
    dt=dt,
    dkmax=dkmax,
    epsrel=epsrel)

# compute PT to final time tf
process_tensor1 = oqupy.pt_tempo_compute(bath=bath,
                                        start_time=0.0,
                                        end_time=ts,
                                        parameters=pt_tempo_parameters)

process_tensor2 = oqupy.pt_tempo_compute(bath=bath,
                                        start_time=ts,
                                        end_time=tf,
                                        parameters=pt_tempo_parameters)

process_tensor_all = oqupy.pt_tempo_compute(bath=bath,
                                        start_time=0.0,
                                        end_time=tf,
                                        parameters=pt_tempo_parameters)
no_control = oqupy.Control(2)
dynamics1 = oqupy.compute_dynamics_with_field(
    system=system,
    process_tensor=process_tensor1,
	initial_field=initial_field,
    control=control,
    start_time=0.0,
    initial_state=initial_state)
steady_field = dynamics1.fields[-1]
steady_state = dynamics1.states[-1]
#steady_state = np.matmul(sigma_m, steady_state)
dynamics2 = oqupy.compute_dynamics_with_field(
    system=system,
    process_tensor=process_tensor2,
	initial_field=steady_field,
    control=control,
    start_time=ts,
    initial_state=steady_state)
dynamics3 = oqupy.compute_dynamics_with_field(
    system=system,
    process_tensor=process_tensor_all,
	initial_field=initial_field,
    control=control,
    start_time=0.0,
    initial_state=initial_state)
t1, sz1 = dynamics1.expectations(oqupy.operators.sigma("x")/2, real=True) 
t2, sz2 = dynamics2.expectations(oqupy.operators.sigma("x")/2, real=True) 
ta, sza = dynamics3.expectations(oqupy.operators.sigma("x")/2, real=True) 
t1, fields1 = dynamics1.field_expectations()
t2, fields2 = dynamics2.field_expectations()
ta, fieldsa = dynamics3.field_expectations()
t = np.concatenate((t1,t2))
s_z = np.concatenate((sz1,sz2))
fields = np.concatenate((fields1,fields2))
fig, axes = plt.subplots(2, figsize=(9,10))
axes[0].plot(t, s_z)
axes[0].plot(ta, sza)
axes[1].plot(t, np.abs(fields)**2)
axes[1].plot(ta, np.abs(fieldsa)**2)
axes[1].set_xlabel('t')
axes[0].set_ylabel('<Sz>')
axes[1].set_ylabel('n')
fig.savefig(dynamics_plotf, bbox_inches='tight')
print('Dynamics plot saved to {}'.format(dynamics_plotf))


