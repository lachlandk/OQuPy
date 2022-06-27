#!/usr/bin/env python

"""Rough script to measure two-time correlators for mean-field Hamiltonian"""

import os
import pickle
import sys

sys.path.insert(0, '../../')  # Make OQuPy accessible

import numpy as np
import matplotlib.pyplot as plt

import oqupy
import oqupy.operators as op


# Output files
data_dir = 'data'
fig_dir = 'figures'
dynamics_plotfp = os.path.join(fig_dir, 'dynamics_plot.png')
datafp = os.path.join(data_dir, 'dynamics_correlators.pkl')
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)
if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)


# Spin operators
I_2 = oqupy.operators.identity(2)
sigma_z = np.kron(oqupy.operators.sigma('z'), I_2)
sigma_p = np.kron(oqupy.operators.sigma('+'), I_2)
sigma_m = np.kron(oqupy.operators.sigma('-'), I_2)
sigma_up = np.matmul(sigma_p, sigma_m)
B_p = np.kron(I_2, oqupy.operators.create(2))
B_m = np.kron(I_2, oqupy.operators.destroy(2))
B_up = np.matmul(B_p, B_m)

# Bath parameters
nu_c = 0.15
a = 0.25
T = 0.026

# System parameters
dim = 4
wc = 0.0
gn = 0.2
gam_down = 0.01
gam_up = 0.01
kappa = 0.01

w0 = 0.0
wv = 1.0  # TODO: find a sensible value for this
sqrt_s = np.sqrt(1.0)  # TODO: this as well

# Initial conditions
initial_state = np.matmul(np.matmul(sigma_m, sigma_p), np.matmul(B_m, B_p))  # electronic and vibrational unexcited
initial_field = np.sqrt(0.05)

# Computational parameters
# N.B. Not sensible values!
ts = 1000  # steady-state time
tp = 800  # time to measure field period from (should really be in steady-state by tp)
tf = 1500  # final time
rotating_frame_freq = None  # record rotating frame freq. (used below)
dt = 0.5
dkmax = 20
epsrel = 10**(-4)


local_times = [0.0]
local_fields = [initial_field]


def store_local_field(t, a):
    if not np.isclose(t, local_times[-1]+dt):
        # in input parsing random times are passed to field_eom, avoid recording
        # these
        return
    if t > local_times[-1]:
        local_times.append(t)
        local_fields.append(a)


# Measure period of oscillation and adjust global variables w0 and wc
# For multiple systems would need to adjust each w0
def move_to_rotating_frame():
    global rotating_frame_freq, w0, wc
    start_step = round(tp/dt)
    end_step = None
    field_samples = local_fields[start_step:end_step]
    # Array of periods measured in steps, taking each period as the time between 3 intercepts of the horizontal axis
    period_steps = []
    # count number of intercepts
    intercepts = 0
    # on first intercept, record number of steps
    recorded_step = 0
    # determine where sign of real part of field changes (assume evolution continuous)
    sign_changes = np.diff(np.sign(np.real(field_samples)))
    for step, change in enumerate(sign_changes):
        # If sign changes, we have an intercept
        if change != 0:
            intercepts += 1
            # record step of first intercept (3 intercepts make 1 period)
            if intercepts == 1:
                recorded_step = step
        if intercepts == 3:
            # Period is difference between step of third intercept and step of first intercept
            period_steps.append(step-recorded_step)
            # reset counter; hopefully measure multiple periods and average to minimise numerical error
            # due to timestep not exactly aligning with intercepts
            intercepts=0
    num_periods = len(period_steps)
    if num_periods == 0:
        # Nothing to do; no periods measured (field not oscillatory)
        print('\nNo field oscillations recorded between t={} and t={}'.format(tp, ts))
        rotating_frame_freq = 0.0
        return
    elif num_periods <= 5:
        print('\nOnly {} periods recorded between t={} and t={} - rotating frame '\
              'frequency may be inaccurate.'.format(num_periods, tp, ts))
    # average period in units time (not steps)
    average_period = dt * np.average(period_steps)
    lasing_angular_freq = 2*np.pi / average_period
    phi0 = np.angle(field_samples[-2])
    phi1 = np.angle(field_samples[-1])
    # whether phase is increasing or decreasing
    lasing_direction = np.sign(phi1-phi0)
    # np.angle has discontinuity on negative Im axis, so above fails if phi0 in upper left quadrant and phi1 in bottom left
    if phi1 < -np.pi/2 and phi0 > np.pi/2:
        lasing_direction = -1
    # add corresponding angular frequency from both w0 and wc. This should result in a stationary solution
    # (add as alpha rotates at negative of rotating frame freq)
    rotating_frame_freq = lasing_direction*lasing_angular_freq
    w0 += rotating_frame_freq  # MULTI-SYSTEM GENERALISATION ?
    wc += rotating_frame_freq
    print('Adjusted w0 and wc by rotating_frame_freq {:.3f}'.format(rotating_frame_freq))


# Functions passed to tempo
def field_eom(t, state, a):
    if rotating_frame_freq is None:
        # need to keep a local copy of field so can calculate period of
        # oscillation later. TODO: implement in OQuPy itself 
        store_local_field(t, a)
    if t >= ts:
        # In steady-state, move to rotating frame and stop evolving field
        if rotating_frame_freq is None:
            move_to_rotating_frame()
        return 0.0
    expect_val = np.matmul(sigma_m, state).trace()
    return -(1j * wc + kappa) * a - 0.5j * gn * expect_val


def H_MF(t, a):
    return 0.5 * w0 * sigma_z +\
        0.5 * gn * (a * sigma_p + np.conj(a) * sigma_m) + wv * (
                       B_up + sqrt_s * np.matmul((B_m + B_p), sigma_up))


system = oqupy.TimeDependentSystemWithField(H_MF,
                                            field_eom,
                                            gammas=[lambda t: gam_down, lambda t: gam_up],
                                            lindblad_operators=[lambda t: sigma_m, lambda t: sigma_p])

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
process_tensor = oqupy.pt_tempo_compute(bath=bath,
                                        start_time=0.0,
                                        end_time=tf,
                                        parameters=pt_tempo_parameters)

# Control objects for two-time correlator measurement
control_sm = oqupy.Control(dim)
control_sp = oqupy.Control(dim)
# N.B. ts must be a float otherwise (if int) interpreted as timestep 
control_sm.add_single(float(ts), op.left_super(sigma_m))
control_sp.add_single(float(ts), op.left_super(sigma_p))

# Two sets of dynamics, one for each two-time correlator
dynamics_sm = oqupy.compute_dynamics_with_field(system=system,
                                                process_tensor=process_tensor,
                                                initial_field=initial_field,
                                                control=control_sm,
                                                start_time=0.0,
                                                initial_state=initial_state)
times, sp = dynamics_sm.expectations(np.kron(oqupy.operators.sigma('+'), I_2)/2, real=False)
ts_index = next((i for i, t in enumerate(times) if t >= ts), None)
corr_times = times[ts_index:] - ts
spsm = sp[ts_index:]
first_rotating_frame_freq = rotating_frame_freq
# reset rotating frame frequency and local storage variables
rotating_frame_freq = None
local_times = [0.0]
local_fields = [initial_field]
dynamics_sp = oqupy.compute_dynamics_with_field(system=system,
                                                process_tensor=process_tensor,
                                                initial_field=initial_field,
                                                control=control_sp,
                                                start_time=0.0,
                                                initial_state=initial_state)
times, sm = dynamics_sp.expectations(np.kron(oqupy.operators.sigma('-'), I_2)/2, real=False)
smsp = sm[ts_index:]  # <sigma^-(t) sigma^+(0)>
# consistency check
assert rotating_frame_freq == first_rotating_frame_freq
assert len(smsp) == len(spsm) == len(corr_times)

# save truncated times, correlators and parameters used by spectrum.py
save_dic = {
        'times': corr_times,
        'spsm': spsm,
        'smsp': smsp,
        'params': {
            'dt': dt,
            'wc': wc,
            'w0': w0,
            'rotating_frame_freq': rotating_frame_freq,
            'kappa': kappa,
            'gn': gn,
            }
        }

with open(datafp, 'wb') as fb:
    pickle.dump(save_dic, fb)
print('Times and correlator values saved to {}'.format(datafp))

# Plot fields and polarisation
times, s_z = dynamics_sm.expectations(np.kron(oqupy.operators.sigma('z'), I_2)/2, real=True)
_, fields = dynamics_sm.field_expectations()
fig, axes = plt.subplots(2, figsize=(9, 10))
axes[0].plot(times, s_z)
axes[1].plot(times, np.real(fields))
axes[1].set_xlabel('t')
axes[0].set_ylabel('<Sz>')
axes[1].set_ylabel('Re<a>')
axes[1].axvline(x=tp, c='g')  # corresponds to time measure period from
axes[1].axvline(x=ts, c='r')  # corresponds to time measure correlators from
fig.savefig(dynamics_plotfp, bbox_inches='tight')



