#!/usr/bin/env python

"""Script to calculate spectral weight and photoluminescence from two-time correlators"""

import pickle

import matplotlib.pyplot as plt
import numpy as np

plt.rc('text')
plt.rc('font', **{'size': 14})

# input file
inputfp = 'data/dynamics_correlators.pkl'

# load
with open(inputfp, 'rb') as fb:
    data = pickle.load(fb)
times = data['times']
spsm = data['spsm']
smsp = data['smsp']
params = data['params']
lasing_dic = data['lasing_dic']

dt = params['dt']
# calculations in rotating frame, plot in original frame
wc_rotating = params['wc'] + lasing_dic['rotating_frame_freq']
w0_rotating = params['w0'] + lasing_dic['rotating_frame_freq']
nus = np.pad(2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(times), d=dt)), (0, 4 * len(times)))
plot_nus = 1e3 * (nus - lasing_dic['rotating_frame_freq'])  # units meV

# In lasing phase have singularity in Keldysh part -> PL
# lasing occurs at 0 in rotating frame i.e. at params['rotating_frame_freq']
lasing_index = None
if lasing_dic['lasing']:
    lasing_index = next((i for i, nu in enumerate(nus) if nu >= 0.0), None)

# calculate self-energies
fft_smsp = np.pad(np.fft.fftshift(dt * np.fft.ifft(smsp, norm='forward')), (0, 4 * len(times)))
fft_spsm_conjugate = np.pad(np.fft.fftshift(dt * np.fft.ifft(np.conjugate(spsm), norm='forward')), (0, 4 * len(times)))
# Sigma^{-+}
energy_mp = -(1j/4)*params['gn']**2*(fft_smsp-fft_spsm_conjugate)
# Sigma^{--} 
energy_mm = -(1j/2)*params['gn']**2*(np.real(fft_smsp+fft_spsm_conjugate))


# calculate unperturbed Green's functions
# inverse of non-interacting retarded function
def D0RI(nu):
    return nu-wc_rotating+1j*params['kappa']
# N.B. DOIK = - DORI * DOK * DOA happens to be a constant


def D0IK(nu):
    return 2j * params['kappa']


# calculate interacting green's functions
inverse_retarded = D0RI(nus) - energy_mp
keldysh_inverse = D0IK(nus) - energy_mm
retarded = 1/inverse_retarded
advanced = np.conjugate(retarded)
keldysh = - retarded * keldysh_inverse * advanced

# calculate pl and spectral weight
# pl = (1j/2) * (keldysh - retarded + advanced)
# use explicit formula for PL in terms of self energies 
pl = (np.imag(energy_mp) - 0.5*np.imag(energy_mm)) \
        / np.abs((nus - wc_rotating) + 1j * params['kappa'] - energy_mp)**2
# spectral weight
spectral_weight = -2*np.imag(retarded)


# Plot correlators, spectral weight, photoluminescence 
fig1, axes1 = plt.subplots(2, figsize=(9, 6), sharex="all")
fig2, ax2 = plt.subplots(figsize=(9, 4))
fig3, ax3 = plt.subplots(figsize=(9, 4))

axes1[1].set_xlabel('$t$')
closing_label=r'\rangle_c' if lasing_dic['lasing'] else r'\rangle' # indicated 'connected' i.e. subtracted long-time value
axes1[0].plot(times, np.real(smsp), label=r'Re$\langle \sigma^-(t) \sigma^+(0) {}$'.format(closing_label))
axes1[0].plot(times, np.imag(smsp), label=r'Im$\langle \sigma^-(t) \sigma^+(0) {}$'.format(closing_label))
axes1[1].plot(times, np.real(spsm), label=r'Re$\langle \sigma^+(t) \sigma^-(0) {}$'.format(closing_label))
axes1[1].plot(times, np.imag(spsm), label=r'Im$\langle \sigma^+(t) \sigma^-(0) {}$'.format(closing_label))
axes1[0].legend()
axes1[1].legend()
ax2.set_xlabel(r'$\nu$')
ax2.set_ylabel(r'$\varrho$', rotation=0, labelpad=20)
ax2.plot(plot_nus, spectral_weight/np.max(spectral_weight)) # Normalise
ax2.set_xlim([-300, 300])
ax3.set_xlabel(r'$\nu$')
ax3.set_ylabel(r'$\mathcal{L}$', rotation=0, labelpad=20)
ax3.plot(plot_nus, pl/np.max(pl)) # Normalise
# indicate lasing
if lasing_dic['lasing']:
    if lasing_dic['spsm_subtracted'] < lasing_dic['zero_cutoff']:
        # Inconsistency e.g. system was tending to normal state but ts value set
        # too short
        print('WARNING: Lasing was indicated but squared steady-state value of correlator'\
                ' less than {} - consider running simulation for longer times. Not adding'\
                ' a lasing peak to spectrum.'.format(lasing_dic['zero_cutoff']))
    else:
        # plot a delta peak to indicate lasing on spectrum
        pl_at_lasing = pl[lasing_index]/np.max(pl)
        lasing_freq = plot_nus[lasing_index]
        ax3.set_ylim([None, 1.25*ax3.get_ylim()[1]])
        arrow_height = ax3.get_ylim()[1]
        ax3.annotate('', xy=(lasing_freq, arrow_height), xytext=(lasing_freq, 0.99*pl_at_lasing),
                arrowprops=dict(arrowstyle='-', color=ax3.lines[-1].get_color(), linewidth=2.5),
                annotation_clip=False)
ax3.set_xlim([-300, 300])
fig1.savefig('figures/correlators.png', bbox_inches='tight')
fig2.savefig('figures/spectral_weight.png', bbox_inches='tight')
fig3.savefig('figures/photoluminescence.png', bbox_inches='tight')
