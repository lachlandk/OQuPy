#!/usr/bin/env python

import os, sys, json
import matplotlib.pyplot as plt
# plt.style.use('matplotlib_style')
from scipy.integrate import quad
MAX_SUBINTERVALS=1000
MAX_CYCLES=100
from scipy.constants import pi
from scipy.special import dawsn
import numpy as np
from pprint import pprint

# Bath parameters
a = 0.25    # coupling strength
nuc = 0.15  # cut-off frequency (eV)
T = 0.026   # 300K

FIG_DIR = 'figures'

# Spectral density
def J(nu):
    return 2 * a * nu * np.exp( -(nu/nuc)**2 )
# Real part of C(s) integrand
def CRe(nu, s):
    if T == 0.0:
        return J(nu) * np.cos(nu * s)
    return J(nu) * np.cos(nu * s) / np.tanh(nu/(2 * T))
# Real part of C(s) integrand without cosine factor
def CRe2(nu):
    if T == 0.0:
        return J(nu)
    return J(nu) / np.tanh(nu/(2 * T))
# Im part of C(s) integrand
def CIm(nu):
    return - J(nu) * np.sin(nu * s)
# Im part of C(s) integrand without sine factor
def CIm2(nu):
    return - J(nu)
# Real and imaginary parts of C(s) integral
# More efficient to include cosine/sine as 'weight' in scipy.integrate.quad, but
# it doesn't always work
def C_re_int(s):
    #return quad(CRe2, 0, np.inf, weight='cos', wvar=s, limit=MAX_SUBINTERVALS, limlst=MAX_CYCLES) # stack smashing error
    return quad(CRe, 0, np.inf, args=s, limit=MAX_SUBINTERVALS)[0] # select result from tuple (result, error)
def C_im_int(s):
    return quad(CIm2, 0, np.inf, weight='sin', wvar=s, limit=MAX_SUBINTERVALS, limlst=MAX_CYCLES)[0]
# Correlation function (3.39)
def C(s):
    return C_re_int(s) + 1j * C_im_int(s)
# Functions used to check Gamma integrand sufficiently decayed at some upper limit UP_max
def re_integrand(s,x):
    return C_re_int(s)*np.cos(s*x)-C_im_int(s)*np.sin(s*x)
def im_integrand(s,x):
    return C_im_int(s)*np.cos(s*x)+C_re_int(s)*np.sin(s*x)
# Final integral to be performed ('x' variable instead of 'lambda'
def Gamma(x, UL_max=1e3, verbose=False):
    # UL_max is default cut-off to avoid numerical error which swamps integrand for large s
    # Try to calculate a better upper limit when integrand within 1e-12 of zero
    UL = next((s for s in np.linspace(1, UL_max) \
            if np.isclose(np.abs(re_integrand(s,x)) + np.abs(im_integrand(s,x)), 0.0, atol=1e-12)),\
            UL_max)
    # Real part of integral (expand e^(i x s) * C(s))
    re_int = quad(C_re_int, 0, UL, weight='cos', wvar=x, # N.B. -
            limit=MAX_SUBINTERVALS, limlst=MAX_CYCLES)[0] -\
                    quad(C_im_int, 0, UL, weight='sin', wvar=x,
                    limit=MAX_SUBINTERVALS, limlst=MAX_CYCLES)[0]
    im_int = quad(C_re_int, 0, UL, weight='sin', wvar=x,
            limit=MAX_SUBINTERVALS, limlst=MAX_CYCLES)[0] + \
                    quad(C_im_int, 0, UL, weight='cos', wvar=x,
                    limit=MAX_SUBINTERVALS, limlst=MAX_CYCLES)[0]
    if verbose:
        print('  - Gamma({:.2g}) calculated (integrand at upper limit={:.0f}: {:.0e}+1j*{:.0e})'.format(x,UL,re_integrand(UL,x),im_integrand(UL, x))) 
    return re_int + 1j * im_int

def plot_Gamma_integrands(w=0.1):
    print('Plotting Gamma integrands at lambda = {}...'.format(w), end=' ', flush=True)
    fig, ax = plt.subplots(figsize=(9,6))
    ts = np.linspace(0, 1e3, num=500)
    ys_re = [re_integrand(t,w) for t in ts] 
    ys_im = [im_integrand(t,w) for t in ts] 
    ax.plot(ts, np.abs(ys_re), label=r'\(|\text{Re}|\)')
    ax.plot(ts, np.abs(ys_im), label=r'\(|\text{Im}|\)')
    ax.set_title(r'\(\Gamma(\lambda)\) \rm{{integrand at }} \(\lambda={}\)'.format(w))
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlabel(r'\(s\)')
    fig.savefig(os.path.join(FIG_DIR, 'integrand.pdf'), bbox_inches='tight')
    print('done')


def plot_SD():
    print('Plotting spectral density...', end=' ', flush=True)
    fig, ax = plt.subplots(figsize=(9,6))
    nus = np.linspace(0, 4*nuc, num=500)
    Js = J(nus)
    ax.plot(1e3*nus, Js, label=r'\(J(\nu)\)')
    #ax.axvline(x=omega_nu)        
    #ax.set_xticks([0,150,300,450,600])
    ax.set_yticks([])
    ax.legend()
    ax.set_xlabel(r'\(\nu\) \rm{(meV)}')
    fig.savefig(os.path.join(FIG_DIR, 'spectral_density.pdf'), bbox_inches='tight')
    print('done')

def plot_correlation():
    print('Plotting bath correlation function...', end=' ', flush=True)
    fig, ax = plt.subplots(figsize=(9,6))
    ss = np.linspace(0, 6*(1/nuc), num=500)
    C_vec = np.vectorize(C)
    Cs = C_vec(ss)
    ax.plot(ss, np.real(Cs), label=r'Re')
    ax.plot(ss, np.imag(Cs), label=r'Im')
    ax.spines['bottom'].set_position('zero')
    ax.legend()
    ax.set_xlabel(r'\(s\)')
    ax.set_ylabel(r'\(C(s)\)', rotation=0)
    fig.savefig(os.path.join(FIG_DIR, 'Cs.pdf'), bbox_inches='tight')
    print('done')


if __name__=='__main__':
    if not os.path.isdir(FIG_DIR):
        os.makedirs(FIG_DIR)
    #plot_Gamma_integrands()
    #plot_SD()
    #plot_correlation()
    omega_nu = 0.25
    print('Gamma(0)     = {:.3g}'.format(Gamma(0)))
    print('Gamma(w_nu)  = {:.3g}'.format(Gamma(omega_nu)))
    print('Gamma(-w_nu) = {:.3g}'.format(Gamma(-omega_nu)))

