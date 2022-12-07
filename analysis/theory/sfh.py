

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm
import cmasher as cmr
from astropy.table import Table

import sys
import os

from synthesizer.filters import TopHatFilterCollection
from synthesizer.grid import SpectralGrid
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh, generate_instant_sfzh
from synthesizer.parametric.galaxy import SEDGenerator
from synthesizer.plt import single, single_histxy, mlabel
from unyt import yr, Myr


if __name__ == '__main__':

    fs = [('FUV', {'lam_min': 1400, 'lam_max': 1600})]

    # -------------------------------------------------
    # --- calcualte the EW for a given line as a function of age

    model = 'bpass-v2.2.1-bin_chab-300_cloudy-v17.03_log10Uref-2'
    grid = SpectralGrid(model)

    fc = TopHatFilterCollection(fs, new_lam=grid.lam)

    log10durations = np.arange(0., 4., 0.1)

    fig, ax = single()

    handles = []

    for sfh_model in ['Constant', 'Increasing Exponential', 'Decreasing Exponential', 'Instantaneous']:

        if sfh_model == 'Constant':
            label = rf'$\rm Constant$'
            sfh_f = SFH.Constant
            sfh_p = {}
            ls = '-'

        if sfh_model == 'Increasing Exponential':
            label = rf'$\rm Increasing\ Exponential\ (\tau=100\ Myr)$'
            sfh_f = SFH.TruncatedExponential
            sfh_p = {'tau': 100*Myr}
            ls = '--'

        if sfh_model == 'Decreasing Exponential':
            label = rf'$\rm Decreasing\ Exponential\ (\tau=-100\ Myr)$'
            sfh_f = SFH.TruncatedExponential
            sfh_p = {'tau': -100*Myr}
            ls = '-.'

        if sfh_model == 'Instantaneous':
            label = rf'$\rm Instantaneous$'
            ls = ':'

        norm = mpl.colors.Normalize(vmin=-5, vmax=-1.5)
        cmap = cmr.get_sub_cmap('cmr.sunburst', 0.05, 0.85)

        for log10Z in [-4., -3., -2.]:

            color = cmap(norm(log10Z))

            if sfh_model == 'Constant':
                handles.append(mlines.Line2D([], [], color=color, ls='-',
                                             lw=1, label=rf'$\rm Z=10^{{ {log10Z:.0f} }}$'))

            Zh = ZH.deltaConstant({'log10Z': log10Z})  # constant metallicity

            ltom = {'stellar': [], 'total': []}

            for log10duration in log10durations:

                if sfh_model == 'Instantaneous':

                    sfzh = generate_instant_sfzh(
                        grid.log10ages, grid.metallicities, log10duration+6, 10**log10Z)

                else:

                    # --- define the parameters of the star formation and metal enrichment histories
                    sfh_p['duration'] = 10**log10duration * Myr

                    # --- define the functional form of the star formation and metal enrichment histories
                    sfh = sfh_f(sfh_p)

                    # --- get the 2D star formation and metal enrichment history for the given SPS grid. This is (age, Z).
                    sfzh = generate_sfzh(grid.log10ages, grid.metallicities, sfh, Zh)

                # --- define galaxy object
                galaxy = SEDGenerator(grid, sfzh)
                galaxy.pacman(fesc=0.0, fesc_LyA=1.0, tauV=0.0)

                for sed_name in ['stellar', 'total']:
                    sed = galaxy.spectra[sed_name]
                    luminosities = sed.get_broadband_luminosities(fc)
                    ltom[sed_name].append(luminosities['FUV'].value)
                    # print(imf_slope, luminosities['FUV'].value)

            ax.plot(log10durations, np.log10(ltom['total']), lw=1, color=color, ls=ls)

        handles.append(mlines.Line2D([], [], color='0.5', ls=ls, lw=1, label=label))

    ax.set_ylabel(r'$\rm log_{10}(duration/Myr)$')
    ax.set_ylabel(r'$\rm log_{10}[(L_{FUV}/M_{\star})/erg\ s^{-1}\ M_{\odot}^{-1}]$')
    ax.legend(handles=handles, fontsize=6, labelspacing=0.1)

    fig.savefig('figs/theory_sfh.pdf')
