

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
from synthesizer.grid import SpectralGrid, parse_grid_id
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh, generate_instant_sfzh
from synthesizer.parametric.galaxy import SEDGenerator
from synthesizer.plt import single, single_histxy, mlabel
from unyt import yr, Myr


if __name__ == '__main__':

    fs = [('FUV', {'lam_min': 1400, 'lam_max': 1600})]

    # -------------------------------------------------
    # --- calcualte the EW for a given line as a function of age

    fig, ax = single()

    ax.axvline(2.35, c='k', alpha=0.1, lw=3)
    ax.axvline(2.3, c='k', alpha=0.1, lw=1)

    handles = []

    sfh = SFH.Constant({'duration': 10 * Myr})
    Zh = ZH.deltaConstant({'log10Z': -3})  # constant metallicity

    for sps_model in ['fsps', 'bpass-100', 'bpass-300']:

        if sps_model == 'bpass-100':
            suffix = r'\ m_{up}=100\ M_{\odot}'
            marker = 'o'
            ls = '-'
            a3s = np.array([2.0, 2.35, 2.7])
            models = [
                f'bpass-v2.2.1-bin_{slope}-100_cloudy-v17.03_log10Uref-2' for slope in ['100', '135', '170']]

        if sps_model == 'bpass-300':
            suffix = r'\ m_{up}=300\ M_{\odot}'
            marker = 'h'
            ls = '--'
            a3s = np.array([2.0, 2.35, 2.7])
            models = [
                f'bpass-v2.2.1-bin_{slope}-300_cloudy-v17.03_log10Uref-2' for slope in ['100', '135', '170']]

        if sps_model == 'fsps':
            suffix = ''
            marker = 'D'
            ls = ':'
            a3s = np.arange(1.5, 3.1, 0.1)
            # a3s = [1.5]
            models = [
                f'fsps-v3.2_imf3:{a3:.1f}_cloudy-v17.03_log10Uref-2' for a3 in a3s]

        model_info = parse_grid_id(models[0])

        ltom = {'stellar': [], 'total': []}

        handles.append(mlines.Line2D([], [], color='0.5', ls=ls, marker=marker, markersize=4,
                                     lw=1, label=rf'$\rm {model_info["sps_model"]}\ {model_info["sps_model_version"]} {suffix}$'))

        for model in models:

            grid = SpectralGrid(model)
            fc = TopHatFilterCollection(fs, new_lam=grid.lam)

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

        ax.plot(a3s, np.log10(ltom['total']), lw=1, color='k',
                ls=ls, zorder=1, marker=marker, markersize=3)

    ax.set_xlabel(r'$\rm \alpha_{3}$')
    ax.set_ylabel(r'$\rm log_{10}[(L_{FUV}/M_{\star})/erg\ s^{-1}\ M_{\odot}^{-1}]$')
    ax.legend(handles=handles, fontsize=6, labelspacing=0.1)

    fig.savefig('figs/theory_imfsps.pdf')
