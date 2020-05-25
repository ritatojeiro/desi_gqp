import sys 
import os 
import h5py 
import numpy as np 
from astropy.io import fits
from scipy.io.idl import readsav

# --- plotting --- 
import matplotlib as mpl
import matplotlib.pyplot as plt
# --- gqp_mc ---
from gqp_mc import util as UT 
from gqp_mc import data as Data 
#from gqp_mc import fitters as Fitters

sys.path.append("/global/homes/r/rtojeiro/desi_gqp/code/")
import desi_gqp_utils as my_utils

import fsps

import ppxf as ppxf_package
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import glob

#simple bulge and disk templates
def read_simple_templates(velscale, lamrange):

    hdul = fits.open(UT.lgal_dir()+"/simple_mocks/template_fluxbc03.fits")
    wave_s = hdul[1].data['wave']
    flux_bulge = hdul[1].data['L_bulge']
    flux_disk = hdul[1].data['L_disk']
    hdul.close()
    
    wave,flux_bulge = to_common_grid(wave_s, flux_bulge,lamrange[0], lamrange[1])
    wave,flux_disk = to_common_grid(wave_s, flux_disk, lamrange[0], lamrange[1])

    mask = ((wave >= lamrange[0]) & (wave <= lamrange[1]))
    wave = wave[mask]
    
    flux_bulge = flux_bulge[mask]
    model1, logLam1, velscale_out = util.log_rebin([wave[0], wave[-1]], flux_bulge, velscale=velscale)
    model1 /= np.median(model1)
    print(velscale, velscale_out)
    flux_disk = flux_disk[mask]
    model2, logLam2, velscale_out = util.log_rebin([wave[0], wave[-1]], flux_disk, velscale=velscale)
    model2 /= np.median(model2)
    
    templates = np.column_stack([model1, model2])
    #print([wave[0], wave[-1]])
    
    plt.plot(np.exp(1)**logLam1, model1)
    plt.plot(np.exp(1)**logLam2, model2)
           
    return (logLam1, templates)

#returns SSP templates for given age and metallicity using FSPS models (MILES library)
def make_ssp_templates_fsps(velscale, lamrange,ages, metallicities):
    
    #initialise FSPS models
    sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1,
                                sfh=0)  
    
    #total number of templates
    N = len(ages) * len(metallicities)
    
    #put in log10(Z/Z_solar)
    metallicities = np.log10(np.array(metallicities)/0.019)
    
    #get the wavelength vector and initialise the array to hold all the templates
    wave = set_common_grid()
    templates = np.zeros((len(wave), N))
    
    #now loop through the given metallicities and ages to create each template in turn
    c=0
    for Z in metallicities:
        sp.params['logzsol']=Z
        for age in ages:
            wave_fsps, flux_fsps = sp.get_spectrum(tage=age,peraa=True) #in L_solar/AA
            #now to common grid
            wave, flux = to_common_grid(wave_fsps, flux_fsps, lamrange[0], lamrange[1])
            templates[:,c] = flux/np.median(flux)
            c=c+1
        
    #now mask and re-grid
    mask = ((wave >= lamrange[0]) & (wave <= lamrange[1]))
    wave = wave[mask]  
    template_rb, logLam, velscale_out = util.log_rebin([wave[0], wave[-1]], templates[mask,0], velscale=velscale)
        
    templates_out = np.zeros((len(logLam), N))
    templates_out[:,0] = template_rb
    plt.plot(np.exp(1)**logLam, templates_out[:,0])
    for i in range(1,N):
        templates_out[:,i], logLam, velscale_out = util.log_rebin([wave[0], wave[-1]], templates[mask,i], velscale=velscale)
        templates_out[:,i] = templates_out[:,i] 
        plt.plot(np.exp(1)**logLam, templates_out[:,i])
    return logLam, templates_out
     
def read_models_bc03():
    raw_file = '../data/all_raw_data_BC03_StelibAtlas.sav'
    tab = readsav(raw_file)
    age_m = tab.get('all_raw_ages')
    flux_m = tab.get('all_raw_data')
    #models_raw = models_raw * 3.839e26 #from L/Lsun to W/AA
    wave = tab.get('wave')[:,0]
    Z_m = tab.get('z')
    return age_m, flux_m, wave, Z_m

#returns SSP templates for given age and metallicity using BC03 models (BaSEL library)
def make_ssp_templates_bc03(velscale, lamrange,ages, metallicities):
   
    N = len(ages) * len(metallicities) #total number of templates

    age_m, flux_m, wave_m, Z_m = my_utils.read_models_bc03()
    D = len(wave_m)

    #first interpolate in age, keep metallicity
    models_t = np.zeros( (len(Z_m),D, len(ages)), dtype=np.float32)
    for l in range(0,D):
        for z in range(0, len(Z_m)):
            #print(l,z)
            models_t[z,l,:] = np.interp(np.log10(ages*1e9), np.log10(age_m*1e9), flux_m[z,l,:])

    #then interpolate in metallicity
    #flux_out= np.zeros( (len(metallicities), D, len(ages)), dtype=np.float32)
    templates = np.zeros((D, N))
    c=0
    for j in range(0, len(metallicities)):
        for i in range(0, len(ages)):
            for l in range(0,D):
                templates[l,c] = np.interp(np.log10(metallicities[j]),np.log10(Z_m),models_t[:,l,i])
            c=c+1


    #now mask and re-grid
    mask = ((wave_m >= lamrange[0]) & (wave_m <= lamrange[1]))
    wave = wave_m[mask]  
    template_rb, logLam, velscale_out = util.log_rebin([wave[0], wave[-1]], templates[mask,0], velscale=velscale)

    templates_out = np.zeros((len(logLam), N))
    templates_out[:,0] = template_rb/np.median(template_rb)
    plt.plot(np.exp(1)**logLam, templates_out[:,0])
    for i in range(1,N):
        templates_out[:,i], logLam, velscale_out = util.log_rebin([wave[0], wave[-1]], templates[mask,i], velscale=velscale)
        templates_out[:,i] = templates_out[:,i] / np.median(templates_out[:,i])
        plt.plot(np.exp(1)**logLam, templates_out[:,i])

    return logLam, templates_out

def set_common_grid():
    #change as needed
    #return np.linspace(wave_i,wave_f, (wave_f-wave_i+1))
    age_m, flux_m, wave_m, Z_m = read_models_bc03()
    return wave_m
 
    
def to_common_grid(wave,flux, wave_i, wave_f):
    wave_grid = set_common_grid()
    flux_grid = np.interp(wave_grid, wave, flux)
    
    #print(flux_grid)
    return wave_grid[(wave_grid >=wave_i) & (wave_grid <= wave_f)], flux_grid[(wave_grid >=wave_i) & (wave_grid <= wave_f)]

def read_BSG_mocks(noise='none', sample='mini_mocha', lib='fsps'):

    meta, mock = Data.read_data(noise=noise, sample='mini_mocha')
    
    iobs = int(noise.strip('bgs'))    
    
    wave_b = mock['spec_wave_b_bgs'][:]
    wave_g = mock['spec_wave_r_bgs'][:]
    wave_r = mock['spec_wave_z_bgs'][:]

    flux_b = mock['spec_flux_b_bgs'][...][iobs,:,:][:]
    flux_g = mock['spec_flux_r_bgs'][...][iobs,:,:][:]
    flux_r = mock['spec_flux_z_bgs'][...][iobs,:,:][:]

    return wave_b, wave_g, wave_r, flux_b, flux_g, flux_r, meta