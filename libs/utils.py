import os
import scipy, math

import numpy   as np
import pandas  as pd
import pymc3   as pm

import theano.tensor as tt

from typing              import Tuple
from pymc3.backends.base import MultiTrace

from . import constants


# Ranges of params (T: Temperature, beta and M: Mass)
T_RANGE    = (5, 80)
BETA_RANGE = (0, 4)
M_RANGE    = (3, 12)

# Columns in datafram
FLUX_COLS = ['F60', 'F70', 'F100', 'F160', 'F250', 'F350', 'F500']
ERR_COLS  = ['sigma60', 'sigma70', 'sigma100', 'sigma160', 'sigma250', 'sigma350', 'sigma500']



class Obs():
    '''
    Class for calculating wavelengths (lambda), frequency and observed flux
    '''
    def __init__(self, df: pd.DataFrame or pd.Series):
        self.df      = df
        self.lambda_ = self.__get_lambda()
        self.freq_   = self.__get_freq()
        self.flux_   = self.__get_flux()



    def __get_lambda(self) -> np.array:
        '''
        | Get the wavelengths from FLUX_COLS if the values is not NULL
        | Extract the wavelength values from column names (e.g: 'F160' -> wavelength = 160 micron)
        |
        | --Params:
        |           ...
        |
        | --Return:
        |            Wavelengths in [m]
        '''
        if isinstance(self.df, pd.Series):
            cols = [int(col[1:]) for col in self.df[FLUX_COLS].dropna().index]
        else:
            cols = [int(col[1:]) for col in self.df[FLUX_COLS].dropna(how='any', axis=1).columns]
        return 1e-6 * np.array(cols)


    def __get_freq(self) -> np.array:
        '''
        | Get frequencies if the values is not NULL
        |
        | --Params:
        |           ...
        |
        | --Return:
        |            Frequencies in [Hz]
        '''
        return constants.c / self.lambda_


    def __get_flux(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''
        | Get frequencies if the values is not NULL
        |
        | --Params:
        |           ...
        |
        | --Return:
        |            Returns dataframe of the flux [mJy] and its error
        '''
        if isinstance(self.df, pd.Series):
            fluxes = self.df[FLUX_COLS].dropna()
            errors = self.df[ERR_COLS].dropna()
        else:
            fluxes = self.df[FLUX_COLS].dropna(how='any', axis=1)
            errors = self.df[ERR_COLS].dropna(how='any', axis=1)
        return fluxes, errors




def MBB_model(nu_obs, T, beta, M, d, z):
    '''
    | Spectral energy distribution of a galaxy as a modified black body in units of mJy
    |
    | Flux_nu = (M_dust / d^2) * kappa_nu0 * (nu/nu0)^beta * Planck function
    |
    | Note: M_dust in the form of 10^{M}*Msun
    |
    | --Params:
    |           nu_obs: 1-D array - the observed frequency of flux measurements
    |           T:      ndarray/array/scalar - dust temperature
    |           beta:   1-D array - the emissivity parameter
    |           M   :   1-D array - the mass of dust in the galaxy in the form of  Msun * 10^M
    |           d:      1-D array - the distance to the galaxy
    |           z:      1-D array - the redshift of the galaxy
    |
    | --Return:
    |            spectral energy distribution of a galaxy as a modified black body in units of mJy
    '''
    h      = constants.h
    c      = constants.c
    kB     = constants.kB
    kappa0 = constants.kappa0
    nu0    = constants.nu0
    Msun   = constants.Msun


    if isinstance(T, pm.model.TransformedRV):
        nu_obs = nu_obs.reshape(1,-1)
        z      = z.values.reshape(-1,1)
        nu     = (1. + z) * nu_obs
        d      = d.values.reshape(-1,1)
        
        beta = beta[:, None]
        M    = M[:, None]
        T    = T[:, None]

        return 1.e29 * (Msun*10**M / (d**2)) * kappa0 * (nu/nu0)**beta * (2*h * nu**3 / c**2) / (np.exp(h*nu / (kB*T)) - 1.)

    elif isinstance(T, np.ndarray):
        if not isinstance(z, np.ndarray):
            z = np.array([z])
        
        if not isinstance(d, np.ndarray):
            d = np.array([d])
        
        nu_obs = nu_obs.reshape(1,-1)
        z      = z.reshape(-1,1)
        nu     = (1. + z) * nu_obs
        d      = d.reshape(-1,1)
        
        beta = beta[:, None]
        M    = M[:, None]
        T    = T[:, None]

        return 1.e29 * (Msun*10**M / (d**2)) * kappa0 * (nu/nu0)**beta * (2*h * nu**3 / c**2) / (np.exp(h*nu / (kB*T)) - 1.)
    else:
        nu = (1. + z) * nu_obs
        return 1.e29 * (Msun*10**M / (d**2)) * kappa0 * (nu/nu0)**beta * (2*h * nu**3 / c**2) / (np.exp(h*nu / (kB*T)) - 1.)


def MCMC_model(flux, err, nu, d, z):
    '''
    | Create a pymc3 model from parameters (Temperature, beta and Mass (in fact, mass index))
    |
    | Note: Flux_nu = (M_dust / d^2) * kappa_nu0 * (nu/nu0)^beta * Planck function
    |
    | Note: M_dust in the form of 10^{M}*Msun
    |
    | --Params:
    |           flux : dataframe - the observed flux
    |           err  : dataframe - Error of the observed flux
    |           nu   : 1-D array - Frequency range
    |           d    : 1-D array - the distance to the galaxy
    |           z    : 1-D array - the redshift of the galaxy
    |
    | --Return:
    |            pymc3 model
    '''
    with pm.Model() as model:
        
        # Priors
        temp = pm.Uniform('temperature', *T_RANGE,    shape=flux.shape[0])
        beta = pm.Uniform('beta',        *BETA_RANGE, shape=flux.shape[0])
        mass = pm.Uniform('mass',        *M_RANGE,    shape=flux.shape[0])

        # Expected value of outcome
        flux_exp = MBB_model(nu, temp, beta, mass, d, z)

        # Likelihood (sampling distribution) of observations
        flux_obs = pm.Normal('flux_obs', mu=flux_exp, sd=err, observed=flux)
    
    # End - with

    return model


def MCMC_trace(flux, err, nu, d, z, n_sample, chains, **kwargs) -> MultiTrace:
    '''
    | Results of MCMC
    |
    | --Params:
    |           flux     : dataframe - the observed flux
    |           err      : dataframe - Error of the observed flux
    |           nu       : 1-D array - Frequency range
    |           d        : 1-D array - the distance to the galaxy
    |           z        : 1-D array - the redshift of the galaxy
    |           n_sample : number of samples to take
    |           chains   : number of concurrent chains
    |
    | --Return:
    |            Traces as the results of MCMC
    '''
    model = MCMC_model(flux, err, nu, d, z)
    with model:
        if chains == 1:
            kwargs['compute_convergence_checks'] = False
        trace = pm.sample(n_sample, chains=chains, init='adapt_diag', tune=2000, **kwargs)
        return trace


def _results(index: list, trace: MultiTrace, burn: int, thin: int) -> pd.DataFrame:
    '''
    | Read the results from traces, and write to a dataframe
    |
    | --Params:
    |           index: Index of the dataframe
    |           trace: Trace object, the MCMC results
    |           burn: number of initial samples to burn
    |           thin: multiple of samples to keep
    |
    | --Return:
    |            dataframe that contains the results of MCMC
    '''
    res = pd.DataFrame(index=index)
    for par in ['temperature', 'beta', 'mass']:
        values = trace.get_values(par, burn=burn, thin=thin)
        res[par+'_mean'] = values.mean(axis=0)
        res[par+'_std'] = values.std(axis=0)

    return res




def _chi2(data: pd.Series, results: pd.Series) -> float:
    '''
    | Calculate the chi-square p value of the fitted data compared with the measured fluxes
    |
    | --Params:
    |           data    : pd.Series - fluxes and errors
    |           results : pd.Series - (mean temperature, beta and log mass)
    |
    | --Return:
    |           chi-square p-value of the fitted data compared with the measured fluxes
    |           (confidence)
    '''
    obs       = Obs(data)
    nu        = obs.freq_
    flux, err = obs.flux_
    flux      = flux.values
    err       = err.values
    
    # 3 free params: Temperature, beta, Mass
    dof  = len(flux) - 3
    T    = results['temperature_mean']
    beta = results['beta_mean']
    M    = results['mass_mean']
    
    mod_flux = MBB_model(nu, T, beta, M, data['D'], data['z'])

    chi2 = np.sum((flux - mod_flux) ** 2 / err ** 2) / dof
    
    return 1. - scipy.stats.chi2.pdf(chi2, dof)



def _run(data_set: pd.DataFrame, n_sample: int, chains: int, burn: int, thin: int, chunk: int, **kwargs) -> pd.DataFrame:
    '''
    | Runs the MCMC sampling, compute the flux
    |
    | --Params:
    |           data_set: DataFrame
    |           n_sample : number of samples to take
    |           chains   : number of concurrent chains
    |           burn     : int - number of initial samples to burn
    |           thin     : int - multiple of samples to keep
    |           chunk    : int - number of samples/rows in a chunk
    |           **kwargs passed to pymc3.sample
    |
    | --Return:
    |            dataframe that contains the results of MCMC
    |            with columns: (temperature_mean, temperature_std, beta_mean, beta_std, mass_mean, mass_std, confidence[%])
    '''
    nu       = Obs(data_set).freq_
    n_chunks = math.ceil(len(data_set) / chunk)
    
    trace_res = []
    for n_chunk, i in enumerate(range(0, len(data_set), chunk)):
        print(f'\nRunning MCMC for {data_set["Origin"].iloc[0]}: chunk {n_chunk + 1} of {n_chunks}')
        batch     = data_set.iloc[i: i + chunk]
        flux, err = Obs(batch).flux_
        trace     = MCMC_trace(flux=flux, err=err, nu=nu, d=batch['D'], z=batch['z'], n_sample=n_sample, chains=chains, **kwargs)

        batch_res               = _results(batch.index, trace, burn, thin)
        batch_res['confidence'] = [_chi2(batch.iloc[i], batch_res.iloc[i]) for i in range(batch.shape[0])]
        trace_res.append(batch_res)
    # End - for
    
    ret      = pd.concat(trace_res)
    Nsuccess = len(ret[ret['confidence'] >= 0.95])
    
    print(str(Nsuccess) + ' with p >= 0.95')
    
    return ret