import tsfel
from tsfel.feature_extraction.features_utils import set_domain
import numpy as np
import pywt
import pycwt as cwt
def wavelet_power(signal,widths):
    """Description of your feature.

    Parameters
    ----------
    signal:
        The time series to calculate the feature of.
    widths:
        wavelet scales
    Returns
    -------
    tuple
        CWT global power

    """
    # Feature implementation
    freqs = 1/np.array(widths)
    mother = cwt.Morlet(6)
    wave, scales, freqs, _, _, _ = cwt.cwt(signal=signal, dt=1/12, wavelet=mother, freqs=freqs)
    power = np.abs(wave)**2
    glbl_power = power.mean(axis=1)
    return tuple(glbl_power)