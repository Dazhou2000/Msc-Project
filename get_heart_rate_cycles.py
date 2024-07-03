import pycwt as cwt
import numpy as np
import pandas as pd
import scipy

def heart_rate_cycles(signal,fs):
    """ signal: calculate multiday cycle of the whole signal

    Parameters
    ---------
    signal: nd-array
        Input from which the multiday cycle is calculated
    fs: float
        sampling frequency(hz) of input signal
    Returns
    -------
    float
        the multiday cycle of input signal under 99% significance peak test
    """
    n = (len(signal)-1)/fs/3600/4
    widths = np.append(np.arange(2.4, 31.2, 1.2), np.arange(31.2, 48, 2.4))
    widths = np.append(widths, np.arange(2.2 * 24, 4 * 24 + 4.8, 4.8))
    widths = np.append(widths, np.arange(5 * 24, int(n), 12))  # scale ( the periods to be tested)
    freqs = (1 / widths)
    mother = cwt.Morlet(6)  # morlet wavelet
    alpha, _, _ = cwt.ar1(signal)
    dt = 1/fs/3600  # samping period(per hour)

    wave, scales, freqs, coi, _, _ = cwt.cwt(signal=signal, dt=dt, wavelet=mother, freqs=freqs)
    power = np.abs(wave) ** 2
    glbl_power = power.mean(axis=1)

    dof = signal.size - scales  # Correction for padding at edges

    period = (1 / freqs).round(1)

    var = signal.std() ** 2
    # 99% significance power
    glbl_signif, tmp = cwt.significance(var, dt, scales, 1, alpha, significance_level=0.99, dof=dof, wavelet=mother)

    # peak analysis
    # Find peaks that are significant
    xpeaks = [];powers = []
    ind_peaks = scipy.signal.find_peaks(var * glbl_power)[0]
    for i in ind_peaks:
        peak = [var * glbl_power > glbl_signif][0][i]
        if peak:
            if period[i] not in xpeaks:
                xpeaks.append(period[i])
                powers.append([var * glbl_power][0][i])

    # keep only stongest peak if there is a peak within +/- 33% of another peak
    xpeaks = np.array(xpeaks)
    new_xpeaks = {}
    for peak in xpeaks:
        ints2 = np.where(np.logical_and(xpeaks >= peak - 0.33 * peak, xpeaks <= peak + 0.33 * peak))
        # is the peak in another peaks BP filter?
        other = [i for i, p in enumerate(xpeaks) if peak >= p - 0.33 * p and peak <= p + 0.33 * p]
        ints2 = set(np.array(list(ints2[0]) + other))
        if len(ints2):
            # if there is a peak within +/- 33%, check the power of it, choose highest
            max_peak = xpeaks[[var * glbl_power][0].tolist().index(np.max([[var * glbl_power][0][i] for i in ints2]))]
            new_xpeaks[peak] = max_peak
    xpeaks = sorted(set(new_xpeaks.values()))
    return xpeaks
