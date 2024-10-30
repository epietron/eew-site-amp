from scipy.optimize import curve_fit
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

PHVSR_DATA = '/u01/epietron/phvsr_results/ppsd_len_600/window_10/smoothing_0_125/Any/final_data/phvsr_*'
PEAK_PLOT_LOCATION = '/u01/epietron/studies/EEW/hvsr/gauss_fit/peak_plots'
PHVSR_STATS_FOLDER = '/u01/epietron/phvsr_results/ppsd_len_600/window_10/smoothing_0_125/Any/other_data/hvsr_stats'

def db_to_amp(dB):
    return 10**(dB/20)


def Gaussian(x, a, xa, sa):
    return a*np.exp(-(x-xa)**2/(2*sa**2))


def peak_pick(file):
    hvsr_df = pd.read_csv(file, header=0, names=["freq", "lower", "med", "upper"])
    xdata = hvsr_df.freq.values
    yupper = db_to_amp(hvsr_df.upper.values)
    ylower = db_to_amp(hvsr_df.lower.values)
    ydata = db_to_amp(hvsr_df.med.values)

    #filter data
    amp_filter = ydata<40
    xdata = xdata[amp_filter]
    yupper = yupper[amp_filter]
    ylower = ylower[amp_filter]
    ydata = ydata[amp_filter]
    #cosine taper
    window = signal.windows.tukey(len(xdata), alpha=0.4)
    ydata_win = ydata*window
    
    # (1)
    # Pick the highest peak
    ymax_filter = ydata_win==ydata_win.max()
    ypeak = ydata_win[ymax_filter][0]
    yupperpeak = yupper[ymax_filter][0]
    ylowerpeak = ylower[ymax_filter][0]
    xpeak = xdata[ymax_filter][0]

    # (2)
    # Initial guess of the solution
    p = [ypeak, xpeak, 1.]
    # Find optimal solution
    try:
        coef, cov = curve_fit(Gaussian, xdata, ydata_win, p0=p)
        x_err = coef[2]
        # Show fit
        fit_y = Gaussian(xdata, coef[0], coef[1], coef[2])
    except RuntimeError as e:
        fit_y = 0
        x_err = 0

    # (3)
    # Remove fit value
    ydata1 = ydata_win - fit_y
    
    # Pick absolute peak
    ymax1_filter = ydata1==ydata1.max()
    ypeak1 = ydata1[ymax1_filter][0]
    yupperpeak1 = yupper[ymax1_filter][0]
    ylowerpeak1 = ylower[ymax1_filter][0]
    xpeak1 = xdata[ymax1_filter][0]

    # (4)
    # Initial guess of the solution
    p = [ypeak1, xpeak1, 1.]
    # Find optimal solution
    try:
        coef1, cov1 = curve_fit(Gaussian, xdata, ydata1, p0=p)
        x_err1 = coef1[2]
        # Show fit
        fit_y1 = Gaussian(xdata, coef1[0], coef1[1], coef1[2])
    except RuntimeError as e:
        fit_y1 = 0
        x_err1 = 0

    # (5)
    # Get peak y value from the non-tapered data
    ypeak0 = ydata[xdata==xpeak][0]
    ypeak1 = ydata[xdata==xpeak1][0]
    ymean = np.mean(ydata[(xdata>=0.1)&(xdata<=10)])
    ymin = ydata.min()
    ymax = ydata.max()
    # Peak Criteria
    freq_min = 0.10
    freq_max = 50
    amp_min = 2

    
    peak0 = (np.round(xpeak,2), np.round(ypeak,2))
    yerr0 = (np.round(ylowerpeak,2), np.round(yupperpeak,2))
    xerr0 = np.round(x_err,2)
    peak1 = (np.round(xpeak1,2), np.round(ypeak1,2))
    yerr1 = (np.round(ylowerpeak1,2), np.round(yupperpeak1,2))
    xerr1 = np.round(x_err1,2)

    real_peak0 = False
    real_peak1 = False
    highf_peak = None


    # Check if both peaks valid (frequency range <10Hz, amplitude > 2)
    if (peak0[0]<=freq_max) and (peak0[1]>=amp_min):
        # Asses amplitude of the peak     
        amp = peak0[1] - ymean
        if amp >= 2:
            real_peak0 = True

    if (peak1[0]<=freq_max) and (peak1[1]>=amp_min):
        # Asses amplitude of the peak
        amp = peak1[1] - ymean
        if amp >= 2:
            real_peak1 = True
    #both true; select lowest frequency peak
    if real_peak0 and real_peak1:
        peak_diff = np.abs(peak0[0] - peak1[0])
        # if peaks are close together, select first pick
        if peak_diff < 1:
            peak = (peak0, yerr0, xerr0)
        elif peak0[0] < peak1[0]:
            peak = (peak0, yerr0, xerr0)
            highf_peak = (peak1, yerr1, xerr1)
        else:
            peak = (peak1, yerr1, xerr1)
            highf_peak = (peak0, yerr0, xerr0)
    # both false; no valid peaks
    elif not(real_peak0) and not(real_peak1):
        peak = None
    # One true; select only valid peak
    else:
        peak = ((peak0, yerr0, xerr0) if real_peak0 else (peak1, yerr1, xerr1))

    fig, ax = plt.subplots()
    ax.semilogx(xdata, ydata)
    station, t1, t2 = file.split("/")[-1].split("_")[2:5]
    ax.set_title(f"HVSR Peak - {station}")
    
    if peak is not None:
        ax.vlines(peak[0][0], ymin-2, ymax+2, color='red',linestyle='--', linewidth=1)
        ax.annotate(peak[0], np.array(peak[0])+0.1)
        ax.set_ylim(-5, ymax+5)
    else:
        ax.set_ylim(-5, 10)
    filename = PEAK_PLOT_LOCATION + f'/{station}_hvsr_peak.png'
    plt.savefig(filename)
    plt.close()

    return peak, highf_peak

def get_phvsr_stats(phvsr_file):
    basename = os.path.basename(phvsr_file)
    file_split = basename.split('_')
    code = file_split[1]
    startDate = file_split[2]
    endDate = file_split[3]
    phvsr_stats_file = '_'.join(
    [PHVSR_STATS_FOLDER,
     code,
     startDate,
     endDate,
     '10m_0.125octave.csv'])
    
    return phvsr_stats_file


def main():
    files = glob.glob(PHVSR_DATA)
    # Get list of hvsr_stats files (actual phvsr data, from the final list)
    stats_files = [get_phvsr_stats(file) for file in files]

    df = pd.DataFrame()
    stations = []
    peak_freqs = []
    peak_amps = []
    amps_err_upper = []
    amps_err_lower = []
    highf_peak_freqs = []
    highf_peak_amps = []
    freq_errs = []

    for file in stats_files:
        #print(file)
        code = file.split("/")[-1].split("_")[2]

        print('\n',code)
        try:
            peak, highf_peak = peak_pick(file)
            #peak = (peak, yerr)
        except FileNotFoundError:
            print("The following file could not be found:",file)
            continue

        stations.append(code)
        if peak is not None:
            peak_freqs.append(peak[0][0])
            peak_amps.append(peak[0][1])
            amps_err_lower.append(peak[1][0])
            amps_err_upper.append(peak[1][1])
            freq_errs.append(peak[2])
        else:
            peak_freqs.append(-999)
            peak_amps.append(-999)
            amps_err_lower.append(-999)
            amps_err_upper.append(-999)
            freq_errs.append(-999)
        if highf_peak is not None:
            highf_peak_freqs.append(highf_peak[0][0])
            highf_peak_amps.append(highf_peak[0][1])
        else:
            highf_peak_freqs.append(-999)
            highf_peak_amps.append(-999)

    
    df['code'] = stations
    df['peak_f'] = peak_freqs
    df['peak_amp'] = peak_amps
    df['secondary_f'] = highf_peak_freqs
    df['amp_err_upper'] = amps_err_upper
    df['amp_err_lower'] = amps_err_lower
    df['freq_err'] = freq_errs
    df['secondary_amp'] = highf_peak_amps

    df.to_csv('peak_pick_summary.csv')


if __name__ == "__main__":
    main()