#Importing modules
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import scipy.io as sio
import pandas as pd
from scipy.optimize import curve_fit
import scipy.signal as signal
import scipy.optimize as opt
import statistics as stats
import astropy.units as u
import numpy.linalg as linalg
import matplotlib.patches as patches
from astropy.timeseries import LombScargle
import math
from scipy.interpolate import UnivariateSpline

#Setting matplotlib colors to the default ones
plt.rcdefaults()


#LIST OF DEFINITIONS:
#- 
#- lomblombscargscarg_window
#- gauss
#- sineCurve
#- double_sineCurve
#- triple_sineCurve
#- quad_sineCurve
#- five_sineCurve
#- six_sineCurve
#- seven_sineCurve
#- nyquist
#- rms
#- gaia_v
#- gaia_V_uncrt
#- Ap_abs_mag
#- Ap_abs_mag_uncrt
#- bol_mag
#- bol_mag_uncrt
#- Ap_bc
#- Ap_bc_uncrt
#- luminosity
#- luminosity_uncrt
#- distance
#- distance_uncrt
#- i_angle
#- i_angle_uncrt
#- B_V_index
#- B_V_index_uncrt
#- obs_to_txt
#- median_int
#- mean_int

#LIST OF CLASSES:
#- Matlab
#- Spectrum
#- Gaussian
#- RV_Plots
#- Tess
#- Doppler_Imaging
#- Radius
#- Magnetic



#------------------------DEFINITIONS---------------------------#
################################################################
def lombscarg(t_values, y_values, minf=None, maxf=None, dpi=None, fname=None,
              xcoord_arrow=None, ycoord_arrow=None, xcoord_text=None, ycoord_text=None,
              text=None, plot=True, save=False):
    """Uses the astropy.py Lomb-Scargle Periodogram class to compute the 
    periodograms.
    
    INPUTS:
        t_values = array of time values
        y_values = array of y values like radial velocity or flux.
        minf = minimum frequency (c/d). Default = None.
        maxf = maximum frequency (c/d). Default = None.
        fname = string of figure name. Only used if save = True. Default = None.
        dpi = float controlling resolution of figure in dots-per-inch. Only used 
              if save = True. Default = None.
        xcoord_arrow = float denoting x coordinate of arrow head. Only used if 
                       save = True. Default = None.
        ycoord_arrow = float denoting y coordinate of arrow head. Only used if 
                       save = True. Default = None.
        xcoord_text = float denoting x coordinate of text for arrow. Only used if 
                       save = True. Default = None.
        ycoord_text = float denoting y coordinate of text for arrow. Only used if 
                       save = True. Default = None.
        text = string that provides text for the arrow. Only used if ave = True.
               Default = None.
        plot = bool response whereby if True, the plot of the periodogram will be
               outputted. Default = True. 
        save = bool response to whether the user wants to save the plot or not
               using arrow annotations. If True, then dpi, fname, xcoord_arrow, 
               ycoord_arrow, xcoord_text, ycoord_text and text variables need to
               be provided.
    RETURN IF PLOT = TRUE:
        None - a plot is outputted
    
    RETURN IF PLOT = FALSE:
        peak_freq = the value of the frequency with the stronget peak power.
    """
    
    #Calculating the lombscargle frequency and power values
    freq, power = LombScargle(t_values, y_values).autopower(minimum_frequency=minf,maximum_frequency=maxf)
    
    if plot == True:
        plt.figure()
        plt.plot(freq, power)
        plt.xlabel("Frequency (c d$^{-1}$)")
        plt.ylabel("Power")
        plt.grid()
        plt.minorticks_on()
        
        #Finding the frequency with the highest power peak
        peak_freq = freq[np.argmax(power)]
        
        return None
    
    if save == True:
        plt.figure(figsize=[10,5])
        plt.plot(freq, power)
        plt.xlabel("Frequency (c d$^{-1}$)")
        plt.ylabel("Power")
        plt.grid()
        plt.minorticks_on()
        
        #Annotating plot with arrows pointing to peak frequency
        plt.annotate(text, xy=(xcoord_arrow, ycoord_arrow), xycoords='data',
                     xytext=(xcoord_text, ycoord_text), textcoords='data',
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

        plt.savefig(fname, dpi=dpi)
        plt.close()
        
        return None
    
    if plot == False:
        #Finding the frequency with the highest power peak
        peak_freq = freq[np.argmax(power)]
        peak_power = max(power)
        
        print("Strongest frequency = {0} c/d with power {1}".format(peak_freq, peak_power))
        return peak_freq, peak_power
    
################################################################
def lombscarg_window(t_values, y_values, y_synth_values, minf=None, maxf=None, ymax=None, ymin=None,
                     dpi=600, fname=None, xcoord_arrow=None, ycoord_arrow=None, 
                     xcoord_text=None, ycoord_text=None, text=None, save=False,
                     plot=True):
    """Uses the astropy.py Lomb-Scargle Periodogram class to compute the 
    lombscarlge periodogram window.
    
    INPUTS:
        t_values = array of time values
        y_values = array of y values like radial velocity or flux.
        y_synth_values = synthetic model fit y values.
        minf = minimum frequency (c/d). Default = None.
        maxf = maximum frequency (c/d). Default = None.
        ymax = maximum y value. Default = None.
        fname = string of figure name. Only used if save = True. Default = None.
        dpi = float controlling resolution of figure in dots-per-inch. Only used 
              if save = True. Default = None.
        xcoord_arrow = float denoting x coordinate of arrow head. Only used if 
                       save = True. Default = None.
        ycoord_arrow = float denoting y coordinate of arrow head. Only used if 
                       save = True. Default = None.
        xcoord_text = float denoting x coordinate of text for arrow. Only used if 
                       save = True. Default = None.
        ycoord_text = float denoting y coordinate of text for arrow. Only used if 
                       save = True. Default = None.
        text = string that provides text for the arrow. Only used if ave = True.
               Default = None.
        plot = bool response whereby if True, the plot of the periodogram will be
               outputted. Default = True. 
        save = bool response to whether the user wants to save the plot or not
               using arrow annotations. If True, then dpi, fname, xcoord_arrow, 
               ycoord_arrow, xcoord_text, ycoord_text and text variables need to
               be provided.
    
    RETURNS:
        None - a plot is outputted
    """
    #Calculating the lombscargle frequency and power values
    freq, power = LombScargle(t_values, y_values).autopower(minimum_frequency=minf,
                                                            maximum_frequency=maxf)
    
    #Calculating the lombscargle synthetic frequency and power values
    freq_synth, power_synth = LombScargle(t_values, y_synth_values).autopower(minimum_frequency=minf,
                                                            maximum_frequency=maxf)
    if plot == True:
        plt.figure()
        plt.plot(freq, power, label="Observed")
        plt.plot(freq_synth, power_synth, label="Window")
        plt.xlabel("Frequency (c d$^{-1}$)")
        plt.ylabel("Power")
        plt.legend()
    
    
    if save == True:
        plt.figure(figsize=[10,5])
        plt.plot(freq, power, label="Observed")
        plt.plot(freq_synth, power_synth, label="Window")
        plt.xlabel("Frequency (c d$^{-1}$)")
        plt.ylabel("Power")
        plt.ylim(bottom=ymin, top=ymax)
        plt.grid()
        plt.minorticks_on()
        plt.legend()
        
        # #Annotating plot with arrows pointing to peak frequency
        # plt.annotate(text, xy=(xcoord_arrow, ycoord_arrow), xycoords='data',
        #              xytext=(xcoord_text, ycoord_text), textcoords='data',
        #              arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

        plt.savefig(fname, dpi=dpi)
        plt.close()
        
        return None

    return None

################################################################
def lombscarg_multiplot(t_values, y_values, y_synth_values, minf, 
                        maxf, ymax, ymin, dpi, fname, xcoord_arrow, ycoord_arrow, 
                        xcoord_text, ycoord_text, text):
    """Creates a subplot of all lomb scargle plots.
    
    INPUTS:
        t_values = array of time values
        y_values = array of y values like radial velocity or flux.
        y_synth_values = list of synthetic model fit y values.
        minf = minimum frequency (c/d) float. 
        maxf = maximum frequency (c/d) float.
        ymin = list of maximum y value for each subplot.
        ymax = list of maximum y value for each subplot.
        fname = string of figure name.
        dpi = float controlling resolution of figure in dots-per-inch. 
        xcoord_arrow = list of floats denoting x coordinate of arrow head. 
        ycoord_arrow = list of floats denoting y coordinate of arrow head.
        xcoord_text = list of floats denoting x coordinate of text for arrow.
        ycoord_text = list of floats denoting y coordinate of text for arrow..
        text = list of strings that provides text for the arrow. 
        
    RETURNS:
        None
    """
    
    freq, power = LombScargle(t_values, y_values).autopower(minimum_frequency=minf,maximum_frequency=maxf)

    plt.figure(figsize=[10,5])
    plt.subplots_adjust(hspace=0.05)
    
    for i in range(len(ymax)):
        ax = plt.subplot(len(ymax), 1, 1+i)
        
        if i == 0:
            ax.plot(freq, power)
            ax.annotate(text[0], xy=(xcoord_arrow[0], ycoord_arrow[0]), xycoords='data',
                        xytext=(xcoord_text[0], ycoord_text[0]), textcoords='data',
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
            ax.set_ylim(bottom=ymin[0], top=ymax[0])
            plt.grid()
            ax.minorticks_on()
            plt.ylabel("Power")
            
        else:
            freq_synth, power_synth = LombScargle(t_values, y_synth_values[i-1]).autopower(minimum_frequency=minf,
                                                                                         maximum_frequency=maxf)
            ax.plot(freq, power)
            ax.plot(freq_synth, power_synth)
            ax.annotate(text[i], xy=(xcoord_arrow[i], ycoord_arrow[i]), xycoords='data',
                        xytext=(xcoord_text[i], ycoord_text[i]), textcoords='data',
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
            ax.set_ylim(bottom=ymin[i], top=ymax[i])
            plt.grid()
            plt.minorticks_on()
            plt.ylabel("Power")
    
    plt.xlabel("Frequency (c d$^{-1}$)")
    plt.savefig(fname, dpi=dpi)
    plt.close()
  
################################################################
def gauss(x,a,mu,sigma):
    """Defining our Gaussian model.

    INPUTS:
        x = range of integers
        a = height of curve's peak
        mu = position of peak's center
        sigma = standard deviation

    RETURNS:
        model = a calculated Gaussian model
    """
    return a*np.exp(-0.5*(x-mu)**2/sigma**2)
        
################################################################   
def sineCurve(t,A,period,phase,offset):
    """Defining our Sine curve model.
    
    INPUTS:
        t = array of evenly-spaced time values from start date to end date.
        A = amplitude. 
        period = period of wave (days).
        phase = phase of wave.
        offset = offset of wave.
        
    RETURNS:
        sine = sine wave equation
    """
    return A*np.sin((2.0*np.pi*t/period) + 2.0*np.pi*phase) + offset
    
################################################################     
def double_sineCurve(t,A1, A2, period1, period2, phase1, phase2, offset1, offset2):
    """Defining our double sine curve model.
    
    INPUTS:
        t = array of evenly-spaced time values from start date to end date.
        A1 = amplitude of first sine wave. 
        A2 = amplitude of second wave. 
        period1 = period of first wave (days).
        period2 = period of second wave (days).
        phase1 = phase of first wave.
        phase2 = phase of second wave.
        offset1 = offset of first wave (mean of radial velocity data).
        offset2 = offset of second wave (mean of radial velocity data.
        
    RERTURNS:
        sine_double = the double sine wave
    """
    sine_1st = A1*np.sin((2.0*np.pi*t/period1) + 2.0*np.pi*phase1) + offset1
    sine_2nd = A2*np.sin((2.0*np.pi*t/period2) + 2.0*np.pi*phase2) + offset2
    
    sine_double = sine_1st + sine_2nd
    
    return sine_double

################################################################    
def triple_sineCurve(t,A1, A2, A3, period1, period2, period3, phase1, phase2, 
                     phase3, offset1, offset2, offset3):
    """Defining our triple sine curve model.
    
    INPUTS:
        t = array of evenly-spaced time values from start date to end date.
        A1 = amplitude of first sine wave. 
        A2 = amplitude of second wave.
        A3 = amplitude of third wave. 
        period1 = period of first wave (days).
        period2 = period of second wave (days).
        period3 = period of third wave (days).
        phase1 = phase of first wave.
        phase2 = phase of second wave.
        phase3 = phase of third wave.
        offset1 = offset of first wave (mean of radial velocity data).
        offset2 = offset of second wave (mean of radial velocity data).
        offset3 = offset of third wave (mean of radial velocity data).
        
    RERTURNS:
        sine_triple = the triple sine wave
    """
    sine_1st = A1*np.sin((2.0*np.pi*t/period1) + 2.0*np.pi*phase1) + offset1
    sine_2nd = A2*np.sin((2.0*np.pi*t/period2) + 2.0*np.pi*phase2) + offset2
    sine_3rd = A3*np.sin((2.0*np.pi*t/period3) + 2.0*np.pi*phase3) + offset3
    
    sine_triple = sine_1st + sine_2nd + sine_3rd
    
    return sine_triple

################################################################      
def quad_sineCurve(t,A1, A2, A3, A4, period1, period2, period3, period4,
                   phase1, phase2, phase3, phase4, offset1, offset2, offset3,
                   offset4):
    """Defining our quadtriple sine curve model.
    
    INPUTS:
        t = array of evenly-spaced time values from start date to end date.
        A1 = amplitude of first sine wave. 
        A2 = amplitude of second wave.
        A3 = amplitude of third wave. 
        A4 = amplitude of fourth wave. 
        period1 = period of first wave (days).
        period2 = period of second wave (days).
        period3 = period of third wave (days).
        period4 = period of fourth wave (days).
        phase1 = phase of first wave.
        phase2 = phase of second wave.
        phase3 = phase of third wave.
        phase4 = phase of fourth wave.
        offset1 = offset of first wave (mean of radial velocity data).
        offset2 = offset of second wave (mean of radial velocity data).
        offset3 = offset of third wave (mean of radial velocity data).
        offset4 = offset of fourth wave (mean of radial velocity data).
        
    RERTURNS:
        sine_quadtriple = the quadtriple sine wave
    """
    sine_1st = A1*np.sin((2.0*np.pi*t/period1) + 2.0*np.pi*phase1) + offset1
    sine_2nd = A2*np.sin((2.0*np.pi*t/period2) + 2.0*np.pi*phase2) + offset2
    sine_3rd = A3*np.sin((2.0*np.pi*t/period3) + 2.0*np.pi*phase3) + offset3
    sine_4th = A4*np.sin((2.0*np.pi*t/period4) + 2.0*np.pi*phase4) + offset4
    
    sine_quad = sine_1st + sine_2nd + sine_3rd + sine_4th
    
    return sine_quad

################################################################      
def five_sineCurve(t,A1, A2, A3, A4, A5, period1, period2, period3, period4,
                   period5, phase1, phase2, phase3, phase4, phase5, offset1, 
                   offset2, offset3, offset4, offset5):
    """Defining our five sine curve model.
    
    INPUTS:
        t = array of evenly-spaced time values from start date to end date.
        A1 = amplitude of first sine wave. 
        A2 = amplitude of second wave.
        A3 = amplitude of third wave. 
        A4 = amplitude of fourth wave. 
        A5 = amplitude of fifth wave. 
        period1 = period of first wave (days).
        period2 = period of second wave (days).
        period3 = period of third wave (days).
        period4 = period of fourth wave (days).
        period5 = period of fifth wave (days).
        phase1 = phase of first wave.
        phase2 = phase of second wave.
        phase3 = phase of third wave.
        phase4 = phase of fourth wave.
        phase5 = phase of fifth wave.
        offset1 = offset of first wave (mean of radial velocity data).
        offset2 = offset of second wave (mean of radial velocity data).
        offset3 = offset of third wave (mean of radial velocity data).
        offset4 = offset of fourth wave (mean of radial velocity data).
        offset5 = offset of fourth wave (mean of radial velocity data).
        
    RERTURNS:
        sine_five = the five sine wave
    """
    sine_1st = A1*np.sin((2.0*np.pi*t/period1) + 2.0*np.pi*phase1) + offset1
    sine_2nd = A2*np.sin((2.0*np.pi*t/period2) + 2.0*np.pi*phase2) + offset2
    sine_3rd = A3*np.sin((2.0*np.pi*t/period3) + 2.0*np.pi*phase3) + offset3
    sine_4th = A4*np.sin((2.0*np.pi*t/period4) + 2.0*np.pi*phase4) + offset4
    sine_5th = A5*np.sin((2.0*np.pi*t/period5) + 2.0*np.pi*phase5) + offset5
    
    sine_five = sine_1st + sine_2nd + sine_3rd + sine_4th + sine_5th
    
    return sine_five

################################################################      
def six_sineCurve(t,A1, A2, A3, A4, A5, A6, period1, period2, period3, period4,
                  period5, period6, phase1, phase2, phase3, phase4, phase5, phase6, 
                  offset1, offset2, offset3, offset4, offset5, offset6):
    """Defining our six sine curve model.
    
    INPUTS:
        t = array of evenly-spaced time values from start date to end date.
        A1 = amplitude of first sine wave. 
        A2 = amplitude of second wave.
        A3 = amplitude of third wave. 
        A4 = amplitude of fourth wave. 
        A5 = amplitude of fifth wave. 
        A6 = amplitude of sixth wave. 
        period1 = period of first wave (days).
        period2 = period of second wave (days).
        period3 = period of third wave (days).
        period4 = period of fourth wave (days).
        period5 = period of fifth wave (days).
        period6 = period of sixth wave (days).
        phase1 = phase of first wave.
        phase2 = phase of second wave.
        phase3 = phase of third wave.
        phase4 = phase of fourth wave.
        phase5 = phase of fifth wave.
        phase6 = phase of sixth wave.
        offset1 = offset of first wave (mean of radial velocity data).
        offset2 = offset of second wave (mean of radial velocity data).
        offset3 = offset of third wave (mean of radial velocity data).
        offset4 = offset of fourth wave (mean of radial velocity data).
        offset5 = offset of fifth wave (mean of radial velocity data).
        offset6 = offset of sixth wave (mean of radial velocity data).
        
    RERTURNS:
        sine_six = the six sine wave
    """
    sine_1st = A1*np.sin((2.0*np.pi*t/period1) + 2.0*np.pi*phase1) + offset1
    sine_2nd = A2*np.sin((2.0*np.pi*t/period2) + 2.0*np.pi*phase2) + offset2
    sine_3rd = A3*np.sin((2.0*np.pi*t/period3) + 2.0*np.pi*phase3) + offset3
    sine_4th = A4*np.sin((2.0*np.pi*t/period4) + 2.0*np.pi*phase4) + offset4
    sine_5th = A5*np.sin((2.0*np.pi*t/period5) + 2.0*np.pi*phase5) + offset5
    sine_6th = A6*np.sin((2.0*np.pi*t/period6) + 2.0*np.pi*phase6) + offset6
    
    sine_six = sine_1st + sine_2nd + sine_3rd + sine_4th + sine_5th + sine_6th
    
    return sine_six

################################################################      
def seven_sineCurve(t, A1, A2, A3, A4, A5, A6, A7, period1, period2, period3, 
                    period4, period5, period6, period7, phase1, phase2, phase3, 
                    phase4, phase5, phase6, phase7, offset1, offset2, offset3, 
                    offset4, offset5, offset6, offset7):
    """Defining our seven sine curve model.
    
    INPUTS:
        t = array of evenly-spaced time values from start date to end date.
        A1 = amplitude of first sine wave. 
        A2 = amplitude of second wave.
        A3 = amplitude of third wave. 
        A4 = amplitude of fourth wave. 
        A5 = amplitude of fifth wave. 
        A6 = amplitude of sixth wave. 
        A7 = amplitude of seventh wave. 
        period1 = period of first wave (days).
        period2 = period of second wave (days).
        period3 = period of third wave (days).
        period4 = period of fourth wave (days).
        period5 = period of fifth wave (days).
        period6 = period of sixth wave (days).
        period7 = period of seventh wave (days).
        phase1 = phase of first wave.
        phase2 = phase of second wave.
        phase3 = phase of third wave.
        phase4 = phase of fourth wave.
        phase5 = phase of fifth wave.
        phase6 = phase of sixth wave.
        phase7 = phase of seventh wave.
        offset1 = offset of first wave (mean of radial velocity data).
        offset2 = offset of second wave (mean of radial velocity data).
        offset3 = offset of third wave (mean of radial velocity data).
        offset4 = offset of fourth wave (mean of radial velocity data).
        offset5 = offset of fifth wave (mean of radial velocity data).
        offset6 = offset of sixth wave (mean of radial velocity data).
        offset7 = offset of sixth wave (mean of radial velocity data).
        
    RERTURNS:
        sine_seven = the seven sine wave
    """
    sine_1st = A1*np.sin((2.0*np.pi*t/period1) + 2.0*np.pi*phase1) + offset1
    sine_2nd = A2*np.sin((2.0*np.pi*t/period2) + 2.0*np.pi*phase2) + offset2
    sine_3rd = A3*np.sin((2.0*np.pi*t/period3) + 2.0*np.pi*phase3) + offset3
    sine_4th = A4*np.sin((2.0*np.pi*t/period4) + 2.0*np.pi*phase4) + offset4
    sine_5th = A5*np.sin((2.0*np.pi*t/period5) + 2.0*np.pi*phase5) + offset5
    sine_6th = A6*np.sin((2.0*np.pi*t/period6) + 2.0*np.pi*phase6) + offset6
    sine_7th = A7*np.sin((2.0*np.pi*t/period7) + 2.0*np.pi*phase7) + offset7
    
    sine_seven = sine_1st + sine_2nd + sine_3rd + sine_4th + sine_5th + sine_6th + sine_7th
    
    return sine_seven

################################################################      
def eight_sineCurve(t, A1, A2, A3, A4, A5, A6, A7, A8, period1, period2, period3, 
                    period4, period5, period6, period7, period8, phase1, phase2, 
                    phase3, phase4, phase5, phase6, phase7, phase8, offset1, 
                    offset2, offset3, offset4, offset5, offset6, offset7, offset8):
    """Defining our eight sine curve model.
    
    INPUTS:
        t = array of evenly-spaced time values from start date to end date.
        A1 = amplitude of first sine wave. 
        A2 = amplitude of second wave.
        A3 = amplitude of third wave. 
        A4 = amplitude of fourth wave. 
        A5 = amplitude of fifth wave. 
        A6 = amplitude of sixth wave. 
        A7 = amplitude of seventh wave.
        A8 = amplitude of eighth wave.
        period1 = period of first wave (days).
        period2 = period of second wave (days).
        period3 = period of third wave (days).
        period4 = period of fourth wave (days).
        period5 = period of fifth wave (days).
        period6 = period of sixth wave (days).
        period7 = period of seventh wave (days).
        period/8 = period of eighth wave (days).
        phase1 = phase of first wave.
        phase2 = phase of second wave.
        phase3 = phase of third wave.
        phase4 = phase of fourth wave.
        phase5 = phase of fifth wave.
        phase6 = phase of sixth wave.
        phase7 = phase of seventh wave.
        phase8 = phase of eighth wave.
        offset1 = offset of first wave (mean of radial velocity data).
        offset2 = offset of second wave (mean of radial velocity data).
        offset3 = offset of third wave (mean of radial velocity data).
        offset4 = offset of fourth wave (mean of radial velocity data).
        offset5 = offset of fifth wave (mean of radial velocity data).
        offset6 = offset of sixth wave (mean of radial velocity data).
        offset7 = offset of seventh wave (mean of radial velocity data).
        offset8 = offset of eighth wave (mean of radial velocity data).
        
    RERTURNS:
        sine_eight = the eight sine wave
    """
    sine_1st = A1*np.sin((2.0*np.pi*t/period1) + 2.0*np.pi*phase1) + offset1
    sine_2nd = A2*np.sin((2.0*np.pi*t/period2) + 2.0*np.pi*phase2) + offset2
    sine_3rd = A3*np.sin((2.0*np.pi*t/period3) + 2.0*np.pi*phase3) + offset3
    sine_4th = A4*np.sin((2.0*np.pi*t/period4) + 2.0*np.pi*phase4) + offset4
    sine_5th = A5*np.sin((2.0*np.pi*t/period5) + 2.0*np.pi*phase5) + offset5
    sine_6th = A6*np.sin((2.0*np.pi*t/period6) + 2.0*np.pi*phase6) + offset6
    sine_7th = A7*np.sin((2.0*np.pi*t/period7) + 2.0*np.pi*phase7) + offset7
    sine_8th = A7*np.sin((2.0*np.pi*t/period8) + 2.0*np.pi*phase8) + offset8
    
    sine_eight = sine_1st + sine_2nd + sine_3rd + sine_4th + sine_5th + sine_6th + sine_7th + sine_8th
    
    return sine_eight

################################################################      
def nine_sineCurve(t, A1, A2, A3, A4, A5, A6, A7, A8, A9, period1, period2, period3, 
                    period4, period5, period6, period7, period8, period9, phase1, phase2, 
                    phase3, phase4, phase5, phase6, phase7, phase8, phase9, offset1, 
                    offset2, offset3, offset4, offset5, offset6, offset7, offset8,
                    offset9):
    """Defining our eight sine curve model."""
    
    sine_1st = A1*np.sin((2.0*np.pi*t/period1) + 2.0*np.pi*phase1) + offset1
    sine_2nd = A2*np.sin((2.0*np.pi*t/period2) + 2.0*np.pi*phase2) + offset2
    sine_3rd = A3*np.sin((2.0*np.pi*t/period3) + 2.0*np.pi*phase3) + offset3
    sine_4th = A4*np.sin((2.0*np.pi*t/period4) + 2.0*np.pi*phase4) + offset4
    sine_5th = A5*np.sin((2.0*np.pi*t/period5) + 2.0*np.pi*phase5) + offset5
    sine_6th = A6*np.sin((2.0*np.pi*t/period6) + 2.0*np.pi*phase6) + offset6
    sine_7th = A7*np.sin((2.0*np.pi*t/period7) + 2.0*np.pi*phase7) + offset7
    sine_8th = A8*np.sin((2.0*np.pi*t/period8) + 2.0*np.pi*phase8) + offset8
    sine_9th = A9*np.sin((2.0*np.pi*t/period9) + 2.0*np.pi*phase9) + offset9
    
    sine_nine = sine_1st + sine_2nd + sine_3rd + sine_4th + sine_5th + sine_6th + sine_7th + sine_8th + sine_9th
    
    return sine_nine

################################################################   
def nyquist(time_array, data_type):
    """Determines the Nyquist frequency of a dataset.
    
    IMPUTS:
        time_array = the time array of the dataset
        data_type = string variable that indicates whether the data is 'hercules'
                    or 'tess'.
        
    RETURNS:
        nyquist_freq = the Nyquist frequency
    """
    
    #HERCULES dataset isn't evenly spaced, so Nyquist Frequency will be computed
    #by taking the inverse of twice the median value of all time differences
    #between two consecutive measurements of the entire dataset
    if data_type == 'hercules':
    
        delta_t = np.zeros(time_array.shape[0])

        for t in range(len(time_array)-1):

            #Calculating the difference between two measurements
            del_t = (time_array[t+1] - time_array[t]) 

            #Inserting into array
            delta_t[t] = del_t

        #Computing Nyquist Frequency
        nyquist_freq = 1/(2*np.median(delta_t))
        print("The Nyquist frequency for {0} is: {1} c/d".format(data_type, nyquist_freq))
    
    #TESS datasets are evenly spaced with a two minute cadence, so Nyquist
    #Frequency is found by taking the inverse of twice the two minute cadence
    if data_type == 'tess':
        del_t = 2 / 60 / 24 #Taking the two minute cadence and converting into days
        
        #Computing Nyquist Frequency
        nyquist_freq = 1/(2*del_t)
        print("The Nyquist frequency for {0} is: {1} c/d".format(data_type, nyquist_freq))
        
    if data_type != 'hercules' and data_type != 'tess':
        print("Please specify what type of data you're working with: 'herucles'"
              + " or 'tess' in the data_type function input variable.")
              
        
    return nyquist_freq

################################################################   
def rms(obs, synth, t_range):
    """Calculating the root mean square error between synthetic fitting values
    and observed values. Low rms values indicates model represents the data well,
    whilst high rms values indicates that the model represents the data poorly.
    
    INPUTS:
        obs = array of observed y-value quantities (e.g. radial velocity or flux
              data)
        synth = array of y-values generated from model fitting.
        t_range = array of times.
    
    RETURNS:
        rmse = root mean square error value.
    """
    
    #Calculating the difference squared between the observed and synthetic values
    N = len(t_range)
    diff_sqr = [(obs[i] - synth[i])**2 for i in range(N)]
    
    #Summing the diff_sqr list
    diff_sqr_sum = sum(diff_sqr)
    
    #Calculating the root mean square error
    rmse = np.sqrt(diff_sqr_sum/N)
    
    return rmse

################################################################
def gaia_V(BV, P, gmag, d_star):
    """Calculates the absolute visual magnitude of a star using apparent gaia
    mangitudes.
    
    INPUTS:
        BV = B-V colour index (mag)
        P = parallax (arcseconds)
        gmag = apparent gaia magnitude (gmag)
        d_star = distance to star (pc)
        
    RETURNS:
        abs_V = absolute visual magnitude (mag)
    """
    
    #Converting apparent gmag into apparent vmag
    vmag = 0.04749 + (0.0124*BV) + (0.2901-BV**2) - (0.02008*BV**3) + gmag
    
    #Calculating distance to star using parallax
    d = 1/P
    
    #Using distance modulus to determine absolute magnitude
    abs_V = vmag - (5 * np.log10(d_star)) + 5
    
    return abs_V

################################################################
def gaia_V_uncrt(BV, BV_uncrt, P, P_uncrt, gmag, gmag_uncrt, d_star, d_star_uncrt):
    """Calculates the absolute visual magnitude uncertainty from gaia magnitudes.
    
    INPUTS:
        BV_uncrt = B-V colour index uncertainty (mag)
        P_uncrt = parallax uncertainty (arcseconds)
        gmag_uncrt = apparent gaia magnitude uncertainty (gmag)
        d_star = distance to star (pc)
        d_star_uncrt = distance to star uncertainty (pc)
        
    RETURNS:
        abs_V_uncrt = absolute visual magnitude uncertainty (mag)
    """
    
    #Computing uncertainty in vmag
    vmag_uncrt = np.sqrt(gmag_uncrt**2 + (-0.0124-(0.5802*BV)+(0.06024*(BV)**2))**2
                         * (BV_uncrt)**2)
    
    #Computing uncertainty in absolute visual magnitude
    abs_V_uncrt = np.sqrt(vmag_uncrt**2 + (-5/(d_star*np.log10(10)))**2*d_star_uncrt**2)
    
    return abs_V_uncrt

################################################################
def Ap_abs_mag(UB, BV):
    """Calculates the absolute visual magnitude of an Ap star using a specific
    expression used to take into account the unique flux distributions present
    in Ap stars.
    
    INPUTS:
        UB = (U-B) colour index (mag)
        BV = (B-V) colour index (mag)
        
    RETURNS:
        abs_V_Ap = absolute visual magniude for an Ap star (mag)
    """
    
    #Defining fit parameter constant values
    a = 2.99
    b = 5.09
    c = 0.74
    
    #Defining Ap absolute magnitude expression
    abs_V_Ap = a*UB + b*BV + c
    
    return abs_V_Ap

################################################################
def Ap_abs_mag_uncrt(UB, UB_uncrt, BV, BV_uncrt):
    """Calculates the absolute visual magnitude uncertainty for the special Ap
    expression.
    
    INPUTS:
        UB = (U-B) colour index (mag)
        UB_uncrt = (U-B) colour index uncertainty (mag)
        BV = (B-V) colour index (mag)
        BV_uncrt = (B-V) colour index uncertainty (mag)
        
    RETURNS:
        abs_V_Ap_uncrt = absolute visual magniude uncertainty for an Ap star (mag)
    """
    
    #Defining fit parameter and uncertainty values
    a = 2.99
    a_uncrt = 0.97
    b = 5.09
    b_uncrt = 1.9
    c = 0.74
    c_uncrt = 0.22
    
    #Defining Ap absolute magnitude uncertainty expression
    abs_V_Ap_uncrt = np.sqrt(UB**2*a_uncrt**2 + a**2*UB_uncrt**2 + BV**2*b_uncrt**2
                             + b**2*BV_uncrt**2 + c_uncrt**2)
    
    return abs_V_Ap_uncrt

################################################################
def bol_mag(BC, abs_V):
    """Determine the bolometric absolute magnitude using a given bolometric 
    correction value.
    
    INPUTS:
        BC = bolometric correction (mag)
        abs_V = absolute visual magnitude (mag)
        
    RETURNS:
        abs_bol_V = absolute bolometric visual magnitude (mag)
    """
    
    abs_bol_V = BC + abs_V
    
    return abs_bol_V

################################################################
def bol_mag_uncrt(BC_uncrt, abs_V_uncrt):
    """Determine the bolometric absolute magnitude uncertainty.
    
    INPUTS:
        BC_uncrt = bolometric correction uncertainty (mag)
        abs_V_uncertainty = absolute visual magnitude uncertainty (mag)
        
    RETURNS:
        abs_bol_V_uncrt = absolute bolometric visual magnitude uncertianty (mag)
    """
    
    abs_bol_V_uncrt = np.sqrt(BC_uncrt**2 + abs_V_uncrt**2)
    
    return abs_bol_V_uncrt

################################################################
def Ap_bc(Teff):
    """Determine the bolometric correction using the special Ap expression.
    
    INPUTS:
        Teff = effective temperature (K)
        
    RETURNS:
        bc = bolometric correction for an Ap star (mag)
    """
    
    #Calculating the theta_e value
    theta_e = 5040/Teff
    
    #Determining bolometric correction value
    bc = -12.98*theta_e**2 + 16.77*theta_e - 5.58
    
    return bc

################################################################
def Ap_bc_uncrt(Teff, Teff_uncrt):
    """Determine the bolometric correction uncertainty using the special Ap 
    expression.
    
    INPUTS:
        Teff = effective temperature (K)
        Teff_uncrt = effective temperature uncertianty (K)
        
    RETURNS:
        bc_uncrt = bolometric correction uncertainty (mag)
    """
    
    #Calculating the theta_e value
    theta_e = 5040/Teff
    
    #Calculating the theta_e uncertainty value
    theta_e_uncrt = abs(-5040/Teff**2) * Teff_uncrt
    
    #Determining bolometric correction uncertainty value
    bc_uncrt = abs(-25.96*theta_e + 16.77) * theta_e_uncrt
    
    return bc_uncrt


################################################################
def luminosity(abs_bol_V):
    """Calculates the luminosity of the star using bolometric absolute magnitudes.

    INPUTS:
        abs_bol_V = absolute bolometric magnitude (mag)

    RETURNS:
        L = stellar luminosity (W)
    """
    #Defining solar values
    L_sol = 3.828e26 #Units W
    abs_bol_V_sol = 4.74 #Units mag

    #Calculating luminosity
    L = L_sol * 10**((abs_bol_V-abs_bol_V_sol)/(-2.5))

    return L

################################################################
def luminosity_uncrt(abs_bol_V, abs_bol_V_uncrt):
    """Calculates the uncertainty in the stellar luminosity.

    INPUTS:
        abs_bol_V = absolute bolometric magnitude (mag)
        abs_bol_V_uncrt = absolute bolometric magnitude uncertainty (mag)

    RETURNS:
        L_uncrt = stellar luminosity uncertainty (W)
    """
    #Defining solar values
    L_sol = 3.828e26 #Units W
    abs_bol_V_sol = 4.74 #Units mag
    abs_bol_V_sol_uncrt = 0.01 #Units mag

    #Calculating luminosity uncertainty
    L_uncrt = np.sqrt((-(5*L_sol)/(2) * 10**((abs_bol_V-abs_bol_V_sol)/(-2.5)))**2
                           * (abs_bol_V_uncrt)**2 + ((5*L_sol)/(2) * 10**((abs_bol_V-abs_bol_V_sol)/(-2.5)))**2
                           * (abs_bol_V_sol_uncrt)**2)

    return L_uncrt

################################################################
def distance(P):
    """Computes distance to star in pc using Gaia parallax measurements.

    INPUTS:
        p = Gaia parallax (arcseconds)

    RETURNS:
        d = distance to star (pc)
    """

    d = 1 /P

    return d

################################################################
def distance_uncrt(P, P_uncrt):
    """Computes distance uncertainty.

    INPUTS:
        p = Gaia parallax (arcsecond)
        P_uncrt = Gaia parallax uncertainty (arcsecond)

    RETURNS:
        d_uncrt = distance uncertainty (pc)
    """

    d_uncrt = P_uncrt / P**2 

    return d_uncrt

################################################################
def i_angle(vsini, P_rot, R_star):
    """Computes the inclination angle of the star using the 'oblique rotator'
    relationship. Uncertainty for this is calculated in the i_angle_uncrt 
    function.
    
    INPUTS:
        vsini = project rotational velocity (km/s)
        P_rot = rotational period (days)
        R_star = radius of the star (R_solar)
        
    RETURNS:
        i = inclination angle (degrees)
    """
    
    i = math.asin((vsini * P_rot)/(50.57818728*R_star)) * 180/np.pi
    
    return i

################################################################
def i_angle_uncrt(vsini, vsini_uncrt, P_rot, P_rot_uncrt, R_star, R_star_uncrt):
    """Computes the inclination angle uncertainty.
    
    INPUTS:
        vsini = project rotational velocity (km/s)
        vsini_uncrt = project rotational velocity uncertainty (km/s)
        P_rot = rotational period (days)
        P_rot_uncrt = rotational period uncertainty (days)
        R_star = radius of the star (R_solar)
        R_star_uncrt = radius uncertainty (R_solar)
        
    RETURNS:
        i_uncrt = inclination angle uncertainty (degrees)
    """
    
    x = 1 / (np.sqrt(1-((vsini*P_rot)/(50.57818728*R_star))**2))
    i_uncrt = (x/(50.57818728*R_star)) * np.sqrt(P_rot**2 * vsini_uncrt**2 + vsini**2 * 
                                                 P_rot_uncrt**2 + (-(vsini*P_rot)/R_star)**2 *
                                                 R_star_uncrt**2) * 180/np.pi
    
    return i_uncrt

################################################################
def bv_index(tycho_B, tycho_V):
    """Calculates the B-V colour index using Tycho B and Tycho V magnitudes.
    
    INPUTS:
        tycho_B = Tycho B apparent magnitude (Tycho mag)
        tycho_V = Tycho V apparent magnitude (Tycho mag)
        
    RETURNS:
        BV = B-V colour index (mag)
    """
    
    BV = 0.850*(tycho_B - tycho_V)
    
    return BV

################################################################
def bv_index_uncrt(tycho_B_uncrt, tycho_V_uncrt):
    """Calculates the B-V colour index uncertainty.
    
    INPUTS:
        tycho_B_uncrt = Tycho B apparent magnitude uncertainty (Tycho mag)
        tycho_V_uncrt = Tycho V apparent magnitude uncertainty (Tycho mag)
        
    RETURNS:
        BV_uncrt = B-V colour index uncertainty (mag)
    """
    
    BV_uncrt = np.sqrt(0.850**2 * tycho_B_uncrt**2 + (-0.850)**2 * tycho_V_uncrt)
    
    return BV_uncrt

################################################################  
def obs_to_txt(finaldata, star_name):
    """Takes in a full finaldata numpy array, and then separates each observation
    within it as .txt files.
    
    INPUTS:
        finaldata = numpy array object that is the Python equivalent of the .mat
                    version.
        star_name = name of star in string format all lower case eg hd_152564
        
    RETURNS:
        
    """
    #Extracting wavelengths and intensities from finaldata
    waves = finaldata[0]
    ints = finaldata[1]
    
    #Identifying number of observations in finaldata
    obs = ints.shape[0]
    
    #Identifying number of waves and ints
    wave_num = waves.shape[0]
    int_num = ints.shape[1]
    
    #Taking each observation, and saving it as an individual .txt file
    for i in range(0, obs):
        ith_wave_str = [str(waves[j][0]/10) for j in range(0, wave_num)]
        ith_int_str = [str(ints[i][k]) for k in range(0, int_num)]


        #Opening up the textfile, and writing waveobs, flux, err into it. This textfile
        #is tab delimited using \t.
        filename = star_name + "_observation_" + str(i)
        path = '/home/users/blo44/Documents/ASTR690/Data_Analysis/iSpec/iSpec_v20201001/' + star_name + '_iSpec'
        
        print("Saving observation number {}".format(str(i)))
        with open(os.path.join(path, filename + ".txt"), "w+") as output:
            output.write("waveobs\tflux\terr\n")
            for j, l in zip(ith_wave_str, ith_int_str):
                output.write("{0}\t{1}\t{2} \n".format(j,l,0))
        
        #Closing the textfile        
        output.close()

    return None

################################################################  
def median_int(ints):
    """Determines the median of a given intensity array.
    
    INPUTS:
        ints = full intenstiy array
        
    RETURNS:
        med_ints = median intensity
    """
    #Transposing the intensity array
    int_trans = (ints).T
    
    #Computing the median of the intensities
    med_ints = np.zeros(int_trans.shape[0])
    for i in range(len(int_trans)):
        #Taking the median of each column in int_trans, giving shape 
        # e.g. (178285, 1)
        ith_med = np.nanmedian(int_trans[i])
        med_ints[i] = ith_med
        
    return med_ints

################################################################  
def mean_int(ints):
    """Determines the mean of a given intensity array.
    
    INPUTS:
        ints = full intenstiy array
        
    RETURNS:
        mean_ints = mean intensity array
    """
    #Transposing the intensity array
    int_trans = (ints).T
    
    #Computing the mean of the intensities
    mean_ints = np.zeros(int_trans.shape[0])
    for i in range(len(int_trans)):
        #Taking the mean of each column in int_trans, giving shape 
        # e.g. (178285, 1)
        ith_mean = np.nanmean(int_trans[i])
        mean_ints[i] = ith_mean
        
    return mean_ints




#------------------------CLASSES-------------------------------#
################################################################
class Matlab:
    """Class that works with the initial MATLAB files"""
    def __init__(self, mat):
        """Initialises the Matlab class.
        
        INPUTS:
            mat = MATLAB file being transferred into Python format.
        """
        self.mat = mat
        
    def data(self):
        """Rewrites the MATLAB .mat files into a format that can be read by 
        Python.
            
        RETURNS:
            array_type = an array consisting of all fields in .mat file
        """     
        
        #Turning the .mat file into a dictionary
        file = sio.loadmat(self.mat)


        #Collecting wavelength, intensity, julian date data
        wavelength = file['finaldata']['fullwave'][0][0]
        intensity = file['finaldata']['fullint'][0][0]
        julian_date = file['finaldata']['jd'][0][0]
        systemic_velocity = file['finaldata']['systemic_velocity'][0][0]
        systemic_velocity_gauss = file['finaldata']['systemic_velocity_gauss'][0][0]
        radial_vels = file['finaldata']['radial_velocity'][0][0]
        radial_vels_error = file['finaldata']['radial_velocity_error'][0][0]
        vsini = file['finaldata']['vsini'][0][0]
        vsini_error = file['finaldata']['vsini_error'][0][0]
        single_vsini = file['finaldata']['single_vsini'][0][0]
        single_vsini_error = file['finaldata']['single_vsini_error'][0][0]
        single_visni_std = file['finaldata']['single_vsini_std'][0][0]
        signal_to_noise = file['finaldata']['signal_to_noise'][0][0] #For all other stars
        # signal_to_noise = 0 #For HD152564 only

        data_length = len(julian_date[0])
        date_list = []
        for i in range(data_length):
            day = (julian_date[0][i]) - julian_date[0][0]
            date_list.append(day)

        outdat = np.zeros((julian_date.shape[1],wavelength.shape[0]))
        
        return np.array([wavelength, intensity, julian_date, date_list, 
                        systemic_velocity, systemic_velocity_gauss, radial_vels,
                        radial_vels_error, vsini, vsini_error, single_vsini, 
                        single_vsini_error, single_visni_std, signal_to_noise], 
                        dtype=object)
    
    def synth_data(self, synth_mat):
        """Converts synthetic wavelengths and intensities stored into .mat files
        into a format that can be read by Python.
        
        INPUTS:
            synth_mat = the .mat file containing the synthetic wavelengths and
                        intensities.
        
        RETURNS:
            synth_array = an array containing the synthetic wavelengths and 
                          intensities.
        """
        
        data = sio.loadmat(synth_mat)
        
        #Extracting synthwave and synthint from the .mat file
        synthwave = data['synthwave']
        synthint = data['synthint']
        
        #Storing synthwave and synthint into an array
        synth_array = np.array([synthwave, synthint], dtype=object)
        
        return synth_array
    
################################################################
class Spectrum:
    """Class that outputs the spectrum of the data file in question."""
    
    def __init__(self, wave, int, date, synth_wave, synth_int):
        """Function that initalises Spectrum class.
        
        INPUTS:
            wave = wavelength
            int = intensity
            date = date
        """
        self.wave = wave
        self.int = int
        self.date = date
        self.synth_wave = synth_wave
        self.synth_int = synth_int
        
    def plot_all(self):
        """Plots all the spectras on top of each other.
        
        RETURNS:
            None - a plot is outputted from the given values.
        """
        length = len(self.int)
        for i in range(length):
            plt.figure(1)
            plt.plot(self.wave, self.int[i])
            plt.xlabel('Wavlength ($\AA{}$)')
            plt.ylabel('Intensity (W m$^{-2}$)')		
            plt.title('Absorption Spectra - Combined (for {})'.format(length))
            plt.minorticks_on()
        return None
        
    def plot_seg(self, tightness=1, num=None, specific=None, legend=None):
            """Segregates all the plots along the y axis.

            INPUTS:
                specific = an array containing specific indexes wanting to be plotted.
                           Default = None
                num = the number of plots wanting to be segregated. If None, then 
                    all spectra will be segregated. Default = None
                tightness = how close you want the spectrums to be from each other. 
                            Greater the number, the tighter the spectrums become. 
                            Default = 1.
                legend = a list of labels added onto legend will be outputted. 
                         Default = None.

            RETURNS:
                None - a plot is outputted from the given values.
            """
            
            #Calculating the median of the intensity
            med_ints = median_int(self.int)
            
            #COLORS ONLY FOR DANIEL'S WORK! 
            colors = ['grey', 'lightcoral', 'maroon', 'red', 'sandybrown', 'orange', 'olive',
                      'lawngreen', 'darkgreen', 'teal', 'deepskyblue', 'navy', 'indigo']
            if num == None and specific == None:
                if legend != None:
                    length = len(self.int)
                    for i in range(length):
                        # plt.figure(2)
                        plt.plot(self.wave, self.int[i] + (i/tightness), label = 'Phase ' + legend[i], 
                                 color=colors[i])
                        plt.plot(self.wave, med_ints + (i/tightness), linestyle = "dashed", 
                                 color="grey", alpha=0.7)
                        plt.xlabel('Wavlength ($\AA{}$)')
                        plt.ylabel('Intensity (W m$^{-2}$)')
                        plt.title('Absorption Spectra - Segregated (for {})'.format(length))
                        # plt.legend()
                        plt.grid()
                        plt.minorticks_on()
                    
                else:
                    length = len(self.int)
                    for i in range(length):
                        # plt.figure(2)
                        plt.plot(self.wave, self.int[i] + (i/tightness))
                        plt.plot(self.wave, med_ints + (i/tightness), linestyle = "dashed", 
                                 color="grey", alpha=0.7)
                        plt.xlabel('Wavlength ($\AA{}$)')
                        plt.ylabel('Intensity (W m$^{-2}$)')
                        plt.title('Absorption Spectra - Segregated (for {})'.format(length))
                        plt.grid()
                        plt.minorticks_on()

            if num != None and specific == None:
                for i in range(num):
                    # plt.figure(2)
                    plt.plot(self.wave, self.int[i] + i/tightness)
                    plt.xlabel('Wavlength ($\AA{}$)')
                    plt.ylabel('Intensity (W m$^{-2}$)')
                    plt.title('Absorption Spectra - Segregated (for {})'.format(num))
                    plt.grid()
                    plt.minorticks_on()

            if specific != None and num == None:
                for i, value in enumerate(specific):
                    specific_date = self.date[value]
                    # plt.figure(2)
                    plt.plot(self.wave, self.int[value] + i/tightness, 
                             label = "Day " + str(specific_date))
                    plt.xlabel('Wavlength ($\AA{}$)')
                    plt.ylabel('Intensity (W m$^{-2}$)')
                    plt.title('Absorption Spectra - Segregated at Specific Intervals')
                    plt.grid()
                    plt.legend()
                    plt.minorticks_on()

            if specific != None and num != None:
                print("Yeah... you can't do that. Either both have to be None, or only one can be.")
                

            return None
        
    def line_var(self, specific,  xmin, xmax, tightness=1, phase=None, cycle=None):
        """Same principle as the plot_seg function, but fine-tuned for line
        variatons within a given range of wavelength values.

        INPUTS:
            tightness = how close you want the spectrums to be from each other. 
                        Greater the number, the tighter the spectrums become. 
                        Default = 1.
            specific = an array containing specific indexes wanting to be plotted.
            xmin = minimum wavelength in angstroms.
            xmax = maximum wavelength in angstroms.
            phase = an array containing phases of the data. Default = None.

        RETURNS:
            None - a plot is outputted from the given values.
        """
        #Transposing wavelength and intensity arrays
        wave_trans = (self.wave).T[0]
        
        #Calculating the median of the intensity
        med_ints = median_int(self.int)
            
        for i, value in enumerate(specific):
            specific_date = self.date[value]

            p = np.where((wave_trans>=xmin) & (wave_trans<=xmax))[0]

            # plt.figure(2)
            plt.grid()
            plt.plot(self.wave[p], self.int[value][p] + i/tightness, 
                    color="black")
            plt.plot(self.wave[p], med_ints[p] + i/tightness, 
                   linestyle = "dashed", color="grey", alpha=0.7)
            plt.xlabel('Wavlength ($\AA{}$)')
            plt.ylabel('Intensity (W m$^{-2}$)')
            plt.title('Absorption Spectra - Segregated at Specific Intervals')
            plt.minorticks_on()
        
#         #Having phase = None will not add any text to plot
#         if phase.all() == None:
#             for i, value in enumerate(specific):
#                 specific_date = self.date[value]
#                 ith_phase = phase[i]

#                 p = np.where((wave_trans>=xmin) & (wave_trans<=xmax))[0]

#                 plt.figure(2)
#                 plt.grid()
#                 plt.plot(self.wave[p], self.int[value][p] + i/tightness, 
#                         coor="black")
#                 plt.plot(self.wave[p], med_ints[p] + i/tightness, 
#                        linestyle = "dashed", color="grey", alpha=0.7)
#                 plt.xlabel('Wavlength (Å)')
#                 plt.ylabel('Intensity (W m^-2)')
#                 plt.title('Absorption Spectra - Segregated at Specific Intervals')
        
#         #Having phase != None will annotate the phase value onto plot
#         if phase.all() != None:
#             for i, value in enumerate(specific):
#                 specific_date = self.date[value]
#                 ith_phase = phase[value]
#                 ith_cycle = cycle[value]

#                 p = np.where((wave_trans>=xmin) & (wave_trans<=xmax))[0]

#                 plt.figure(2, figsize=(6, 180), dpi=300)
#                 plt.grid()
#                 plt.plot(self.wave[p], self.int[value][p] + i/tightness, 
#                         color="black")
#                 plt.plot(self.wave[p], med_ints[p] + i/tightness, 
#                        linestyle = "dashed", color="grey", alpha=0.7)
#                 plt.annotate(text=str(round(ith_phase, ndigits=3)), xy=(xmax-1, 1.01+i/tightness), 
#                              xycoords='data', color="red")
#                 plt.annotate(text=str(round(ith_cycle, ndigits=3)), xy=(xmin, 1.01+i/tightness), 
#                              xycoords='data', color="blue")
#                 plt.xlabel('Wavlength (Å)')
#                 plt.ylabel('Intensity (W m^-2)')
#                 plt.title('Absorption Spectra - Segregated at Specific Intervals')
#                 #plt.savefig('HD152564_SiII_variation.png')
                
        return None
                
    def plot_median(self, synth_plot=False):
        """Plots the median spectrum of given wavelength and intesnity values.
        
        INPUTS:
            synth_plot = boolean that asks if you want to plot the synthetic data
                         alongside the observed. Default = False
        RETURNS:
            None - a plot is outputted from the given values
        """
           
        
        int_trans = (self.int).T
        
        #Finding the median of the intensities
        med_ints = np.zeros(int_trans.shape[0])
        for i in range(len(int_trans)):
            #Taking the median of each column in int_trans, giving shape 
            # e.g. (178285, 1)
            ith_med = np.nanmedian(int_trans[i])
            med_ints[i] = ith_med
        
        #Plotting
        if synth_plot == False:
            plt.figure()
            plt.plot(self.wave, med_ints)
            plt.xlabel('Wavlength ($\AA{}$)')
            plt.ylabel('Intensity (W m$^{-2}$)')
            plt.title('Absorption Spectra - Median')
            plt.minorticks_on()
        else:
            plt.figure()
            plt.plot(self.wave, med_ints, label="Observed")
            plt.plot(self.synth_wave, self.synth_int, color='red', label="Synthetic")
            plt.xlabel('Wavlength ($\AA{}$)')
            plt.ylabel('Intensity (W m$^{-2}$)')
            plt.title('Absorption Spectra - Median')
            plt.legend()
            plt.minorticks_on()
        return None

    def plot_mean(self, synth_plot=False):
        """Plots the mean spectrum of given wavelength and intesnity values.
        
        INPUTS:
            synth_plot = boolean that asks if you want to plot the synthetic data
                         alongside the observed. Default = False
        RETURNS:
            None - a plot is outputted from the given values
        """
           
        #Finding the mean of the intensities
        ints_mean = mean_int(self.int)
        
        #Plotting
        if synth_plot == False:
            plt.figure()
            plt.plot(self.wave, ints_mean)
            plt.xlabel('Wavlength ($\AA{}$)')
            plt.ylabel('Intensity (W m$^{-2}$)')
            plt.title('Absorption Spectra - Mean')
            plt.minorticks_on()
        else:
            plt.figure()
            plt.plot(self.wave, ints_mean, label="Observed")
            plt.plot(self.synth_wave, self.synth_int, color='red', label="Synthetic")
            plt.xlabel('Wavlength ($\AA{}$)')
            plt.ylabel('Intensity (W m$^{-2}$)')
            plt.title('Absorption Spectra - Mean')
            plt.legend()
            plt.minorticks_on()

        return None
    
    def plot_save(self, image_size, fname, dpi, plot_type, xmin, xmax, ymin, ymax, 
                  tightness=1, synth_plot=False, num=None, specific=None, legend=None):
        """Saves a particular plot type at a given filename, figure size, dpi,
        x-range and y-range. 
        
        INPUTS:
            image_size = list of two floats denoting the size of the saved figure.
                         Format is [width, height] in inches.
            fname = string of figure name.
            dpi = float controlling resolution of figure in dots-per-inch.
            plot_type = string stating what type of plot you want to save. Avaliable
                        options are: 'plot_all', 'plot_seg', 'plot_median' and 
                        'plot_mean'. Refer to source code for descriptions on 
                        what each of these do.
            xmin = float of the minimum x-value to be plotted.
            xmax = float of the maximum x-value to be plotted.
            ymin = float of the minimum y-value to be plotted.
            ymax = float of the maximum y-value to be plotted.
            tightness = for 'plot_seg' - how close you want the spectrums to be 
                        from each other. Greater the number, the tighter the 
                        spectrums become. Default = 1.
            synth_plot = for 'plot_median'. Boolean that asks if you want to plot 
                         the synthetic data alongside the observed. Default = False.
            specific = for 'plot_seg'. An array containing specific indexes 
                        wanting to be plotted. Default = None
            num = for 'plot_seg'. The number of plots wanting to be segregated. 
                  If None, then all spectra will be segregated. Default = None
            legend = for 'plot_seg'. A list of labels added onto legend will be
                     outputted. Default = None. 
                         
         RETURNS:
             None
         """
        
        #Making and saving the 'plot_all' plot type
        if plot_type == 'plot_all':
            length = len(self.int)
            for i in range(length):
                plt.figure(1,figsize=image_size)
                plt.plot(self.wave, self.int[i])
                plt.xlim(xmin, xmax)
                plt.ylim(ymin, ymax)
                plt.xlabel('Wavlength ($\AA{}$)')
                plt.ylabel('Intensity (W m$^{-2}$)')		
            plt.minorticks_on()
            plt.savefig(fname, dpi=dpi)
            plt.close()
           
        #Making and saving the 'plot_seg' plot type    
        elif plot_type == 'plot_seg':
            #Calculating the median of the intensity
            med_ints = median_int(self.int)
            
            #COLORS ONLY FOR DANIEL'S WORK! 
            colors = ['grey', 'lightcoral', 'maroon', 'red', 'sandybrown', 'orange', 'olive',
                      'lawngreen', 'darkgreen', 'teal', 'deepskyblue', 'navy', 'indigo']
            if num == None and specific == None:
                if legend != None:
                    length = len(self.int)
                    for i in range(length):
                        plt.figure(2,figsize=image_size)
                        plt.plot(self.wave, self.int[i] + (i/tightness), label = 'Phase ' + legend[i], 
                                 color=colors[i])
                        plt.plot(self.wave, med_ints + (i/tightness), linestyle = "dashed", 
                                 color="grey", alpha=0.7)
                        plt.xlabel('Wavlength ($\AA{}$)')
                        plt.ylabel('Intensity (W m$^{-2}$)')
                        plt.xlim(xmin, xmax)
                        plt.ylim(ymin, ymax)    
                        # plt.title('Absorption Spectra - Segregated (for {})'.format(length))
                        # plt.legend()
                        plt.grid()
                        plt.minorticks_on()
                    plt.savefig(fname, dpi=dpi)
                    plt.close()
                    
                else:
                    length = len(self.int)
                    for i in range(length):
                        plt.figure(2,figsize=image_size)
                        plt.plot(self.wave, self.int[i] + (i/tightness))
                        plt.plot(self.wave, med_ints + (i/tightness), linestyle = "dashed", 
                                 color="grey", alpha=0.7)
                        plt.xlabel('Wavlength ($\AA{}$)')
                        plt.ylabel('Intensity (W m$^{-2}$)')
                        plt.xlim(xmin, xmax)
                        plt.ylim(ymin, ymax)
                        # plt.title('Absorption Spectra - Segregated (for {})'.format(length))
                        plt.grid()
                        plt.minorticks_on()
                    plt.savefig(fname, dpi=dpi)
                    plt.close()

            if num != None and specific == None:
                for i in range(num):
                    plt.figure(2,figsize=image_size)
                    plt.plot(self.wave, self.int[i] + i/tightness)
                    plt.xlabel('Wavlength ($\AA{}$)')
                    plt.ylabel('Intensity (W m$^{-2}$)')
                    plt.xlim(xmin, xmax)
                    plt.ylim(ymin, ymax)
                    # plt.title('Absorption Spectra - Segregated (for {})'.format(num))
                    plt.grid()
                    plt.minorticks_on()
                plt.savefig(fname, dpi=dpi)
                plt.close()

            if specific != None and num == None:
                for i, value in enumerate(specific):
                    specific_date = self.date[value]
                    plt.figure(2,figsize=image_size)
                    plt.plot(self.wave, self.int[value] + i/tightness, 
                             label = "Day " + str(specific_date))
                    plt.xlabel('Wavlength ($\AA{}$)')
                    plt.ylabel('Intensity (W m$^{-2}$)')
                    plt.xlim(xmin, xmax)
                    plt.ylim(ymin, ymax)
                    # plt.title('Absorption Spectra - Segregated at Specific Intervals')
                    plt.grid()
                    plt.legend()
                    plt.minorticks_on()
                plt.savefig(fname, dpi=dpi)
                plt.close()

            if specific != None and num != None:
                print("Yeah... you can't do that. Either both have to be None, or only one can be.")
        
        #Making and saving the 'plot_median' plot type
        elif plot_type == 'plot_median':
            int_trans = (self.int).T
            
            #Finding the median of the intensities
            med_ints = np.zeros(int_trans.shape[0])
            for i in range(len(int_trans)):
                #Taking the median of each column in int_trans, giving shape 
                # e.g. (178285, 1)
                ith_med = np.nanmedian(int_trans[i])
                med_ints[i] = ith_med
            
            #Plotting
            if synth_plot == False:
                plt.figure(figsize=image_size)
                plt.plot(self.wave, med_ints)
                plt.xlabel('Wavlength ($\AA{}$)')
                plt.ylabel('Intensity (W m$^{-2}$)')
                plt.xlim(xmin, xmax)
                plt.ylim(ymin, ymax)
                plt.grid()
                plt.minorticks_on()
                plt.savefig(fname, dpi=dpi)
                plt.close()

            else:
                plt.figure(figsize=image_size)
                plt.plot(self.wave, med_ints, label="Observed")
                plt.plot(self.synth_wave, self.synth_int, color='red', label="Synthetic")
                plt.xlabel('Wavlength ($\AA{}$)')
                plt.ylabel('Intensity (W m$^{-2}$)')
                plt.xlim(xmin, xmax)
                plt.ylim(ymin, ymax)
                plt.grid()
                plt.minorticks_on()
                # plt.legend()
                plt.savefig(fname, dpi=dpi)
                plt.close()
################################################################
class Gaussian:
    """Class that calculates the Gaussian of an absorotion line, fits it, then 
    computes its centroid wavelength. Also computes the radial velocities of the
    line using these centroids."""
    
    def __init__(self, wave, int, jul):
        """Function that initalises Spectrum class."""
        self.wave = wave
        self.int = int
        self.jul = jul
            
    def fit_gaussian(self, xlim, xmax, absorb_line, guess_params, plot = False):
        """Fits gaussians to specified absorption lines to obtain centroid 
        wavelengths.
        
        INPUTS:
            xlim = minimum wavelength of absorption line
            xmax = maximum wavelength of absorption line
            absorb_line = string stating what absorption line is being fitted.
            guess_params = a list having [amplitude, rest wavelength (angstroms), 
                           standard deviation], which is used to provide guess
                           parameters for the Gaussian function and SciPy to 
                           work with to fit a Gaussian to the specified line.
            plot = a True/False variable where if True, the plots are outputted.
                   Default = False. 
            
        RETURNS:
            array_type = an array containing the centroid wavelength and 
                         corresponding uncertainties.
            """
            
        #Transposing the wavelength array 
        wave_trans = (self.wave).T[0]

        #Centroid and centroid uncertainties
        cent = np.zeros((self.int).shape[0])
        cent_uncrt = np.zeros((self.int).shape[0])
        
        data_length = len(self.int)
        for data in range(data_length):
            
            #Defining the positions of x and y 
            p = np.where((wave_trans>=xlim) & (wave_trans<=xmax))[0]
            #Fitting centroids using 'gauss' function, through Scipy module
            fit, cov = curve_fit(gauss,wave_trans[p],1-self.int[data][p], 
            guess_params)
            
            #Obtaining centroids
            cent[data] = fit[1]
            cent_uncrt[data] = np.sqrt(np.diag(cov))[1]
            
            #Plotting if boolean condition met
            if plot == True:
                plt.figure()
                plt.plot(wave_trans[p], 1-self.int[data][p], label="Observed Data")
                plt.plot(wave_trans[p],gauss(wave_trans[p],*fit), label="Gaussian Fit")
                plt.vlines(fit[1], min(1-self.int[data][p]), max(1-self.int[data][p]), 
                            color="black", linestyle=":", label="Centroid")
                plt.xlabel("Wavelength ($\AA{}$)")
                plt.ylabel("Intensity")
                plt.title("Gaussian Fit over {0} Spectral Line (Number {1})".format(absorb_line, data))
                plt.minorticks_on()
                plt.legend()
                plt.grid()
                plt.show()
                    
        return np.array([cent, cent_uncrt])
    
    
    def plotGauss_save(self, xlim, xmax, guess_params,  image_size, fname, dpi, obs_num):
        """Fits gaussians to specified absorption lines to obtain centroid 
        wavelengths, then saves the plot.
        
        INPUTS:
            xlim = minimum wavelength of absorption line 
            xmax = maximum wavelength of absorption line
            guess_params = a list having [amplitude, rest wavelength (angstroms), 
                           standard deviation], which is used to provide guess
                           parameters for the Gaussian function and SciPy to 
                           work with to fit a Gaussian to the specified line.
            image_size = list of two floats denoting the size of the saved figure.
                         Format is [width, height] in inches.
            fname = string of figure name.
            dpi = float controlling resolution of figure in dots-per-inch.
            obs_num = integer denoting preferred observation number to be plotted
                      and then saved.  
            
        RETURNS:
            None
            """
            
        #Transposing the wavelength array 
        wave_trans = (self.wave).T[0]
            
        #Defining the positions of x and y 
        p = np.where((wave_trans>=xlim) & (wave_trans<=xmax))[0]
        
        #Fitting centroids using 'gauss' function, through Scipy module
        fit, cov = curve_fit(gauss,wave_trans[p],1-self.int[obs_num][p], 
        guess_params, maxfev=50000)
        
        #Plotting 
        plt.figure(figsize=image_size)
        plt.plot(wave_trans[p], 1-self.int[obs_num][p], label="Observed Data")
        plt.plot(wave_trans[p],gauss(wave_trans[p],*fit), label="Gaussian Fit")
        plt.vlines(fit[1], min(1-self.int[obs_num][p]), max(1-self.int[obs_num][p]), 
                    color="black", linestyle=":", label="Centroid")
        plt.xlabel("Wavelength ($\AA{}$)")
        plt.ylabel("Intensity")
        plt.minorticks_on()
        # plt.title("Gaussian Fit over {0} Spectral Line (Number {1})".format(absorb_line, data))
        # plt.legend()
        plt.grid()
        plt.savefig(fname, dpi=dpi)
        plt.close()
                
        return None
    
    
    def radial_velocities(self, cent_array, rest_wave):
        """Obtain radial velocities and radial velocity uncertainties of specified
        absorption lines, using their rest wavelength from SpectroWeb.
        
        INPUTS:
            cent_array = the array object outputted from the fit_gaussian function,
                         containing the centroid wavelength and centroid uncertainty 
                         (in angstroms).
            rest_wave = the absorption line's rest wavelegnth (in angstroms).
        
        RETURNS:
            array_type = an array containing radial velocities and uncertainties
                         (in km/s). 
        """
        
        #Extracting centroid wavelength and centroid uncertainty from cent_array
        cent_wave = cent_array[0]
        cent_uncrt = cent_array[1]
            
        #Converting angstroms to meters
        rest_wave_m = rest_wave * (1*10**-10)
        cent_wave_m = cent_wave * (1*10**-10)
        cent_uncrt_m = cent_uncrt * (1*10**-10)


        #Using the Doppler Shift equation to obtain radial velocities from centroids.
        rad_vel = (((cent_wave_m - rest_wave_m)/rest_wave_m)*3.0e8)
        rad_vel /= 1000 #Converting from m/s to km/s


        #Calculating the uncertainties for the radial velocities
        rad_vel_uncrt = (3.0e8/rest_wave_m) * cent_uncrt_m 
        rad_vel_uncrt /= 1000 #Converting from m/s to km/s


        return np.array([rad_vel, rad_vel_uncrt])
    
    def rv_mean(self, rv_list, rv_uncrt_list):
        """Computes the mean of given radial velocity arrays.

        INPUTS:
            rv_list = list of radial velocities (Km/s), whereby each entry is an array
                      of radial velocities corresponding to a absorption line.
            rv_uncrt_list = list of radial velocity uncertainties (Km/s), whereby
                            each entry is an array of uncertainties for each 
                            absorption line. 

        RETURNS:
            rv_mean_array = an array of the mean radial velocities (first entry)
                            and mean radial radial velocity uncertainties (second
                            entry) from the given inputs.
                      
        """
        #Number of absorption lines
        list_len = len(rv_list)
        
        #Computing mean of radial velocities
        rv_mean = np.mean(rv_list, axis=0)
        
        #Computing mean of radial velocity uncertainties
        rv_uncrt_zip = zip(*rv_uncrt_list)
        
        #For 10 lines
        if list_len == 10:
            rv_mean_uncrt = [np.sqrt((i**2+j**2+k**2+y**2+u**2+p**2+e**2+x**2+v**2+l**2))/list_len for i,j,k,y,u,p,e,x,v,l in rv_uncrt_zip]
            
            #Creating array with rv_mean and rv_mean_uncrt lists    
            rv_mean_array = np.array([rv_mean, rv_mean_uncrt], dtype=object)

            return rv_mean_array
        
        #For 9 lines
        if list_len == 9:
            rv_mean_uncrt = [np.sqrt((i**2+j**2+k**2+y**2+u**2+p**2+e**2+x**2+v**2))/list_len for i,j,k,y,u,p,e,x,v in rv_uncrt_zip]
            
            #Creating array with rv_mean and rv_mean_uncrt lists    
            rv_mean_array = np.array([rv_mean, rv_mean_uncrt], dtype=object)
            
            return rv_mean_array
        
        #For 8 lines
        if list_len == 8:
            rv_mean_uncrt = [np.sqrt((i**2+j**2+k**2+y**2+u**2+p**2+e**2+x**2))/list_len for i,j,k,y,u,p,e,x in rv_uncrt_zip]
            
            #Creating array with rv_mean and rv_mean_uncrt lists    
            rv_mean_array = np.array([rv_mean, rv_mean_uncrt], dtype=object)
            
            return rv_mean_array
        
        #For 7 lines
        if list_len == 7:
            rv_mean_uncrt = [np.sqrt((i**2+j**2+k**2+y**2+u**2+p**2+e**2))/list_len for i,j,k,y,u,p,e in rv_uncrt_zip]
            
            #Creating array with rv_mean and rv_mean_uncrt lists    
            rv_mean_array = np.array([rv_mean, rv_mean_uncrt], dtype=object)

            return rv_mean_array
        #Prompting user that they need to adjust the code to consider their specific
        #number of absorption lines
        else:
            print("NOTE: This number of absorption lines wasn't taken into account," +
                  " so you'll need to go into the script code and manually write" +
                  " in a new IF statement for your number of absorption lines.")
            
                  
        
#     def average_rv(self, rv_zip, rv_uncrt_zip):
#         """Takes zipped-list of radial velocities and another zipped-list for 
#         corresponding uncertainties, and averages the two. For differing sized 
#         averages, raising ValueError enables the function to average out the 
#         absorption lines for varing numbers of lines wanting to be averaged.
        
#         NOTE: This function can be used to average up to 4 lines. If warning 
#         message has been printed out, then there's more lines than what this function
#         is expecting. Thus, you need to add another exception line at the top. 
#         E.g. if wanting to average 5 lines, you would add the following code:
#         ------------------------------------------------------------------------
#         try:
#                 #Putting the zip objects into list format
#                 rv_list = list(rv_zip)
#                 rv_uncrt_list = list(rv_uncrt_zip)
                
#                 #Computing the average radial velocities for each observation
#                 rv_average = [stats.mean(i) for i in radial_velocites_zip]
                
#                 #Computing the average radial velocitiy uncertainty for each observation
#                 uncertainty_average = []    
#                 for a,b,c,d,e in radial_velocity_uncertainties_zip:
#                     uncertainties = np.sqrt(a**2 + b**2 + c**2 + d**2 + e**2) / 5     
#                     uncertainty_average.append(uncertainties) 
                        
#                 return radial_velocity_average, uncertainty_average
#                 break
#         ------------------------------------------------------------------------
#         INPUTS:
#             rv_zip = a zipped array containing the radial velocities wanting to 
#             be averaged e.g. zip(H_alpha, H_beta)
#             rv_uncrt_zip = a zipped array of the uncertainties, e.g. 
#             zip(H_alpha_uncrt, H_beta_uncrt)
        
#         RETURNS:
#             array_type = an array containing radial velocites and uncertainties.        
#         """
        
#         #For 4 lines
#         while True:
#             try:
#                 #Putting the zip objects into list format
#                 rv_list = list(rv_zip)
#                 rv_uncrt_list = list(rv_uncrt_zip)
                
#                 #Computing the average radial velocities for each observation
#                 rv_average = [stats.mean(i) for i in rv_list]
                
#                 #Computing the average radial velocitiy uncertainty for each observation.
#                 #Print warning message if number of lines exceed 4.
#                 if len(rv_list[-1]) > 4:
#                     print("WARNING: There's more aborption lines than what this" +
#                     "average function can handle. Please read instructions laid" +
#                     "out in function code to fix this.")
#                 else:
#                     rv_uncrt_avg = np.zeros((self.jul).shape[1])    
#                     for a,b,c,d in rv_uncrt_list:
#                         uncrt_avg = np.sqrt(a**2 + b**2 + c**2 + d**2) / 4     
#                         for i in range(len(rv_list)):
#                             rv_uncrt_avg[i] = uncrt_avg
                        
#                 return np.array([rv_average, rv_uncrt_avg])
#                 break
            
#             except ValueError:
#                 #For 3 lines
#                 while True:
#                     try:
#                         rv_uncrt_avg = np.zeros((self.jul).shape[1])   
#                         for a,b,c in rv_uncrt_list:
#                             uncrt_avg = np.sqrt(a**2 + b**2 + c**2) / 3     
#                             for i in range(len(rv_list)):
#                                 rv_uncrt_avg[i] = uncrt_avg
                                
#                         return np.array([rv_average, rv_uncrt_avg])
#                         break
            
#                     except ValueError:
#                         #For 2 lines
#                         rv_uncrt_avg = np.zeros((self.jul).shape[1])     
#                         for a,b in rv_uncrt_list:
#                             uncrt_avg = np.sqrt(a**2 + b**2) / 2     
#                             for i in range(len(rv_list)):
#                                 rv_uncrt_avg[i] = uncrt_avg 
                                
#                         return np.array([rv_average, rv_uncrt_avg])
                    
################################################################      
class rv_plots:
    """Class that deals with creating radial velocity plots and performs model
    fitting if neccessary."""
    
    def __init__(self, rv_list, rv_uncrt_list, date):
        """Function that initialises the rv_plots class.
        
        INPUTS:
            rv_list = array of radial velocities (km/s) within list, whereby 
                      each entry in list is a specific absorption line  
            rv_uncrt_list = array of radial velocity uncertainties (km/s) within
                            list, whereby each entry in list is a specific 
                            absorption line.
            date = array of date values beginning at 0 days
        """
        self.rv_list = rv_list
        self.rv_uncrt_list = rv_uncrt_list
        self.date = date
        
    def plot(self, model_fit=None):
        """Produces a radial velocity scatter plot.
        
        INPUTS:
            model_fit = guesses for sine model fitting. If None, then a sine 
                        won't be fitted to the data (see sineCurve definitions
                        for required guesses). Default = None.
        RETURNS:
            None - a plot is outputted
        """
        
        plt.figure(figsize=[15,10])
        
        #Plotting without model fit
        if model_fit == None:
            list_len = len(self.rv_list)
            if list_len == 1:
                plt.scatter(self.date, self.rv_list[0], marker='.')
                plt.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], 
                             linestyle='', capsize=3.0)
                plt.xlabel("Date (days)")
                plt.ylabel("Radial Velocity (Km s$^{-1}$)")
                plt.title("Long-term Radial Velocity")
                plt.minorticks_on()
                plt.grid()
            if list_len > 1:
                plt.figure()

                #Using while loop to plot all rvs in rv_list
                i=0
                while i <= (list_len-1):
                    ith_rv = self.rv_list[i]
                    ith_rv_uncrt = self.rv_uncrt_list[i]

                    plt.scatter(self.date, ith_rv, marker='.', label=str(i+1))
                    plt.errorbar(self.date, ith_rv, ith_rv_uncrt, linestyle='',
                                 capsize=3.0)
                    i += 1

                plt.legend()
                plt.xlabel("Date (days)")
                plt.ylabel("Radial Velocity (Km s$^{-1}$)")
                plt.title("Long-term Radial Velocity")
                plt.minorticks_on()
                plt.grid()
                plt.show()
                
        #Plotting with model fit        
        if model_fit != None:
            #Fitting 1 sine wave
            if len(model_fit) == 4:
                t = np.linspace(start=self.date[0], stop=self.date[-1], num=50000)
                pFit, covFit = opt.curve_fit(sineCurve, np.array(self.date), list(self.rv_list[0]),
                                             model_fit, sigma=list(self.rv_uncrt_list[0]),
                                             absolute_sigma=True)

                plt.scatter(self.date, self.rv_list[0], marker='.')
                plt.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], 
                             linestyle='', capsize=3.0)
                plt.plot(t,sineCurve(t,*pFit))
                plt.xlabel("Date (days)")
                plt.ylabel("Radial Velocity (Km s$^{-1}$)")
                plt.title("Long-term Radial Velocity")
                plt.minorticks_on()
                plt.grid()
                
                #Calculating rms value
                synth_vals = sineCurve(np.array(self.date), *pFit)
                rms_val = rms(self.rv_list[0], synth_vals, self.date)
                print("RMSE value for this long-term radial velocity model fit is: {}".format(rms_val))
                
            #Fitting 2 sine waves
            if len(model_fit) == 8:
                t = np.linspace(start=self.date[0], stop=self.date[-1], num=50000)
                pFit, covFit = opt.curve_fit(double_sineCurve, np.array(self.date), 
                                             list(self.rv_list[0]), model_fit, 
                                             sigma=list(self.rv_uncrt_list[0]),
                                             absolute_sigma=True)

                plt.scatter(self.date, self.rv_list[0], marker='.')
                plt.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], 
                             linestyle='', capsize=3.0)
                plt.plot(t,double_sineCurve(t,*pFit))
                plt.xlabel("Date (days)")
                plt.ylabel("Radial Velocity (Km s$^{-1}$)")
                plt.title("Long-term Radial Velocity")
                plt.minorticks_on()
                plt.grid()
                
                #Calculating rms value
                synth_vals = double_sineCurve(np.array(self.date), *pFit)
                rms_val = rms(self.rv_list[0], synth_vals, self.date)
                print("RMSE value for this long-term radial velocity model fit is: {}".format(rms_val))
                
            #Fitting 3 sine waves
            if len(model_fit) == 12:
                t = np.linspace(start=self.date[0], stop=self.date[-1], num=50000)
                pFit, covFit = opt.curve_fit(triple_sineCurve, np.array(self.date), 
                                             list(self.rv_list[0]), model_fit, 
                                             sigma=list(self.rv_uncrt_list[0]),
                                             absolute_sigma=True)

                plt.scatter(self.date, self.rv_list[0], marker='.')
                plt.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], 
                             linestyle='', capsize=3.0)
                plt.plot(t,triple_sineCurve(t,*pFit))
                plt.xlabel("Date (days)")
                plt.ylabel("Radial Velocity (Km s$^{-1}$)")
                plt.title("Long-term Radial Velocity")
                plt.minorticks_on()
                plt.grid()
                
                #Calculating rms value
                synth_vals = triple_sineCurve(np.array(self.date), *pFit)
                rms_val = rms(self.rv_list[0], synth_vals, self.date)
                print("RMSE value for this long-term radial velocity model fit is: {}".format(rms_val))
                
            #Fitting 4 sine waves
            if len(model_fit) == 16:
                t = np.linspace(start=self.date[0], stop=self.date[-1], num=50000)
                pFit, covFit = opt.curve_fit(quad_sineCurve, np.array(self.date), 
                                             list(self.rv_list[0]), model_fit, 
                                             sigma=list(self.rv_uncrt_list[0]),
                                             absolute_sigma=True, maxfev=500000)

                plt.scatter(self.date, self.rv_list[0], marker='.')
                plt.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], 
                             linestyle='', capsize=3.0)
                plt.plot(t,quad_sineCurve(t,*pFit))
                plt.xlabel("Date (days)")
                plt.ylabel("Radial Velocity (Km s$^{-1}$)")
                plt.title("Long-term Radial Velocity")
                plt.minorticks_on()
                plt.grid()
                
                #Calculating rms value
                synth_vals = quad_sineCurve(np.array(self.date), *pFit)
                rms_val = rms(self.rv_list[0], synth_vals, self.date)
                print("RMSE value for this long-term radial velocity model fit is: {}".format(rms_val))
                
            #Fitting 5 sine waves
            if len(model_fit) == 20:
                t = np.linspace(start=self.date[0], stop=self.date[-1], num=50000)
                pFit, covFit = opt.curve_fit(five_sineCurve, np.array(self.date), 
                                             list(self.rv_list[0]), model_fit, 
                                             sigma=list(self.rv_uncrt_list[0]),
                                             absolute_sigma=True)

                plt.scatter(self.date, self.rv_list[0], marker='.')
                plt.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], 
                             linestyle='', capsize=3.0)
                plt.plot(t,five_sineCurve(t,*pFit))
                plt.xlabel("Date (days)")
                plt.ylabel("Radial Velocity (Km s$^{-1}$)")
                plt.title("Long-term Radial Velocity")
                plt.minorticks_on()
                plt.grid()
                
                #Calculating rms value
                synth_vals = five_sineCurve(np.array(self.date), *pFit)
                rms_val = rms(self.rv_list[0], synth_vals, self.date)
                print("RMSE value for this long-term radial velocity model fit is: {}".format(rms_val))
                
            #Fitting 6 sine waves
            if len(model_fit) == 24:
                t = np.linspace(start=self.date[0], stop=self.date[-1], num=50000)
                pFit, covFit = opt.curve_fit(six_sineCurve, np.array(self.date), 
                                             list(self.rv_list[0]), model_fit, 
                                             sigma=list(self.rv_uncrt_list[0]),
                                             absolute_sigma=True)

                plt.scatter(self.date, self.rv_list[0], marker='.')
                plt.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], 
                             linestyle='', capsize=3.0)
                plt.plot(t,six_sineCurve(t,*pFit))
                plt.xlabel("Date (days)")
                plt.ylabel("Radial Velocity (Km s$^{-1}$)")
                plt.title("Long-term Radial Velocity")
                plt.minorticks_on()
                plt.grid()
                
                #Calculating rms value
                synth_vals = six_sineCurve(np.array(self.date), *pFit)
                rms_val = rms(self.rv_list[0], synth_vals, self.date)
                print("RMSE value for this long-term radial velocity model fit is: {}".format(rms_val))
                
            #Fitting 7 sine waves
            if len(model_fit) == 28:
                t = np.linspace(start=self.date[0], stop=self.date[-1], num=50000)
                pFit, covFit = opt.curve_fit(seven_sineCurve, np.array(self.date), 
                                             list(self.rv_list[0]), model_fit, 
                                             sigma=list(self.rv_uncrt_list[0]),
                                             absolute_sigma=True, maxfev=50000)

                plt.scatter(self.date, self.rv_list[0], marker='.')
                plt.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], 
                             linestyle='', capsize=3.0)
                plt.plot(t,seven_sineCurve(t,*pFit))
                plt.xlabel("Date (days)")
                plt.ylabel("Radial Velocity (Km s$^{-1}$)")
                plt.title("Long-term Radial Velocity")
                plt.minorticks_on()
                plt.grid()
                
                #Calculating rms value
                synth_vals = seven_sineCurve(np.array(self.date), *pFit)
                rms_val = rms(self.rv_list[0], synth_vals, self.date)
                print("RMSE value for this long-term radial velocity model fit is: {}".format(rms_val))
            
            #Fitting 8 sine waves
            if len(model_fit) == 32:
                t = np.linspace(start=self.date[0], stop=self.date[-1], num=50000)
                pFit, covFit = opt.curve_fit(eight_sineCurve, np.array(self.date), 
                                             list(self.rv_list[0]), model_fit, 
                                             sigma=list(self.rv_uncrt_list[0]),
                                             absolute_sigma=True, maxfev=50000)

                plt.scatter(self.date, self.rv_list[0], marker='.')
                plt.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], 
                             linestyle='', capsize=3.0)
                plt.plot(t,eight_sineCurve(t,*pFit))
                plt.xlabel("Date (days)")
                plt.ylabel("Radial Velocity (Km s$^{-1}$)")
                plt.title("Long-term Radial Velocity")
                plt.minorticks_on()
                plt.grid()
                
                #Calculating rms value
                synth_vals = eight_sineCurve(np.array(self.date), *pFit)
                rms_val = rms(self.rv_list[0], synth_vals, self.date)
                print("RMSE value for this long-term radial velocity model fit is: {}".format(rms_val))
            
            return [pFit, covFit]
            
    def phase(self, period, model_fit=None):
        """Produces a radial velocity phase plot.

        INPUTS:
            period = rotational period of star (days)
            model_fit = guesses for sine model fitting. If None, then a sine 
                        won't be fitted to the data (see sineCurve definitions
                        for required guesses). Default = None.

        RETURNS IF MODEL_FIT == NONE:
            None - a plot is outputted
            
        RETURNS IF MODEL_FIT != NONE:
            model_params = an array containing pFit in first entry, and covFit 
                           in second entry from the model fitting process
        """
        #Converting dates into phases
        cycle = np.array(self.date)/period
        phase_array = cycle - np.floor(cycle)
        plt.figure()
        
        #Plotting without model fit
        if model_fit == None:
            #Plotting phase diagram
            plt.figure()
            plt.scatter(phase_array, self.rv_list[0], marker='.')
            plt.errorbar(phase_array, self.rv_list[0], self.rv_uncrt_list[0], linestyle='',
                         capsize=3.0)
            plt.xlabel("Phase")
            plt.ylabel("Radial Velocity (Km s$^{-1}$)")
            plt.title("Phase Diagram")
            plt.minorticks_on()
            plt.grid()
            
            
        #Plotting with model fit
        if model_fit != None:
            t = np.linspace(start=min(phase_array), stop=max(phase_array), 
                                num=50000)
            
            #itting 1 sine wave
            if len(model_fit) == 4:
                #Making model fit
                pFit, covFit = opt.curve_fit(sineCurve, phase_array, list(self.rv_list[0])
                                             ,model_fit, sigma=list(self.rv_uncrt_list[0]),
                                             absolute_sigma=True, maxfev=10000)
                
                #Plotting phase diagram with model fit
                plt.scatter(phase_array, self.rv_list[0], marker='.')
                plt.errorbar(phase_array, self.rv_list[0], self.rv_uncrt_list[0],
                             linestyle='', capsize=3.0)
                plt.plot(t,sineCurve(t,*pFit))
                plt.xlabel("Phase")
                plt.ylabel("Radial Velocity (Km s$^{-1}$)")
                plt.title("Phase Diagram")
                plt.grid()
                plt.minorticks_on()
                
                #Calculating rms value
                synth_vals = sineCurve(np.array(self.date), *pFit)
                rms_val = rms(self.rv_list[0], synth_vals, self.date)
                print("RMSE value for this phase radial velocity model fit is: {}".format(rms_val))
            
            #Fitting 2 sine waves
            if len(model_fit) == 8:
                #Making model fit
                pFit, covFit = opt.curve_fit(double_sineCurve, phase_array, 
                                             list(self.rv_list[0]),model_fit, 
                                             sigma=list(self.rv_uncrt_list[0]),
                                             absolute_sigma=True, maxfev=10000)
                
                #Plotting phase diagram with model fit
                plt.scatter(phase_array, self.rv_list[0], marker='.')
                plt.errorbar(phase_array, self.rv_list[0], self.rv_uncrt_list[0],
                             linestyle='', capsize=3.0)
                plt.plot(t,double_sineCurve(t,*pFit))
                plt.xlabel("Phase")
                plt.ylabel("Radial Velocity (Km s$^{-1}$)")
                plt.title("Phase Diagram")
                plt.minorticks_on()
                plt.grid()
                
                #Calculating rms value
                synth_vals = double_sineCurve(np.array(self.date), *pFit)
                rms_val = rms(self.rv_list[0], synth_vals, self.date)
                print("RMSE value for this phase radial velocity model fit is: {}".format(rms_val))
                
            #Fitting 3 sine waves
            if len(model_fit) == 12:
                #Making model fit
                pFit, covFit = opt.curve_fit(triple_sineCurve, phase_array, 
                                             list(self.rv_list[0]),model_fit, 
                                             sigma=list(self.rv_uncrt_list[0]),
                                             absolute_sigma=True, maxfev=50000)
                
                #Plotting phase diagram with model fit
                plt.scatter(phase_array, self.rv_list[0], marker='.')
                plt.errorbar(phase_array, self.rv_list[0], self.rv_uncrt_list[0],
                             linestyle='', capsize=3.0)
                plt.plot(t,triple_sineCurve(t,*pFit))
                plt.xlabel("Phase")
                plt.ylabel("Radial Velocity (Km s$^{-1}$)")
                plt.title("Phase Diagram")
                plt.minorticks_on()
                plt.grid()
                
                #Calculating rms value
                synth_vals = triple_sineCurve(np.array(self.date), *pFit)
                rms_val = rms(self.rv_list[0], synth_vals, self.date)
                print("RMSE value for this phase radial velocity model fit is: {}".format(rms_val))
                
            #Fitting 4 sine waves
            if len(model_fit) == 16:
                #Making model fit
                pFit, covFit = opt.curve_fit(quad_sineCurve, phase_array, 
                                             list(self.rv_list[0]),model_fit, 
                                             sigma=list(self.rv_uncrt_list[0]),
                                             absolute_sigma=True, maxfev=200000)
                
                #Plotting phase diagram with model fit
                plt.scatter(phase_array, self.rv_list[0], marker='.')
                plt.errorbar(phase_array, self.rv_list[0], self.rv_uncrt_list[0],
                             linestyle='', capsize=3.0)
                plt.plot(t,quad_sineCurve(t,*pFit))
                plt.xlabel("Phase")
                plt.ylabel("Radial Velocity (Km s$^{-1}$)")
                plt.title("Phase Diagram")
                plt.minorticks_on()
                plt.grid()
                
                #Calculating rms value
                synth_vals = quad_sineCurve(np.array(self.date), *pFit)
                rms_val = rms(self.rv_list[0], synth_vals, self.date)
                print("RMSE value for this phase radial velocity model fit is: {}".format(rms_val))
                
            #Fitting 5 sine waves
            if len(model_fit) == 20:
                #Making model fit
                pFit, covFit = opt.curve_fit(five_sineCurve, phase_array, 
                                             list(self.rv_list[0]),model_fit, 
                                             sigma=list(self.rv_uncrt_list[0]),
                                             absolute_sigma=True, maxfev=50000)
                
                #Plotting phase diagram with model fit
                plt.scatter(phase_array, self.rv_list[0], marker='.')
                plt.errorbar(phase_array, self.rv_list[0], self.rv_uncrt_list[0],
                             linestyle='', capsize=3.0)
                plt.plot(t,five_sineCurve(t,*pFit))
                plt.xlabel("Phase")
                plt.ylabel("Radial Velocity (Km s$^{-1}$)")
                plt.title("Phase Diagram")
                plt.minorticks_on()
                plt.grid()
                
                #Calculating rms value
                synth_vals = five_sineCurve(np.array(self.date), *pFit)
                rms_val = rms(self.rv_list[0], synth_vals, self.date)
                print("RMSE value for this phase radial velocity model fit is: {}".format(rms_val))
                
            #Fitting 6 sine waves
            if len(model_fit) == 24:
                #Making model fit
                pFit, covFit = opt.curve_fit(six_sineCurve, phase_array, 
                                             list(self.rv_list[0]),model_fit, 
                                             sigma=list(self.rv_uncrt_list[0]),
                                             absolute_sigma=True, maxfev=50000)
                
                #Plotting phase diagram with model fit
                plt.scatter(phase_array, self.rv_list[0], marker='.')
                plt.errorbar(phase_array, self.rv_list[0], self.rv_uncrt_list[0],
                             linestyle='', capsize=3.0)
                plt.plot(t,six_sineCurve(t,*pFit))
                plt.xlabel("Phase")
                plt.ylabel("Radial Velocity (Km s$^{-1}$)")
                plt.title("Phase Diagram")
                plt.minorticks_on()
                plt.grid()
                
                #Calculating rms value
                synth_vals = six_sineCurve(np.array(self.date), *pFit)
                rms_val = rms(self.rv_list[0], synth_vals, self.date)
                print("RMSE value for this phase radial velocity model fit is: {}".format(rms_val))
                
            #Fitting 7 sine waves
            if len(model_fit) == 28:
                #Making model fit
                pFit, covFit = opt.curve_fit(seven_sineCurve, phase_array, 
                                             list(self.rv_list[0]),model_fit, 
                                             sigma=list(self.rv_uncrt_list[0]),
                                             absolute_sigma=True, maxfev=5000000)
                
                #Plotting phase diagram with model fit
                plt.scatter(phase_array, self.rv_list[0], marker='.')
                plt.errorbar(phase_array, self.rv_list[0], self.rv_uncrt_list[0],
                             linestyle='', capsize=3.0)
                plt.plot(t,seven_sineCurve(t,*pFit))
                plt.xlabel("Phase")
                plt.ylabel("Radial Velocity (Km s$^{-1}$)")
                plt.title("Phase Diagram")
                plt.minorticks_on()
                plt.grid()
                
                #Calculating rms value
                synth_vals = seven_sineCurve(np.array(self.date), *pFit)
                rms_val = rms(self.rv_list[0], synth_vals, self.date)
                print("RMSE value for this phase radial velocity model fit is: {}".format(rms_val))
            
            return [pFit, covFit]
    
    def plot_save(self, model_fit, xmin1, xmax1, xmin2, xmax2, ymin1, ymax1,
                  ymin2, ymax2, fname, dpi=600):
        """Makes a full radial velocity plot against time, and then adds two 
        subplots that zoom into a certain section of the main plot.
        
        INPUTS:
            model_fit = guesses for sine model fitting (see sineCurve definitions
                        for required guesses).
            xmin1 = float value denoting minimum x value for first subplot
            xmax1 = float value denoting maximum x value for first subplot
            xmin2 = float value denoting minimum x value for second subplot
            xmax2 = float value denoting maximum x value for second subplot
            ymin1 = float value denoting minimum y value for first subplot
            ymax1 = float value denoting maximum y value for first subplot
            ymin2 = float value denoting minimum y value for second subplot
            ymax2 = float value denoting maximum y value for second subplot
            fname = string denoting name of file
            dpi =  
            
        RETURNS:
            None
        """
        
        #Creating subplot layout
        plt.figure(figsize=(15, 10))
        ax1 = plt.subplot(2,1,2)
        ax2 = plt.subplot(2,2,1)
        ax3 = plt.subplot(2,2,2)
        axes = [ax1, ax2, ax3]
        
        #Fitting 1 sine wave
        if len(model_fit) == 4:
            t = np.linspace(start=self.date[0], stop=self.date[-1], num=50000)
            pFit, covFit = opt.curve_fit(sineCurve, np.array(self.date), list(self.rv_list[0]),
                                         model_fit, sigma=list(self.rv_uncrt_list[0]),
                                         absolute_sigma=True)
        
            #Defining first subplot
            ax1.scatter(self.date, self.rv_list[0], marker='.')
            ax1.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], capsize=3.0, fmt='none', color="#1f77b4")
            ax1.plot(t,sineCurve(t,*pFit), c="#1f77b4", alpha=0.7)
            ax1.fill_between((xmin1,xmax1), min(self.rv_list[0]), max(self.rv_list[0]), facecolor='green', alpha=0.2)
            ax1.fill_between((xmin2,xmax2), min(self.rv_list[0]), max(self.rv_list[0]), facecolor='orange', alpha=0.2)
            ax1.set_xlabel("Time (Days)")
            ax1.set_ylabel("Radial Velocity (Km $s^{-1}$)")
            ax1.grid()
            ax1.minorticks_on()
        
            #Defining second subplot
            ax2.scatter(self.date, self.rv_list[0], marker='.')
            ax2.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], capsize=3.0, fmt='none', color="#1f77b4")
            ax2.plot(t,sineCurve(t,*pFit), c="#1f77b4", alpha=0.7)
            ax2.set_xlim(xmin1,xmax1)
            ax2.set_ylim(ymin1,ymax1)
            ax2.set_ylabel("Radial Velocity (Km $s^{-1}$)")
            ax2.grid()
            ax2.minorticks_on()
        
            #Defining third subplot
            ax3.scatter(self.date, self.rv_list[0], marker='.')
            ax3.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], capsize=3.0, fmt='none', color="#1f77b4")
            ax3.plot(t,sineCurve(t,*pFit), c="#1f77b4", alpha=0.7)
            ax3.set_xlim(xmin2,xmax2)
            ax3.set_ylim(ymin2,ymax2)
            ax3.grid()
            ax3.minorticks_on()

            
        #Fitting 2 sine waves
        if len(model_fit) == 8:
            t = np.linspace(start=self.date[0], stop=self.date[-1], num=50000)
            pFit, covFit = opt.curve_fit(double_sineCurve, np.array(self.date), list(self.rv_list[0]),
                                         model_fit, sigma=list(self.rv_uncrt_list[0]),
                                         absolute_sigma=True)
        
            #Defining first subplot
            ax1.scatter(self.date, self.rv_list[0], marker='.')
            ax1.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], capsize=3.0, fmt='none', color="#1f77b4")
            ax1.plot(t,double_sineCurve(t,*pFit), c="#1f77b4", alpha=0.7)
            ax1.fill_between((xmin1,xmax1), min(self.rv_list[0]), max(self.rv_list[0]), facecolor='green', alpha=0.2)
            ax1.fill_between((xmin2,xmax2), min(self.rv_list[0]), max(self.rv_list[0]), facecolor='orange', alpha=0.2)
            ax1.set_xlabel("Time (Days)")
            ax1.set_ylabel("Radial Velocity (Km $s^{-1}$)")
            ax1.grid()
            ax1.minorticks_on()
        
            #Defining second subplot
            ax2.scatter(self.date, self.rv_list[0], marker='.')
            ax2.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], capsize=3.0, fmt='none', color="#1f77b4")
            ax2.plot(t,double_sineCurve(t,*pFit), c="#1f77b4", alpha=0.7)
            ax2.set_xlim(xmin1,xmax1)
            ax2.set_ylim(ymin1,ymax1)
            ax2.set_ylabel("Radial Velocity (Km $s^{-1}$)")
            ax2.grid()
            ax2.minorticks_on()
        
            #Defining third subplot
            ax3.scatter(self.date, self.rv_list[0], marker='.')
            ax3.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], capsize=3.0, fmt='none', color="#1f77b4")
            ax3.plot(t,double_sineCurve(t,*pFit), c="#1f77b4", alpha=0.7)
            ax3.set_xlim(xmin2,xmax2)
            ax3.set_ylim(ymin2,ymax2)
            ax3.grid()
            ax3.minorticks_on()
            
        #Fitting 3 sine waves
        if len(model_fit) == 12:
            t = np.linspace(start=self.date[0], stop=self.date[-1], num=50000)
            pFit, covFit = opt.curve_fit(triple_sineCurve, np.array(self.date), 
                                          list(self.rv_list[0]), model_fit, 
                                          sigma=list(self.rv_uncrt_list[0]),
                                          absolute_sigma=True)

            #Defining first subplot
            ax1.scatter(self.date, self.rv_list[0], marker='.')
            ax1.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], capsize=3.0, fmt='none', color="#1f77b4")
            ax1.plot(t,triple_sineCurve(t,*pFit), c="#1f77b4", alpha=0.7)
            ax1.fill_between((xmin1,xmax1), min(self.rv_list[0]), max(self.rv_list[0]), facecolor='green', alpha=0.2)
            ax1.fill_between((xmin2,xmax2), min(self.rv_list[0]), max(self.rv_list[0]), facecolor='orange', alpha=0.2)
            ax1.set_xlabel("Time (Days)")
            ax1.set_ylabel("Radial Velocity (Km $s^{-1}$)")
            ax1.grid()
            ax1.minorticks_on()
        
            #Defining second subplot
            ax2.scatter(self.date, self.rv_list[0], marker='.')
            ax2.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], capsize=3.0, fmt='none', color="#1f77b4")
            ax2.plot(t,triple_sineCurve(t,*pFit), c="#1f77b4", alpha=0.7)
            ax2.set_xlim(xmin1,xmax1)
            ax2.set_ylim(ymin1,ymax1)
            ax2.set_ylabel("Radial Velocity (Km $s^{-1}$)")
            ax2.grid()
            ax2.minorticks_on()
        
            #Defining third subplot
            ax3.scatter(self.date, self.rv_list[0], marker='.')
            ax3.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], capsize=3.0, fmt='none', color="#1f77b4")
            ax3.plot(t,triple_sineCurve(t,*pFit), c="#1f77b4", alpha=0.7)
            ax3.set_xlim(xmin2,xmax2)
            ax3.set_ylim(ymin2,ymax2)
            ax3.grid()
            ax3.minorticks_on()
            
        #Fitting 4 sine waves
        if len(model_fit) == 16:
            t = np.linspace(start=self.date[0], stop=self.date[-1], num=50000)
            pFit, covFit = opt.curve_fit(quad_sineCurve, np.array(self.date), 
                                          list(self.rv_list[0]), model_fit, 
                                          sigma=list(self.rv_uncrt_list[0]),
                                          absolute_sigma=True)

            #Defining first subplot
            ax1.scatter(self.date, self.rv_list[0], marker='.')
            ax1.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], capsize=3.0, fmt='none', color="#1f77b4")
            ax1.plot(t,quad_sineCurve(t,*pFit), c="#1f77b4", alpha=0.7)
            ax1.fill_between((xmin1,xmax1), min(self.rv_list[0]), max(self.rv_list[0]), facecolor='green', alpha=0.2)
            ax1.fill_between((xmin2,xmax2), min(self.rv_list[0]), max(self.rv_list[0]), facecolor='orange', alpha=0.2)
            ax1.set_xlabel("Time (Days)")
            ax1.set_ylabel("Radial Velocity (Km $s^{-1}$)")
            ax1.grid()
            ax1.minorticks_on()
        
            #Defining second subplot
            ax2.scatter(self.date, self.rv_list[0], marker='.')
            ax2.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], capsize=3.0, fmt='none', color="#1f77b4")
            ax2.plot(t,quad_sineCurve(t,*pFit), c="#1f77b4", alpha=0.7)
            ax2.set_xlim(xmin1,xmax1)
            ax2.set_ylim(ymin1,ymax1)
            ax2.set_ylabel("Radial Velocity (Km $s^{-1}$)")
            ax2.grid()
            ax2.minorticks_on()
        
            #Defining third subplot
            ax3.scatter(self.date, self.rv_list[0], marker='.')
            ax3.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], capsize=3.0, fmt='none', color="#1f77b4")
            ax3.plot(t,quad_sineCurve(t,*pFit), c="#1f77b4", alpha=0.7)
            ax3.set_xlim(xmin2,xmax2)
            ax3.set_ylim(ymin2,ymax2)
            ax3.grid()
            ax3.minorticks_on()
            
        #Fitting 5 sine waves
        if len(model_fit) == 20:
            t = np.linspace(start=self.date[0], stop=self.date[-1], num=50000)
            pFit, covFit = opt.curve_fit(five_sineCurve, np.array(self.date), 
                                          list(self.rv_list[0]), model_fit, 
                                          sigma=list(self.rv_uncrt_list[0]),
                                          absolute_sigma=True)

            #Defining first subplot
            ax1.scatter(self.date, self.rv_list[0], marker='.')
            ax1.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], capsize=3.0, fmt='none', color="#1f77b4")
            ax1.plot(t,five_sineCurve(t,*pFit), c="#1f77b4", alpha=0.7)
            ax1.fill_between((xmin1,xmax1), min(self.rv_list[0]), max(self.rv_list[0]), facecolor='green', alpha=0.2)
            ax1.fill_between((xmin2,xmax2), min(self.rv_list[0]), max(self.rv_list[0]), facecolor='orange', alpha=0.2)
            ax1.set_xlabel("Time (Days)")
            ax1.set_ylabel("Radial Velocity (Km $s^{-1}$)")
            ax1.grid()
            ax1.minorticks_on()
        
            #Defining second subplot
            ax2.scatter(self.date, self.rv_list[0], marker='.')
            ax2.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], capsize=3.0, fmt='none', color="#1f77b4")
            ax2.plot(t,five_sineCurve(t,*pFit), c="#1f77b4", alpha=0.7)
            ax2.set_xlim(xmin1,xmax1)
            ax2.set_ylim(ymin1,ymax1)
            ax2.set_ylabel("Radial Velocity (Km $s^{-1}$)")
            ax2.grid()
            ax2.minorticks_on()
        
            #Defining third subplot
            ax3.scatter(self.date, self.rv_list[0], marker='.')
            ax3.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], capsize=3.0, fmt='none', color="#1f77b4")
            ax3.plot(t,five_sineCurve(t,*pFit), c="#1f77b4", alpha=0.7)
            ax3.set_xlim(xmin2,xmax2)
            ax3.set_ylim(ymin2,ymax2)
            ax3.grid()
            ax3.minorticks_on()
            
        #Fitting 6 sine waves
        if len(model_fit) == 24:
            t = np.linspace(start=self.date[0], stop=self.date[-1], num=50000)
            pFit, covFit = opt.curve_fit(six_sineCurve, np.array(self.date), 
                                          list(self.rv_list[0]), model_fit, 
                                          sigma=list(self.rv_uncrt_list[0]),
                                          absolute_sigma=True)

            #Defining first subplot
            ax1.scatter(self.date, self.rv_list[0], marker='.')
            ax1.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], capsize=3.0, fmt='none', color="#1f77b4")
            ax1.plot(t,six_sineCurve(t,*pFit), c="#1f77b4", alpha=0.7)
            ax1.fill_between((xmin1,xmax1), min(self.rv_list[0]), max(self.rv_list[0]), facecolor='green', alpha=0.2)
            ax1.fill_between((xmin2,xmax2), min(self.rv_list[0]), max(self.rv_list[0]), facecolor='orange', alpha=0.2)
            ax1.set_xlabel("Time (Days)")
            ax1.set_ylabel("Radial Velocity (Km $s^{-1}$)")
            ax1.grid()
            ax1.minorticks_on()
        
            #Defining second subplot
            ax2.scatter(self.date, self.rv_list[0], marker='.')
            ax2.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], capsize=3.0, fmt='none', color="#1f77b4")
            ax2.plot(t,six_sineCurve(t,*pFit), c="#1f77b4", alpha=0.7)
            ax2.set_xlim(xmin1,xmax1)
            ax2.set_ylim(ymin1,ymax1)
            ax2.set_ylabel("Radial Velocity (Km $s^{-1}$)")
            ax2.grid()
            ax2.minorticks_on()
        
            #Defining third subplot
            ax3.scatter(self.date, self.rv_list[0], marker='.')
            ax3.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], capsize=3.0, fmt='none', color="#1f77b4")
            ax3.plot(t,six_sineCurve(t,*pFit), c="#1f77b4", alpha=0.7)
            ax3.set_xlim(xmin2,xmax2)
            ax3.set_ylim(ymin2,ymax2)
            ax3.grid()
            ax3.minorticks_on()
            
        #Fitting 7 sine waves
        if len(model_fit) == 28:
            t = np.linspace(start=self.date[0], stop=self.date[-1], num=50000)
            pFit, covFit = opt.curve_fit(seven_sineCurve, np.array(self.date), 
                                          list(self.rv_list[0]), model_fit, 
                                          sigma=list(self.rv_uncrt_list[0]),
                                          absolute_sigma=True, maxfev=50000)

            #Defining first subplot
            ax1.scatter(self.date, self.rv_list[0], marker='.')
            ax1.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], capsize=3.0, fmt='none', color="#1f77b4")
            ax1.plot(t,seven_sineCurve(t,*pFit), c="#1f77b4", alpha=0.7)
            ax1.fill_between((xmin1,xmax1), min(self.rv_list[0]), max(self.rv_list[0]), facecolor='green', alpha=0.2)
            ax1.fill_between((xmin2,xmax2), min(self.rv_list[0]), max(self.rv_list[0]), facecolor='orange', alpha=0.2)
            ax1.set_xlabel("Time (Days)")
            ax1.set_ylabel("Radial Velocity (Km $s^{-1}$)")
            ax1.grid()
            ax1.minorticks_on()
        
            #Defining second subplot
            ax2.scatter(self.date, self.rv_list[0], marker='.')
            ax2.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], capsize=3.0, fmt='none', color="#1f77b4")
            ax2.plot(t,seven_sineCurve(t,*pFit), c="#1f77b4", alpha=0.7)
            ax2.set_xlim(xmin1,xmax1)
            ax2.set_ylim(ymin1,ymax1)
            ax2.set_ylabel("Radial Velocity (Km $s^{-1}$)")
            ax2.grid()
            ax2.minorticks_on()
        
            #Defining third subplot
            ax3.scatter(self.date, self.rv_list[0], marker='.')
            ax3.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], capsize=3.0, fmt='none', color="#1f77b4")
            ax3.plot(t,seven_sineCurve(t,*pFit), c="#1f77b4", alpha=0.7)
            ax3.set_xlim(xmin2,xmax2)
            ax3.set_ylim(ymin2,ymax2)
            ax3.grid()
            ax3.minorticks_on()
        
        #Fitting 8 sine waves
        if len(model_fit) == 32:
            t = np.linspace(start=self.date[0], stop=self.date[-1], num=50000)
            pFit, covFit = opt.curve_fit(eight_sineCurve, np.array(self.date), 
                                          list(self.rv_list[0]), model_fit, 
                                          sigma=list(self.rv_uncrt_list[0]),
                                          absolute_sigma=True, maxfev=50000)

            #Defining first subplot
            ax1.scatter(self.date, self.rv_list[0], marker='.')
            ax1.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], capsize=3.0, fmt='none', color="#1f77b4")
            ax1.plot(t,eight_sineCurve(t,*pFit), c="#1f77b4", alpha=0.7)
            ax1.fill_between((xmin1,xmax1), min(self.rv_list[0]), max(self.rv_list[0]), facecolor='green', alpha=0.2)
            ax1.fill_between((xmin2,xmax2), min(self.rv_list[0]), max(self.rv_list[0]), facecolor='orange', alpha=0.2)
            ax1.set_xlabel("Time (Days)")
            ax1.set_ylabel("Radial Velocity (Km $s^{-1}$)")
            ax1.grid()
            ax1.minorticks_on()
        
            #Defining second subplot
            ax2.scatter(self.date, self.rv_list[0], marker='.')
            ax2.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], capsize=3.0, fmt='none', color="#1f77b4")
            ax2.plot(t, eight_sineCurve(t,*pFit), c="#1f77b4", alpha=0.7)
            ax2.set_xlim(xmin1,xmax1)
            ax2.set_ylim(ymin1,ymax1)
            ax2.set_ylabel("Radial Velocity (Km $s^{-1}$)")
            ax2.grid()
            ax2.minorticks_on()
        
            #Defining third subplot
            ax3.scatter(self.date, self.rv_list[0], marker='.')
            ax3.errorbar(self.date, self.rv_list[0], self.rv_uncrt_list[0], capsize=3.0, fmt='none', color="#1f77b4")
            ax3.plot(t,eight_sineCurve(t,*pFit), c="#1f77b4", alpha=0.7)
            ax3.set_xlim(xmin2,xmax2)
            ax3.set_ylim(ymin2,ymax2)
            ax3.grid()
            ax3.minorticks_on()


        # Create left side of Connection patch for second subplot
        con1 = patches.ConnectionPatch(xyA=(xmin1, ymin1), coordsA=ax2.transData, 
                               xyB=(xmin1, np.mean(self.rv_list[0])), coordsB=ax1.transData, 
                               color = 'green')
        ax1.add_artist(con1)

        # Create right side of Connection patch for second subplot
        con2 = patches.ConnectionPatch(xyA=(xmax1, ymin1), coordsA=ax2.transData, 
                               xyB=(xmax1, np.mean(self.rv_list[0])), coordsB=ax1.transData, 
                               color = 'green')
        ax2.add_artist(con2)

        # Create left side of Connection patch for third subplot
        con3 = patches.ConnectionPatch(xyA=(xmin2, ymin2), coordsA=ax3.transData, 
                               xyB=(xmin2, np.mean(self.rv_list[0])), coordsB=ax1.transData, color = 'orange')
        ax1.add_artist(con3)

        # Create right side of Connection patch for third subplot
        con4 = patches.ConnectionPatch(xyA=(xmax2, ymin2), coordsA=ax3.transData, 
                               xyB=(xmax2, np.mean(self.rv_list[0])), coordsB=ax1.transData, color = 'orange')
        ax1.add_artist(con4)
        
        #Saving the plot
        plt.savefig(fname, dpi=dpi)
        plt.close()
    
################################################################  
class Tess:
    """Class that utilises TESS data to do neat things"""
    
    def __init__(self, name):
        """Function that initalises the Tess class"""
        self.name = name
        
    def get_data(self):
        """Assuming the user has already made extracted the TESS data using the
        Python pipeline, and that they're stored within the directory ASTR690/
        Data_Analysis/TESS_Analysis by their name eg HD152564, then this 
        function goes into the directory. Then from the 'name' variable, this
        function will retrieve the flux, flux_err and time .dat files by going
        into the star's folder.
        
        INPUTS:
            None
        
        Returns:
            data = numpy array of TESS flux, flux uncertainty and Julian dates
        """
        
        #Extracting the flux, flux_err and jd values from the TESS directory 
        flux = np.loadtxt(self.name + '_flux.dat')
        flux_err = np.loadtxt(self.name + '_flux_err.dat')
        jd = np.loadtxt(self.name + '_time.dat')

        #Storing data values into an array
        data = np.array([flux, flux_err, jd])

        return data
    
    def lightcurve(self, data, model_fit=None, save=False, fname=None, dpi=600):
        """Plots the lightcurve of the star's TESS data.
        
        INPUTS:
            data = numpy array containing TESS flux, flux_err and time values
            save = bool if wanting to save the plot or not. True will save, whilst
                   False won't. Default = False.
            fname = string of figure name. Only used if save = True.   
            
        RETURNS:
            None - a plot is outputted.
        """
        
        #Extracting flux, flux_err and time values from data
        t_flux = data[0]
        t_flux_err = data[1]
        t_time = data[2]
        
        if model_fit == None:
            #Creating plot
            plt.figure()
            plt.grid()
            plt.scatter(t_time, t_flux, s=0.2, c="orange")
            # plt.errorbar(t_time, t_flux, t_flux_err)
            plt.xlabel("Time (days)")
            plt.ylabel("Flux (electron / s)")
            plt.minorticks_on()
            #plt.title("TESS Lightcurve for " + self.name)
            plt.grid()
            
            return None
            
        if model_fit != None:
            t = np.linspace(start=t_time[0], stop=t_time[-1], num=50000)
            #fitting 1 sine wave
            if len(model_fit)==4:
                pFit, covFit = opt.curve_fit(sineCurve, np.array(t_time), 
                                             list(t_flux), model_fit, 
                                             sigma=list(t_flux_err),
                                             absolute_sigma=True)

                #Creating plot
                plt.figure()
                plt.grid()
                plt.scatter(t_time, t_flux, s=0.2, c="orange")
                plt.plot(t,sineCurve(t,*pFit))
                # plt.errorbar(t_time, t_flux, t_flux_err)
                plt.xlabel("Time (days)")
                plt.ylabel("Flux (electron / s)")
                #plt.title("TESS Lightcurve for " + self.name)
                plt.grid()
                plt.minorticks_on()
                
                #Calculating rms value
                synth_vals = sineCurve(np.array(t_time), *pFit)
                rms_val = rms(t_flux, synth_vals, t_time)
                print("RMSE value for this TESS model fit is: {}".format(rms_val))
                
            #Fitting 2 sine waves
            if len(model_fit)==8:
                pFit, covFit = opt.curve_fit(double_sineCurve, np.array(t_time), 
                                             list(t_flux), model_fit, 
                                             sigma=list(t_flux_err),
                                             absolute_sigma=True)

                #Creating plot
                plt.figure()
                plt.grid()
                plt.scatter(t_time, t_flux, s=0.2, c="orange")
                plt.plot(t,double_sineCurve(t,*pFit))
                # plt.errorbar(t_time, t_flux, t_flux_err)
                plt.xlabel("Time (days)")
                plt.ylabel("Flux (electron / s)")
                #plt.title("TESS Lightcurve for " + self.name)
                plt.grid()
                plt.minorticks_on()
                
                #Calculating rms value
                synth_vals = double_sineCurve(np.array(t_time), *pFit)
                rms_val = rms(t_flux, synth_vals, t_time)
                print("RMSE value for this TESS model fit is: {}".format(rms_val))
                
            #Fitting 3 sine waves
            if len(model_fit)==12:
                pFit, covFit = opt.curve_fit(triple_sineCurve, np.array(t_time), 
                                             list(t_flux), model_fit, 
                                             sigma=list(t_flux_err),
                                             absolute_sigma=True, maxfev=30000)

                #Creating plot
                # plt.figure()
                plt.figure()
                plt.grid()
                plt.scatter(t_time, t_flux, s=0.2, c="orange")
                plt.plot(t,triple_sineCurve(t,*pFit))
                # plt.errorbar(t_time, t_flux, t_flux_err)
                plt.xlabel("Time (days)")
                plt.ylabel("Flux (electron / s)")
                #plt.title("TESS Lightcurve for " + self.name)
                plt.minorticks_on()
                
                # plt.savefig(fname, dpi=dpi)
                
                #Calculating rms value
                synth_vals = triple_sineCurve(np.array(t_time), *pFit)
                rms_val = rms(t_flux, synth_vals, t_time)
                print("RMSE value for this TESS model fit is: {}".format(rms_val))

            #Fitting 4 sine waves
            if len(model_fit)==16:
                pFit, covFit = opt.curve_fit(quad_sineCurve, np.array(t_time), 
                                             list(t_flux), model_fit, 
                                             sigma=list(t_flux_err),
                                             absolute_sigma=True)

                #Creating plot
                plt.figure()
                plt.grid()
                plt.scatter(t_time, t_flux, s=0.2, c="orange")
                plt.plot(t,quad_sineCurve(t,*pFit))
                # plt.errorbar(t_time, t_flux, t_flux_err)
                plt.xlabel("Time (days)")
                plt.ylabel("Flux (electron / s)")
                #plt.title("TESS Lightcurve for " + self.name)
                plt.grid()
                plt.minorticks_on()
                
                #Calculating rms value
                synth_vals = quad_sineCurve(np.array(t_time), *pFit)
                rms_val = rms(t_flux, synth_vals, t_time)
                print("RMSE value for this TESS model fit is: {}".format(rms_val))
                
                
            #Fitting 5 sine waves
            if len(model_fit)==20:
                pFit, covFit = opt.curve_fit(five_sineCurve, np.array(t_time), 
                                             list(t_flux), model_fit, 
                                             sigma=list(t_flux_err),
                                             absolute_sigma=True, maxfev=100000)


                #Creating plot
                plt.figure()
                plt.grid()
                plt.scatter(t_time, t_flux, s=0.2, c="orange")
                plt.plot(t,five_sineCurve(t,*pFit))
                # plt.errorbar(t_time, t_flux, t_flux_err)
                plt.xlabel("Time (days)")
                plt.ylabel("Flux (electron / s)")
                #plt.title("TESS Lightcurve for " + self.name)
                plt.grid()
                plt.minorticks_on()
                
                #Calculating rms value
                synth_vals = five_sineCurve(np.array(t_time), *pFit)
                rms_val = rms(t_flux, synth_vals, t_time)
                print("RMSE value for this TESS model fit is: {}".format(rms_val))
                
            
            #Fitting 6 sine waves
            if len(model_fit)==24:
                pFit, covFit = opt.curve_fit(six_sineCurve, np.array(t_time), 
                                             list(t_flux), model_fit, 
                                             sigma=list(t_flux_err),
                                             absolute_sigma=True, maxfev=100000)

                #Creating plot
                plt.figure()
                plt.grid()
                plt.scatter(t_time, t_flux, s=0.2, c="orange")
                plt.plot(t,six_sineCurve(t,*pFit))
                # plt.errorbar(t_time, t_flux, t_flux_err)
                plt.xlabel("Time (days)")
                plt.ylabel("Flux (electron / s)")
                #plt.title("TESS Lightcurve for " + self.name)
                plt.grid()
                plt.minorticks_on()
                
                #Calculating rms value
                synth_vals = six_sineCurve(np.array(t_time), *pFit)
                rms_val = rms(t_flux, synth_vals, t_time)
                print("RMSE value for this TESS model fit is: {}".format(rms_val))
                
            #Fitting 7 sine waves
            if len(model_fit)==28:
                pFit, covFit = opt.curve_fit(seven_sineCurve, np.array(t_time), 
                                             list(t_flux), model_fit, 
                                             sigma=list(t_flux_err),
                                             absolute_sigma=True, maxfev=500000)

                #Creating plot
                plt.figure()
                plt.grid()
                plt.scatter(t_time, t_flux, s=0.2, c="orange")
                plt.plot(t,seven_sineCurve(t,*pFit))
                # plt.errorbar(t_time, t_flux, t_flux_err)
                plt.xlabel("Time (days)")
                plt.ylabel("Flux (electron / s)")
                #plt.title("TESS Lightcurve for " + self.name)
                plt.minorticks_on()
                
                # plt.savefig(fname, dpi=dpi)
                
                #Calculating rms value
                synth_vals = seven_sineCurve(np.array(t_time), *pFit)
                rms_val = rms(t_flux, synth_vals, t_time)
                print("RMSE value for this TESS model fit is: {}".format(rms_val))
                
            #Fitting 8 sine waves
            if len(model_fit)==32:
                pFit, covFit = opt.curve_fit(eight_sineCurve, np.array(t_time), 
                                             list(t_flux), model_fit, 
                                             sigma=list(t_flux_err),
                                             absolute_sigma=True, maxfev=500000)

                #Creating plot
                plt.figure()
                plt.grid()
                plt.scatter(t_time, t_flux, s=0.2, c="orange")
                plt.plot(t,eight_sineCurve(t,*pFit))
                # plt.errorbar(t_time, t_flux, t_flux_err)
                plt.xlabel("Time (days)")
                plt.ylabel("Flux (electron / s)")
                #plt.title("TESS Lightcurve for " + self.name)
                plt.minorticks_on()
                
                
                #Calculating rms value
                synth_vals = eight_sineCurve(np.array(t_time), *pFit)
                rms_val = rms(t_flux, synth_vals, t_time)
                print("RMSE value for this TESS model fit is: {}".format(rms_val))
                
            #Fitting 9 sine waves
            if len(model_fit)==36:
                pFit, covFit = opt.curve_fit(nine_sineCurve, np.array(t_time), 
                                             list(t_flux), model_fit, 
                                             sigma=list(t_flux_err),
                                             absolute_sigma=True, maxfev=500000)

                #Creating plot
                plt.figure()
                plt.grid()
                plt.scatter(t_time, t_flux, s=0.2, c="orange")
                plt.plot(t,nine_sineCurve(t,*pFit))
                # plt.errorbar(t_time, t_flux, t_flux_err)
                plt.xlabel("Time (days)")
                plt.ylabel("Flux (electron / s)")
                #plt.title("TESS Lightcurve for " + self.name)
                plt.grid()
                plt.minorticks_on()
                
                #Calculating rms value
                synth_vals = nine_sineCurve(np.array(t_time), *pFit)
                rms_val = rms(t_flux, synth_vals, t_time)
                print("RMSE value for this TESS model fit is: {}".format(rms_val))
        
        # if fname != None:
        #     plt.grid()
        #     plt.savefig(fname, dpi=dpi)

                
        return [pFit, covFit]
                
    def remove_nan(self, data):
        """Function to remove all nans in tess data.
        
        INPUTS:
            data = numpy array containing TESS flux, flux_err and time values.
            
        RETURNS:
            data_nanless = numpy array containing TESS flux, flux_err and time 
                           values with all nan entries removed.
        """
        
        #Extracting flux, flux_err and time values from data
        t_flux = data[0]
        t_flux_err = data[1]
        t_time = data[2]
        
        #Removing all nans
        nan_index = np.argwhere(np.isnan(t_flux)) #Finding index of each nan
                                                  #entry in t_flux  

        #Removing all nan entries using their index from nan_index, and 
        #adjusting shape of the t_time array accordingly
        nanless_t_flux = np.delete(t_flux, nan_index) 
        nanless_t_flux_err = np.delete(t_flux_err, nan_index)
        nanless_t_time = np.delete(t_time, nan_index)

        data_nanless = np.array([nanless_t_flux, nanless_t_flux_err, nanless_t_time])
            
        return data_nanless
            
        
################################################################          
class Doppler_Imaging:
    """Class that prepares data for Doppler Imaging."""
    
    def __init__(self, waves, ints, dates, snr, period):
        """Initialises the Doppler_Imaging class.
        
        INPUTS:
            waves = wavelength array of spectral data
            ints = intensity array of spectral data
            dates = list of dates starting at 0
            snr = signal-to-noise ratios
            period = rotational period of star
        """
        
        self.waves = waves
        self.ints = ints
        self.dates = dates
        self.snr = snr
        self.period = period
        
    def binned(self):
        """Computes the phases of the dataset, organizes it in order from 0 to 1,
        determines each of their indexes, then uses these to organise items
        into phase bins.
        
        RETURNS:
            phased_array = array containing two lists: 1st corresponding to 20
                           individual arrays pertaining to binned phases with
                           their intensities, whilst 2nd corresponds to the 
                           signal-to-noise ratios for each binned phase.
                       
        """
        
        #Computing phase from 2.16375 day period 
        cycle = np.array(self.dates)/(self.period)
        phase_array = cycle - np.floor(cycle)

        #Using list comprehension to bin phases to size 0.05
        ph_1 = [ph for ph in phase_array if ph >= 0 and ph < 0.05]
        ph_2 = [ph for ph in phase_array if ph >= 0.05 and ph < 0.1]
        ph_3 = [ph for ph in phase_array if ph >= 0.1 and ph < 0.15]
        ph_4 = [ph for ph in phase_array if ph >= 0.15 and ph < 0.2]
        ph_5 = [ph for ph in phase_array if ph >= 0.2 and ph < 0.25]
        ph_6 = [ph for ph in phase_array if ph >= 0.25 and ph < 0.3]
        ph_7 = [ph for ph in phase_array if ph >= 0.3 and ph < 0.35]
        ph_8 = [ph for ph in phase_array if ph >= 0.35 and ph < 0.4]
        ph_9 = [ph for ph in phase_array if ph >= 0.4 and ph < 0.45]
        ph_10 = [ph for ph in phase_array if ph >= 0.45 and ph < 0.5]
        ph_11 = [ph for ph in phase_array if ph >= 0.5 and ph < 0.55]
        ph_12 = [ph for ph in phase_array if ph >= 0.55 and ph < 0.6]
        ph_13 = [ph for ph in phase_array if ph >= 0.6 and ph < 0.65]
        ph_14 = [ph for ph in phase_array if ph >= 0.65 and ph < 0.7]
        ph_15 = [ph for ph in phase_array if ph >= 0.7 and ph < 0.75]
        ph_16 = [ph for ph in phase_array if ph >= 0.75 and ph < 0.8]
        ph_17 = [ph for ph in phase_array if ph >= 0.8 and ph < 0.85]
        ph_18 = [ph for ph in phase_array if ph >= 0.85 and ph < 0.9]
        ph_19 = [ph for ph in phase_array if ph >= 0.9 and ph < 0.95]
        ph_20 = [ph for ph in phase_array if ph >= 0.95 and ph < 1]
        
        #Finding the index of all phases
        def index(ph_group):
            """Determines the index of each entry in the phase groups,
            according to their observation number from orb_phase.

            INPUTS:
                ph_group = list of phase groups.

            RETURNS:
                all_ints = list of phase groups with observation number
                            as their entries.
            """
            #Finding the index of each phase
            ind_group = []
            for i, ph in enumerate(ph_group):
                ind_list = []
                for value in ph:
                    ind = list(phase_array).index(value)
                    ind_list.append(ind)
                ind_group.append(ind_list)

                if len(ph) != 0:
                    print("Phase {0} has {1} data points within it.".format(i+1, len(ph)))
                elif len(ph) == 0:
                    print("Phase {} is empty.".format(i+1))

            #Using phase index, extracting intensities and SNRs in phase groups
            int_group = []
            snr_group = []
            for ind in ind_group:
                if len(ind) != 0:
                    #Organising intensities into binned phases
                    ints = self.ints[ind]
                    int_group.append(ints)
                    #Organising SNRs into binned phases
                    snrs = self.snr[ind]
                    snr_group.append(snrs)
            
            #Storing int_group and snr_group into an array
            binned_array = np.array([int_group, snr_group], dtype=object)
            
            return np.array([binned_array, ind_group], dtype=object)

        phs = [ph_1, ph_2, ph_3, ph_4, ph_5, ph_6, ph_7, ph_8, ph_9, ph_10,
               ph_11, ph_12, ph_13, ph_14, ph_15, ph_16, ph_17, ph_18, ph_19, ph_20]
        phased_array, index_array = index(phs)
        
        return np.array([phased_array, index_array], dtype=object)

    def med_binned_ints(self, ints_binned):
        """Takes the binned intensities from the binned function, and determines
        the medians of each of them.
        
        INPUTS:
            ints_binned = list of intensities organized by rotational phase 
                          (computed using the index function).
        
        RETURNS:
            all_med_ints = list of lists, with each inner list containing
                           the median intesnity at a specific phase range.
       """
        
        all_med_ints = []
        for i, ints in enumerate(ints_binned):
            #Transposing the intensity list
            trans_int = ints.T

            #Computing the median
            med_int = []
            for j in range(len(trans_int)):
                jth_med = np.nanmedian(trans_int[j])
                med_int.append(jth_med)

            all_med_ints.append(med_int)
            print("Median of the {0} phase has been completed!".format(i+1))
            
        return all_med_ints
    
    def mean_binned_ints(self, ints_binned):
        """Takes the binned intensities from the binned function, and determines
        the means of each of them.
        
        INPUTS:
            ints_binned = list of intensities organized by rotational phase 
                          (computed using the index function).
        
        RETURNS:
            all_mean_ints = list of lists, with each inner list containing
                            the mean intesnity at a specific phase range.
       """
        
        all_mean_ints = []
        for i, ints in enumerate(ints_binned):
            #Transposing the intensity list
            trans_int = ints.T

            #Computing the median
            mean_int = []
            for j in range(len(trans_int)):
                jth_mean = np.nanmean(trans_int[j])
                mean_int.append(jth_mean)

            all_mean_ints.append(mean_int)
            print("Mean of the {0} phase has been completed!".format(i+1))
            
        return all_mean_ints
    
    def med_binned_snr(self, snr_binned):
        """Takes the binned signal-to-noise ratios from the binned_ints function, and determines
        the medians of each of them.
        
        INPUTS:
            snr_binned = list of signal=to-noise ratios organized by rotational 
                         phase (computed using the index function).
        
        RETURNS:
            all_med_snr = list of lists, with each inner list containing
                           the median intesnity at a specific phase range.
       """
        
        all_med_snr = []
        for i, snr in enumerate(snr_binned):
            #Computing the median
            snr_med = np.nanmedian(snr)

            all_med_snr.append(snr_med)
            print("Median of the {0} phase has been completed!".format(i+1))
            
        return all_med_snr
    
    def wave_range(self, binned_ints, min_wave, max_wave):
        """Crops out the wavelength ranges being used for Doppler Imaging by 
        extracting their corresponding intensity values.
        
        INPUTS:
            binned_ints = list of intensities organized by rotational phase 
            min_wave = minimum wavelength range.
            max_wave = maximum wavelength range.
                              
        RETURNS:
            phased_ints_crop = list of lists, with each inner list having the
                               intensities defined by the min_wave and max_wave 
                               wavelength ranges.
        """
        #Getting index values for upper and lower wavelength boundaries.
        index_min = list(self.waves).index(min_wave)
        index_max = list(self.waves).index(max_wave)
        
        #Checking if binned_ints has each bin being the average of the observations
        #or not. If it is, then when doing eg len(binned_ints[0]), it should be some
        #big number of around 177399, denoting that it's the whole spectra which
        #has had the median being done on it. However, if the length of binned_ints
        #is a small number like 13, then when doing eg len(binned_ints[0]), it 
        #has each individual spectra embodied within that rotational phase. This
        #will be checked by doing these tests to see if the length is large or 
        #small.
        
        #binned_ints has the average of each rotational phase
        if len(binned_ints[0]) > 100:
            #Using these index values to extract corresponding intensities
            phased_ints_crop = []
            for ph in binned_ints:
                int_crop = ph[index_min:index_max+1]
                phased_ints_crop.append(int_crop)
       
        #binned_ints has the individual spectra in each rotational phase
        if len(binned_ints[0]) < 100:
            phased_ints_crop = []
            for ph in binned_ints:
                obs_ints_crop = []
                for spectra in ph:
                    int_crop = spectra[index_min:index_max+1]
                    obs_ints_crop.append(int_crop)
                phased_ints_crop.append(obs_ints_crop)
                
        return phased_ints_crop


    def fatplot(self, binned_ints, wave, txt_xpos, ymin, txt_ypos=1.015, ymax=1.05, 
                binned_ints2=0, wave2=0, seg = 0, fname=None):
        """Takes in an array of binned ints over 20 rotational phases of size 
        0.05, and makes a fat
        plot of 20 subplots of all the spectra in their corresponding bin.
        
        INPUTS:
            binned_ints = an array containing 20 arrays corresponding to a 
                          specific rotation10al phase.  Each array has its own list 
                          of spectra observations. Recommended that this is 
                          cropped to only consider a specific absorption line.
            wave = array of wavelengths. Recommended that this is cropped to 
                   only consider a specific absorption line.     
            txt_xpos = float x-axis position for the phase text in wavelength-space
            ymin = float minimum intensity value for subplots.
            txt_ypos = float y-axis position for the phase text in intensity-space.
                       Default = 1.015.
            ymax = float maximum intensity value for subplots. Default = 1.05
            binned_ints2 = secondary array of intensities, typically used if 
                           wanting to plot two datasets on one figure (e.g. 
                           original vs shifted). Defualt = 0. 
            wave2 = secondary array of wavelengths, typically used if 
                    wanting to plot two datasets on one figure (e.g. 
                    original vs shifted). Defualt = 0.    
            seg = float specifying the seperation between the shifted and original
                  datasets. Only used if plotting the shifted and original datasets
                  on the same plot. Default = 0.                    
            fname = string that saves the file based off of this. Default = None.
            
        RETURNS:
            None
        """
        if binned_ints2 == 0 and wave2 == 0:
            fig = plt.figure()
            gs = fig.add_gridspec(nrows=5, ncols=4, hspace=0, wspace=0)
            (ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12), (ax13, ax14, ax15, ax16), (ax17, ax18, ax19, ax20) = gs.subplots(sharex='col', sharey='row')
            
            # if title != None:
            #     fig.suptitle(title, fontsize=16)
            
            # else:
            #     pass
            
            #Adding gridlines to subplots
            ax1.grid()
            ax2.grid()
            ax3.grid()
            ax4.grid()
            ax5.grid()
            ax6.grid()
            ax7.grid()
            ax8.grid()
            ax9.grid()
            ax10.grid()
            ax11.grid()
            ax12.grid()
            ax13.grid()
            ax14.grid()
            ax15.grid()
            ax16.grid()
            ax17.grid()
            ax18.grid()
            ax19.grid()
            ax20.grid()
            
            #Setting minimum and maximum y limits        
            ax1.set_ylim([ymin, ymax])
            ax2.set_ylim([ymin, ymax])
            ax3.set_ylim([ymin, ymax])
            ax4.set_ylim([ymin, ymax])
            ax5.set_ylim([ymin, ymax])
            ax6.set_ylim([ymin, ymax])
            ax7.set_ylim([ymin, ymax])
            ax8.set_ylim([ymin, ymax])
            ax9.set_ylim([ymin, ymax])
            ax10.set_ylim([ymin, ymax])
            ax11.set_ylim([ymin, ymax])
            ax12.set_ylim([ymin, ymax])
            ax13.set_ylim([ymin, ymax])
            ax14.set_ylim([ymin, ymax])
            ax15.set_ylim([ymin, ymax])
            ax16.set_ylim([ymin, ymax])
            ax17.set_ylim([ymin, ymax])
            ax18.set_ylim([ymin, ymax])
            ax19.set_ylim([ymin, ymax])
            ax20.set_ylim([ymin, ymax])
            
            #Adding text to subplots denoting the phases of each       
            ax1.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.05")
            ax2.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.10")
            ax3.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.15")
            ax4.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.20")
            ax5.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.25")
            ax6.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.30")
            ax7.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.35")
            ax8.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.40")
            ax9.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.45")
            ax10.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.50")
            ax11.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.55")
            ax12.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.60")
            ax13.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.65")
            ax14.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.70")
            ax15.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.75")
            ax16.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.80")
            ax17.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.85")
            ax18.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.90")
            ax19.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.95")
            ax20.text(x=txt_xpos, y=txt_ypos, s="$\phi$=1.00")
            
            #Checking if binned_ints has 20 phases or not
            if len(binned_ints) == 20:
                for i in range(len(binned_ints)):
                    
                    if i == 0:
                        for j in binned_ints[i]:
                            ax1.plot(wave, j, )
                    
                    if i == 1:
                        for j in binned_ints[i]:
                            ax2.plot(wave, j)
                            
                    if i == 2:
                        for j in binned_ints[i]:
                            ax3.plot(wave, j)
                            
                    if i == 3:
                        for j in binned_ints[i]:
                            ax4.plot(wave, j)
                            
                    if i == 4:
                        for j in binned_ints[i]:
                            ax5.plot(wave, j)
                            
                    if i == 5:
                        for j in binned_ints[i]:
                            ax6.plot(wave, j)
                    
                    if i == 6:
                        for j in binned_ints[i]:
                            ax7.plot(wave, j)
                            
                    if i == 7:
                        for j in binned_ints[i]:
                            ax8.plot(wave, j)
                            
                    if i == 8:
                        for j in binned_ints[i]:
                            ax9.plot(wave, j)
                            
                    if i == 9:
                        for j in binned_ints[i]:
                            ax10.plot(wave, j)
                            
                    if i == 10:
                        for j in binned_ints[i]:
                            ax11.plot(wave, j)
                    
                    if i == 11:
                        for j in binned_ints[i]:
                            ax12.plot(wave, j)
                            
                    if i == 12:
                        for j in binned_ints[i]:
                            ax13.plot(wave, j)
                            
                    if i == 13:
                        for j in binned_ints[i]:
                            ax14.plot(wave, j)
                            
                    if i == 14:
                        for j in binned_ints[i]:
                            ax15.plot(wave, j)
                            
                    if i == 15:
                        for j in binned_ints[i]:
                            ax16.plot(wave, j)
                    
                    if i == 16:
                        for j in binned_ints[i]:
                            ax17.plot(wave, j)
                            
                    if i == 17:
                        for j in binned_ints[i]:
                            ax18.plot(wave, j)
                            
                    if i == 18:
                        for j in binned_ints[i]:
                            ax19.plot(wave, j)
                            
                    if i == 19:
                        for j in binned_ints[i]:
                            ax20.plot(wave, j)
            
            else:
                print("This dataset doesn't have the full 20 phases. Please open" +
                      " source code and manually adjust it so that one of the subplots" +
                      " is empty. This can be done in the 'else' statement.")
                
                #For HD129899 with 11th bin empty
                for i in range(len(binned_ints)):
                    
                    if i == 0:
                        for j in binned_ints[i]:
                            ax1.plot(wave, j)
                    
                    if i == 1:
                        for j in binned_ints[i]:
                            ax2.plot(wave, j)
                            
                    if i == 2:
                        for j in binned_ints[i]:
                            ax3.plot(wave, j)
                            
                    if i == 3:
                        for j in binned_ints[i]:
                            ax4.plot(wave, j)
                            
                    if i == 4:
                        for j in binned_ints[i]:
                            ax5.plot(wave, j)
                            
                    if i == 5:
                        for j in binned_ints[i]:
                            ax6.plot(wave, j)
                    
                    if i == 6:
                        for j in binned_ints[i]:
                            ax7.plot(wave, j)
                            
                    if i == 7:
                        for j in binned_ints[i]:
                            ax8.plot(wave, j)
                            
                    if i == 8:
                        for j in binned_ints[i]:
                            ax9.plot(wave, j)
                            
                    if i == 9:
                        for j in binned_ints[i]:
                            ax10.plot(wave, j)
                            
                    if i == 10:
                        for j in binned_ints[i]:
                            ax12.plot(wave, j)
                    
                    if i == 11:
                        for j in binned_ints[i]:
                            ax13.plot(wave, j)
                            
                    if i == 12:
                        for j in binned_ints[i]:
                            ax14.plot(wave, j)
                            
                    if i == 13:
                        for j in binned_ints[i]:
                            ax15.plot(wave, j)
                            
                    if i == 14:
                        for j in binned_ints[i]:
                            ax16.plot(wave, j)
                            25
                    if i == 15:
                        for j in binned_ints[i]:
                            ax17.plot(wave, j)
                    
                    if i == 16:
                        for j in binned_ints[i]:
                            ax18.plot(wave, j)
                            
                    if i == 17:
                        for j in binned_ints[i]:
                            ax19.plot(wave, j)
                            
                    if i == 18:
                        for j in binned_ints[i]:
                            ax20.plot(wave, j)
                
            
    
        else:
            fig = plt.figure(figsize=[10,10])
            gs = fig.add_gridspec(nrows=5, ncols=4, hspace=0, wspace=0)
            (ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12), (ax13, ax14, ax15, ax16), (ax17, ax18, ax19, ax20) = gs.subplots(sharex='col', sharey='row')
            
            fig.supxlabel("Wavelength ($\AA$)")
            fig.supylabel("Intensity (W m$^{-2}$)")
            
            
            #Adding gridlines to subplots
            ax1.grid()
            ax2.grid()
            ax3.grid()
            ax4.grid()
            ax5.grid()
            ax6.grid()
            ax7.grid()
            ax8.grid()
            ax9.grid()
            ax10.grid()
            ax11.grid()
            ax12.grid()
            ax13.grid()
            ax14.grid()
            ax15.grid()
            ax16.grid()
            ax17.grid()
            ax18.grid()
            ax19.grid()
            ax20.grid()
            
            #Setting minimum and maximum y limits        
            ax1.set_ylim([ymin, ymax])
            ax2.set_ylim([ymin, ymax])
            ax3.set_ylim([ymin, ymax])
            ax4.set_ylim([ymin, ymax])
            ax5.set_ylim([ymin, ymax])
            ax6.set_ylim([ymin, ymax])
            ax7.set_ylim([ymin, ymax])
            ax8.set_ylim([ymin, ymax])
            ax9.set_ylim([ymin, ymax])
            ax10.set_ylim([ymin, ymax])
            ax11.set_ylim([ymin, ymax])
            ax12.set_ylim([ymin, ymax])
            ax13.set_ylim([ymin, ymax])
            ax14.set_ylim([ymin, ymax])
            ax15.set_ylim([ymin, ymax])
            ax16.set_ylim([ymin, ymax])
            ax17.set_ylim([ymin, ymax])
            ax18.set_ylim([ymin, ymax])
            ax19.set_ylim([ymin, ymax])
            ax20.set_ylim([ymin, ymax])
            
            #Adding text to subplots denoting the phases of each       
            ax1.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.05")
            ax2.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.10")
            ax3.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.15")
            ax4.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.20")
            ax5.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.25")
            ax6.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.30")
            ax7.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.35")
            ax8.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.40")
            ax9.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.45")
            ax10.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.50")
            ax11.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.55")
            ax12.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.60")
            ax13.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.65")
            ax14.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.70")
            ax15.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.75")
            ax16.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.80")
            ax17.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.85")
            ax18.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.90")
            ax19.text(x=txt_xpos, y=txt_ypos, s="$\phi$=0.95")
            ax20.text(x=txt_xpos, y=txt_ypos, s="$\phi$=1.00")
            
            #Adjusting rotation of xlabel ticks
            ax17.tick_params(axis='x', labelrotation=45)
            ax18.tick_params(axis='x', labelrotation=45)
            ax19.tick_params(axis='x', labelrotation=45)
            ax20.tick_params(axis='x', labelrotation=45)
            
            #Checking if binned_ints has 20 phases or not
            if len(binned_ints) == 20:
                for i in range(len(binned_ints)):
                    
                    if i == 0:
                        for j in binned_ints[i]:
                            ax1.plot(wave, j, )
                            ax1.minorticks_on()
                        for j in binned_ints2[i]:
                            ax1.plot(wave2, j + seg, alpha=0.2)
                    
                    if i == 1:
                        for j in binned_ints[i]:
                            ax2.plot(wave, j)   
                            ax2.minorticks_on()
                        for j in binned_ints2[i]:
                            ax2.plot(wave2, j + seg, alpha=0.2)
                            
                    if i == 2:
                        for j in binned_ints[i]:
                            ax3.plot(wave, j)
                            ax3.minorticks_on()
                        for j in binned_ints2[i]:
                            ax3.plot(wave2, j + seg, alpha=0.2)
                            
                    if i == 3:
                        for j in binned_ints[i]:
                            ax4.plot(wave, j)
                            ax4.minorticks_on()
                        for j in binned_ints2[i]:
                            ax4.plot(wave2, j + seg, alpha=0.2)
                            
                    if i == 4:
                        for j in binned_ints[i]:
                            ax5.plot(wave, j)
                            ax5.minorticks_on()
                        for j in binned_ints2[i]:
                            ax5.plot(wave2, j + seg, alpha=0.2)
                          
                    if i == 5:
                        for j in binned_ints[i]:
                            ax6.plot(wave, j)
                            ax6.minorticks_on()
                        for j in binned_ints2[i]:
                            ax6.plot(wave2, j + seg, alpha=0.2)
                    
                    if i == 6:
                        for j in binned_ints[i]:
                            ax7.plot(wave, j)
                            ax7.minorticks_on()
                        for j in binned_ints2[i]:
                            ax7.plot(wave2, j + seg, alpha=0.2)
                            
                    if i == 7:
                        for j in binned_ints[i]:
                            ax8.plot(wave, j)
                            ax8.minorticks_on()
                        for j in binned_ints2[i]:
                            ax8.plot(wave2, j + seg, alpha=0.2)
                            
                    if i == 8:
                        for j in binned_ints[i]:
                            ax9.plot(wave, j)
                            ax9.minorticks_on()
                        for j in binned_ints2[i]:
                            ax9.plot(wave2, j + seg, alpha=0.2)
                            
                    if i == 9:
                        for j in binned_ints[i]:
                            ax10.plot(wave, j)
                            ax10.minorticks_on()
                        for j in binned_ints2[i]:
                            ax10.plot(wave2, j + seg, alpha=0.2)
                            
                    if i == 10:
                        for j in binned_ints[i]:
                            ax11.plot(wave, j)
                            ax11.minorticks_on()
                        for j in binned_ints2[i]:
                            ax11.plot(wave2, j + seg, alpha=0.2)
                    
                    if i == 11:
                        for j in binned_ints[i]:
                            ax12.plot(wave, j)
                            ax12.minorticks_on()
                        for j in binned_ints2[i]:
                            ax12.plot(wave2, j + seg, alpha=0.2)
                            
                    if i == 12:
                        for j in binned_ints[i]:
                            ax13.plot(wave, j)
                            ax13.minorticks_on()
                        for j in binned_ints2[i]:
                            ax13.plot(wave2, j + seg, alpha=0.2)
                            
                    if i == 13:
                        for j in binned_ints[i]:
                            ax14.plot(wave, j)
                            ax14.minorticks_on()
                        for j in binned_ints2[i]:
                            ax14.plot(wave2, j + seg, alpha=0.2)
                            
                    if i == 14:
                        for j in binned_ints[i]:
                            ax15.plot(wave, j)
                            ax15.minorticks_on()
                        for j in binned_ints2[i]:
                            ax15.plot(wave2, j + seg, alpha=0.2)
                            
                    if i == 15:
                        for j in binned_ints[i]:
                            ax16.plot(wave, j)
                            ax16.minorticks_on()
                        for j in binned_ints2[i]:
                            ax16.plot(wave2, j + seg, alpha=0.2)
                            
                    if i == 16:
                        for j in binned_ints[i]:
                            ax17.plot(wave, j)
                            ax17.minorticks_on()
                        for j in binned_ints2[i]:
                            ax17.plot(wave2, j + seg, alpha=0.2)
                            
                    if i == 17:
                        for j in binned_ints[i]:
                            ax18.plot(wave, j)
                            ax18.minorticks_on()
                        for j in binned_ints2[i]:
                            ax18.plot(wave2, j + seg, alpha=0.2)
                            
                    if i == 18:
                        for j in binned_ints[i]:
                            ax19.plot(wave, j)
                            ax19.minorticks_on()
                        for j in binned_ints2[i]:
                            ax19.plot(wave2, j + seg, alpha=0.2)
                            
                    if i == 19:
                        for j in binned_ints[i]:
                            ax20.plot(wave, j)
                            ax20.minorticks_on()
                        for j in binned_ints2[i]:
                            ax20.plot(wave2, j + seg, alpha=0.2)
                            
    
            else:
                print("This dataset doesn't have the full 20 phases. Please open" +
                      " source code and manually adjust it so that one of the subplots" +
                      " is empty. This can be done in the 'else' statement.")
                
                #For HD129899 with 5th 18th bins empty
                for i in range(len(binned_ints)):
                    
                    if i == 0:
                        for j in binned_ints[i]:
                            ax1.plot(wave, j)
                        for j in binned_ints2[i]:
                            ax1.plot(wave2, j, alpha=0.2)    
                        
                    
                    if i == 1:
                        for j in binned_ints[i]:
                            ax2.plot(wave, j)
                        for j in binned_ints2[i]:
                            ax2.plot(wave2, j, alpha=0.2)
                            
                    if i == 2:
                        for j in binned_ints[i]:
                            ax3.plot(wave, j)
                        for j in binned_ints2[i]:
                            ax3.plot(wave2, j, alpha=0.2)
                            
                    if i == 3:
                        for j in binned_ints[i]:
                            ax4.plot(wave, j)
                        for j in binned_ints2[i]:
                            ax4.plot(wave2, j, alpha=0.2)
                            
                    if i == 4:
                        for j in binned_ints[i]:
                            ax5.plot(wave, j)
                        for j in binned_ints2[i]:
                            ax5.plot(wave2, j, alpha=0.2)
                            
                    if i == 5:
                        for j in binned_ints[i]:
                            ax6.plot(wave, j)
                        for j in binned_ints2[i]:
                            ax6.plot(wave2, j, alpha=0.2)
                            
                    if i == 6:
                        for j in binned_ints[i]:
                            ax7.plot(wave, j)
                        for j in binned_ints2[i]:
                            ax7.plot(wave2, j, alpha=0.2)
                    
                    if i == 7:
                        for j in binned_ints[i]:
                            ax8.plot(wave, j)
                        for j in binned_ints2[i]:
                            ax8.plot(wave2, j, alpha=0.2)
                            
                    if i == 8:
                        for j in binned_ints[i]:
                            ax9.plot(wave, j)
                        for j in binned_ints2[i]:
                            ax9.plot(wave2, j, alpha=0.2)
                            
                    if i == 9:
                        for j in binned_ints[i]:
                            ax10.plot(wave, j)
                        for j in binned_ints2[i]:
                            ax10.plot(wave2, j, alpha=0.2)
                            
                    if i == 10:
                        for j in binned_ints[i]:
                            ax12.plot(wave, j)
                        for j in binned_ints2[i]:
                            ax12.plot(wave2, j, alpha=0.2)
                    
                    if i == 11:
                        for j in binned_ints[i]:
                            ax13.plot(wave, j)
                        for j in binned_ints2[i]:
                            ax13.plot(wave2, j, alpha=0.2)
                            
                    if i == 12:
                        for j in binned_ints[i]:
                            ax14.plot(wave, j)
                        for j in binned_ints2[i]:
                            ax14.plot(wave2, j, alpha=0.2)
                            
                    if i == 13:
                        for j in binned_ints[i]:
                            ax15.plot(wave, j)
                        for j in binned_ints2[i]:
                            ax15.plot(wave2, j, alpha=0.2)
                            
                    if i == 14:
                        for j in binned_ints[i]:
                            ax16.plot(wave, j)
                        for j in binned_ints2[i]:
                            ax16.plot(wave2, j, alpha=0.2)
                            
                    if i == 15:
                        for j in binned_ints[i]:
                            ax17.plot(wave, j)
                        for j in binned_ints2[i]:
                            ax17.plot(wave2, j, alpha=0.2)
                    
                    if i == 16:
                        for j in binned_ints[i]:
                            ax18.plot(wave, j)
                        for j in binned_ints2[i]:
                            ax18.plot(wave2, j, alpha=0.2)
                            
                    if i == 17:
                        for j in binned_ints[i]:
                            ax19.plot(wave, j)
                        for j in binned_ints2[i]:
                            ax19.plot(wave2, j, alpha=0.2)
                            
                    if i == 18:
                        for j in binned_ints[i]:
                            ax20.plot(wave, j)
                        for j in binned_ints2[i]:
                            ax20.plot(wave2, j, alpha=0.2)

            
            plt.savefig(fname, dpi=600)
            plt.close()
    
################################################################  
class Radius:
    """Class that computes the radii of target stars using three different 
    methods:
    
    - general Stefan-Boltzmann law
    - gaia magnitudes with Ap bolometric correction
    - Ap absolute magnitudes with Ap bolometric correction.
    """
    
    def __init__(self, Teff, Teff_uncrt, BV, BV_uncrt, UB, UB_uncrt, P, P_uncrt,
                gmag, gmag_uncrt):
        """Initalises the Radius class.
        
        INPUTS:
            Teff = effective temperature (K)
            BV = B-V colour index (mag)
            BV_uncrt = B-V colour index uncertainty (mag)
            UB = U-B colour index (mag)
            UB_uncrt = U-B colour index uncertainty (mag)
            P = parallax (arcseconds)
            P_uncrt = parallax uncertainty (arcseconds)
            gmag = Gaia apparent magnitude (mag)
            gmag_uncrt = Gaia apparent magnitude uncertainty (mag)
            
        """
        
        self.Teff = Teff
        self.Teff_uncrt = Teff_uncrt
        self.BV = BV
        self.BV_uncrt = BV_uncrt
        self.UB = UB
        self.UB_uncrt = UB_uncrt
        self.P = P
        self.P_uncrt = P_uncrt
        self.gmag = gmag
        self.gmag_uncrt = gmag_uncrt
        
    def R_boltzmann(self, BC, BC_uncrt):
        """Calculates the radius (and its uncertainty) of the star using 
        Stefan-Boltzman law (assumingblack body).

        INPUTS:
            BC = bolometric correction from K.R. Lang textbook.
            BC_uncrt = bolometric correction uncertainty.

        RETURNS:
            R_array = array with first entry radius, and second entry radius
                      uncertainty.  
        """
        #Defining constants
        stef_bolt_const = 5.670374419e-8 #Units W m^2 K^-4
        
        #Using Gaia parallax to determine distance
        d_star = distance(self.P)
        d_star_uncrt = distance_uncrt(self.P, self.P_uncrt)
        
        
        #Calling 'gaia_V' function to calculate absolute magnitude from gaia 
        #apparent magnitudes
        V_abs = gaia_V(self.BV, self.P, self.gmag, d_star)
        V_abs_uncrt = gaia_V_uncrt(self.BV, self.BV_uncrt, self.P, self.P_uncrt, 
                                   self.gmag, self.gmag_uncrt, d_star, d_star_uncrt)
        
        
        #Determining bolometric magnitude
        V_abs_bol = bol_mag(BC, V_abs)
        V_abs_bol_uncrt = bol_mag_uncrt(BC_uncrt, V_abs_uncrt)
                
            
        #Calculating luminosity in watts
        L_star = luminosity(V_abs_bol)
        L_star_uncrt = luminosity_uncrt(V_abs_bol, V_abs_bol_uncrt)
        
        
        #Using Stefan-Boltzmann law to calculate the stellar radius, then converting
        #into solar radii from meters
        R_star = (np.sqrt(L_star/(4*np.pi*self.Teff**(4)*stef_bolt_const)))/6.95700e8
        R_star_uncrt = (np.sqrt(L_star_uncrt**(2)/(16*L_star*np.pi*stef_bolt_const*self.Teff**(4)) 
                               + (L_star * self.Teff_uncrt**(2))/(np.pi*stef_bolt_const*self.Teff**(6))))/6.95700e8
        
        
        #Storing R_star and R_star_uncrt into an array
        R_array = np.array([R_star, R_star_uncrt])
        
        return R_array

    def R_gaia(self):
        """Calculates the radius (and its uncertianty) of the star using Gaia
        magnitudes wihtin the special Ap stellar radius equation.
        
        INPUTS:
        
        RETURNS:
            R_array2 = array with first entry radius, and second entry radius
                      uncertainty.
        """
        
        #Using Gaia parallax to determine distance
        d_star = distance(self.P)
        d_star_uncrt = distance_uncrt(self.P, self.P_uncrt)
        
        
        #Calling 'gaia_V' function to calculate absolute magnitude from gaia 
        #apparent magnitudes
        V_abs = gaia_V(self.BV, self.P, self.gmag, d_star)
        V_abs_uncrt = gaia_V_uncrt(self.BV, self.BV_uncrt, self.P, self.P_uncrt, 
                                   self.gmag, self.gmag_uncrt, d_star, d_star_uncrt)
        
        
        #Determining bolometric correction using special Ap expression
        bc_Ap = Ap_bc(self.Teff)
        bc_Ap_uncrt = Ap_bc_uncrt(self.Teff, self.Teff_uncrt)
        
        
        #Using bc_Ap to determine bolometric magnitudes
        V_abs_bol2 = bol_mag(bc_Ap, V_abs)
        V_abs_bol_uncrt2 = bol_mag_uncrt(bc_Ap_uncrt, V_abs_uncrt)


        #Calculating radius of star using radii Ap expression
        R_star2 = 10**((42.31 - V_abs_bol2 - 10*np.log10(self.Teff))/5)
        R_star_uncrt2 = np.sqrt(((-np.log10(10)/5)*10**((42.31-V_abs_bol2-10*np.log10(self.Teff))/5))**2*V_abs_bol_uncrt2**2
                               + ((-2*np.log10(10)*10**((42.31-V_abs_bol2-10*np.log10(self.Teff))/5))/(self.Teff*np.log(10)))**2*self.Teff_uncrt**2)
        
        
        #Storing R_star and R_star_uncrt into an array
        R_array2 = np.array([R_star2, R_star_uncrt2])
        
        return R_array2

    def R_Ap_mag(self):
        """Calcultes the radius (and its uncertainty) of the star using a Ap-specific
        absolute magnitude expression that takes into account its unique flux 
        distributions. This will then be used with the Ap radii expression.

        INPUTS:

        RETURNS:
            R_array3 = array with first entry radius, and second entry radius
                       uncertainty.
        """
        
        #Calculating absolute visual magnitudes using the Ap-specific expression
        V_abs_ap = Ap_abs_mag(self.UB, self.BV)
        V_abs_ap_uncrt = Ap_abs_mag_uncrt(self.UB, self.UB_uncrt, self.BV, self.BV_uncrt)
        
        
        #Determining bolometric correction using special Ap expression
        bc_Ap = Ap_bc(self.Teff)
        bc_Ap_uncrt = Ap_bc_uncrt(self.Teff, self.Teff_uncrt)
        
        
        #Using bc_Ap to determine bolometric magnitudes
        V_abs_bol3 = bol_mag(bc_Ap, V_abs_ap)
        V_abs_bol_uncrt3 = bol_mag_uncrt(bc_Ap_uncrt, V_abs_ap_uncrt)
        
        
        #Calculating radius of star using radii Ap expression
        R_star3 = 10**((42.31 - V_abs_bol3 - 10*np.log10(self.Teff))/5)
        R_star_uncrt3 = np.sqrt(((-np.log10(10)/5)*10**((42.31-V_abs_bol3-10*np.log10(self.Teff))/5))**2*V_abs_bol_uncrt3**2
                               + ((-2*np.log10(10)*10**((42.31-V_abs_bol3-10*np.log10(self.Teff))/5))/(self.Teff*np.log(10)))**2*self.Teff_uncrt**2)
        
        
        #Storing R_star and R_star_uncrt into an array
        R_array3 = np.array([R_star3, R_star_uncrt3])
        
        
        return R_array3


################################################################  
class Magnetic:
    """Class that estimates the magnetic field strength of stars using the 
    methods outlined by Bailey (2014).
    """
    
    def __init__(self, wave, ints):
        """Initalises the Radius class.
        
        INPUTS:
            wave = wavelength array
            ints = intensity array
            
        """
        
        self.wave = wave
        self.ints = ints
        

    def fwhd(self, xlim, xmax, gauss_guess, line):
        """Calculates FWHD of a given region. The line being considered must be 
        close to being symmetric i.e. the line's peak must be in the middle, with
        equal distance to both the left and right sides.'
        
        INPUTS:
            xlim = minimum wavelength of region
            xmax = maximum wavelength of region
            gauss_guess = list of guesses for the Gaussian fitting
            line = string of line name (eg '4491')
            
        RETURNS:
            fwhd = full width at half depth of the line"""


        #Transposing the wavelength array 
        wave_trans = (self.wave).T
        
        #Defining the positions of x and y 
        p = np.where((wave_trans>=xlim) & (wave_trans<=xmax))

        #Fitting centroids using 'gauss' function, through Scipy module
        fit, cov = curve_fit(gauss,wave_trans[p],1-self.ints[p], 
        gauss_guess, maxfev=50000)

        #Using spline to fit [curve - peak/2], then finding the roots
        spline = UnivariateSpline(wave_trans[p], gauss(wave_trans[p],*fit)-np.max(gauss(wave_trans[p],*fit))/2)
        r1, r2 = spline.roots()
        
        
        #Obtaining centroids
        cent = fit[1]
        cent_uncrt = np.sqrt(np.diag(cov))[1]
        
        
        #Calculating FWHD by taking the difference of the spline fitting roots       
        fwhd = r2 - r1
        print("The FWHD for line {0}$\AA$ is: {1}$\AA$".format(line, fwhd))
        
        #Plotting results
        plt.figure()
        plt.plot(wave_trans[p], 1-self.ints[p], label="Observed Data")
        plt.plot(wave_trans[p],gauss(wave_trans[p],*fit), label="Gaussian Fit")
        plt.axvspan(r1, r2, facecolor='g', alpha=0.2)
        plt.vlines(cent, min(1-self.ints[p]), max(1-self.ints[p]),
                   color="black", linestyle=":", label="Centroid")
        plt.xlabel("Wavelength ($\AA{}$)")
        plt.ylabel("Intensity")
        # plt.title("Line {}$\AA{}$".format(line))
        plt.legend()
        plt.grid()
        plt.show()
        
        return fwhd
    
    def K(self, w_L, w_S, wave_L, wave_S, z_L, z_S):
        """Determining the wavenumber K in cm^-1.
        
        INPUTS:
            w_L = array of widths corresponding to large z values (angstroms)
            w_S = array of widths corresponding to small z values (angstroms)
            wave_L = array of wavelengths corresponding to large z values (angstroms)
            wave_S = array of wavelengths corresponding to small z values (angstroms)
            z_L = array of large z values
            z_S = array of small z values
            
        RETURNS:
            K = wavenumber (cm^-1)
        """
        #Converting widths W_L and w_S from A to cm
        w_L = w_L * 1e-8
        w_S = w_S * 1e-8

        #Converting wavelengths wave_L and wave_S from A to cm
        wave_L = wave_L * 1e-8
        wave_S = wave_S * 1e-8
        
        #Calculating K for every combination of lines
        #For each value in width_L, compute K for every value in width_S
        K_list = []
        
        for i in range(len(w_L)):
            j = 0
            while j <= len(w_S)-1:
                numerator = w_L[i]**2 - w_S[j]**2

                mean_L = np.mean(wave_L**4 * z_L**2)
                mean_S = np.mean(wave_S**4 * z_S**2)
                
                denominator = mean_L - mean_S

                K = np.sqrt((numerator/denominator))
                
                K_list.append(K)
                j += 1
        K_array = np.array(K_list) 
        # K_array = np.array([1.46470319, 1.96373179, 1.30759258])
        
        #Taking the average of K_list
        K_mean = np.mean(K_array)
        #Finding uncertainty in K_mean by finding largest spread of K values from mean
        K_uncrt = max(abs(K_array-K_mean))

        print("K = {0} ± {1} cm^-1 ".format(K_mean, K_uncrt))    
        
        return K_mean, K_uncrt
        
    def B(self, K, K_uncrt=None):
        """Estimates the magnetic field strength of the star. 
        
        INPUTS:
            K = wavenumber (cm^-1)
            
        RETURNS:
            magnet = magnetic field strength (kG)
            magnet_uncrt = magnetic field strength uncertainty (kG)
        """
        #Estimating magnetic field strength
        magnet = 8.11 * K
        
        #Calculating magnetic field strength uncertianty if K_uncrt is given
        if K_uncrt != None:
            magnet_uncrt = 8.11 * K_uncrt
            print("B = {0} ± {1} kG".format(magnet, magnet_uncrt)) 
            return magnet, magnet_uncrt
        
        else:
            print("B = {0} kG".format(magnet)) 
            return magnet
    
        

#-----------------OLD FUNCTIONS/CLASSES------------------------#
################################################################

################################################################
# def lomb_scarg(tt,yy,minP,maxP,line=None, name=None, plot=True):
#     """A function that computes the Lomb Scargle periodogram for a set of
#     data.

#     INPUTS:
#         tt = time (days)
#         yy = radial velocity data (km/s) or flux data (electrons / second)
#         minP = the minimum period range
#         maxP = the maximum period range
#         line = name of line in string format, used for title. Default = None
#         name = name of star in string format, used for title. Default = None 
#         plot = boolean response, where if True, a plot is made. Default = True.

#     RETURNS:
#         period = maximum peak value corresponding to the star's period 
#     """    
    
#     minP = minP 
#     maxP = maxP

#     #Periodgram asumes that the variation is y are about zero. This will shift
#     #your data so that this is true
#     y = yy -  np.mean(yy)

#     #sets the number of frequencies to calculate between Pmin and Pmax
#     frequencySamples = 10000 

#     #The frequencies converted to days^-1
#     omega = 2.0*np.pi*np.linspace(1.0/(maxP),
#                                   1.0/(minP),frequencySamples) 

#     #Because we will be calculating power values for many frequency values
#     #it is most efficient to do this using vectorized arrays rather than 
#     #loops. This line produces the vectroized arrays
#     dummy, omegaArray = np.meshgrid(tt,omega) 

#     #Calculates the delay constant
#     tau = (1.0/(2.0*omega))*(np.arctan(np.sum(np.sin(2.0*omegaArray*tt),axis=1)) 
#                               + np.arctan(np.sum(np.cos(2.0*omegaArray*tt),axis=1))) 

#     #Produces vectorized array of delay constants
#     dummy, tauArray = np.meshgrid(tt,tau)

#     #Compute the power spectrum
#     powerOmega = 0.5 * ((np.sum(y*np.cos(omegaArray*(tt - tauArray)),axis=1)**2/
#                       (np.sum(np.cos(omegaArray*(tt - tauArray))**2,axis=1))) + 
#                       (np.sum(y*np.sin(omegaArray*(tt - tauArray)),axis=1)**2/
#                       (np.sum(np.sin(omegaArray*(tt - tauArray))**2,axis=1))) )

#     #plot the spectrum
#     if plot == True:
#         plt.figure()
#         plt.plot(2.0*np.pi/omega,powerOmega)
#         plt.xlabel(r'Frequency (Days)')
#         plt.ylabel(r'Power')
#         if line != None:
#             plt.title("Lomb Scargle Periodogram for {}.".format(line))
#         elif name != None:
#             plt.title("Lomb Scargle Periodogram for {}.".format(name))
#         elif name != None and line != None:
#             plt.title("Lomb Scargle Periodogram for {0} at line {1}.".format(name, line))
#         else:
#             plt.title("Lomb Scargle Periodogram")
#         plt.show()
        
#         period = 2.0*np.pi/omega[np.argmax(powerOmega)]
#         print("Period = {} (days)".format(period))
    
#     #No plot outputted
#     if plot == False:
#         period = 2.0*np.pi/omega[np.argmax(powerOmega)]
#         print("Period = {} (days)".format(period))
        
#     return period

# ################################################################
# def lomb_scarg_window(tt,yy_1,yy_2,minP,maxP,line=None, name=None):
#     """A function that computes the Lomb Scargle periodogram  window for a set 
#     of data.

#     INPUTS:
#         tt = time (days)
#         yy = radial velocity data (km/s) or flux data (electrons / second)
#         yy_synth = the synthetic radial velocity/flux data, done by fitting sine
#                    model to data then extracting the pFit values into yy_synth. 
#         minP = the minimum period range
#         maxP = the maximum period range
#         line = name of line in string format, used for title. Default = None
#         name = name of star in string format, used for title. Default = None 

#     RETURNS:
#         period = maximum peak value corresponding to the star's period 
#     """    
#     minP = minP * 1440
#     maxP = maxP * 1440 
        
#     #Periodgram asumes that the variation is y are about zero. This will shift
#     #your data so that this is true
#     y_1 = yy_1 -  np.mean(yy_1)
#     y_2 = yy_2 -  np.mean(yy_2)
    
#     #sets the number of frequencies to calculate between Pmin and Pmax
#     frequencySamples = 10000 
    
#     #The frequencies converted to days^-1
#     omega = 2.0*np.pi*np.linspace(1.0/(maxP/24.0/60.0),
#                                   1.0/(minP/24.0/60.0),frequencySamples) 
    
#     #Because we will be calculating power values for many frequency values
#     #it is most efficient to do this using vectorized arrays rather than 
#     #loops. This line produces the vectroized arrays
#     dummy, omegaArray = np.meshgrid(tt,omega)

#     #Calculates the delay constant
#     tau = (1.0/(2.0*omega))*(np.arctan(np.sum(np.sin(2.0*omegaArray*tt),axis=1)) 
#                               + np.arctan(np.sum(np.cos(2.0*omegaArray*tt),axis=1))) 
    
#     #Produces vectorized array of delay constants
#     dummy, tauArray = np.meshgrid(tt,tau)
    
#     #Compute the power spectrum
#     powerOmega_1 = 0.5 * ((np.sum(y_1*np.cos(omegaArray*(tt - tauArray)),axis=1)**2/
#                       (np.sum(np.cos(omegaArray*(tt - tauArray))**2,axis=1))) + 
#                       (np.sum(y_1*np.sin(omegaArray*(tt - tauArray)),axis=1)**2/
#                       (np.sum(np.sin(omegaArray*(tt - tauArray))**2,axis=1))) )
#     powerOmega_2 = 0.5 * ((np.sum(y_2*np.cos(omegaArray*(tt - tauArray)),axis=1)**2/
#                       (np.sum(np.cos(omegaArray*(tt - tauArray))**2,axis=1))) + 
#                       (np.sum(y_2*np.sin(omegaArray*(tt - tauArray)),axis=1)**2/
#                       (np.sum(np.sin(omegaArray*(tt - tauArray))**2,axis=1))) )

#     #plot the spectrum
#     plt.figure()
#     plt.grid()
#     plt.plot(2.0*np.pi/omega,powerOmega_1, label="Observed")
#     plt.plot(2.0*np.pi/omega,powerOmega_2, label="Window")
#     plt.xlabel(r'Period (Days)')
#     plt.ylabel(r'Power')
#     plt.legend()
#     # if line != None:
#     #     plt.title("Lomb Scargle Periodogram for {}.".format(line))
#     # elif name != None:
#     #     plt.title("Lomb Scargle Periodogram for {}.".format(name))
#     # elif name != None and line != None:
#     #     plt.title("Lomb Scargle Periodogram for {0} at line {1}.".format(name, line))
#     # else:
#     #     plt.title("Lomb Scargle Periodogram")
#     plt.show()
    
#     # period = 2.0*np.pi/omega[np.argmax(powerOmega)]


# class GMag_to_VMag:
#     """Class that converts Gaia magnitudes into visual."""
        
#     def __init__(self, TBMag, TBMag_uncrt, TVMag, TVMag_uncrt, GMag, GMag_uncrt):
#         """Initialises the GMag_to_VMag class.
        
#         INPUTS:
#             TBMag = Tycho B magnitude
#             TBMag_uncrt = Tycho B magnitude uncertainty
#             TVMag = Tycho V magnitude
#             TVMag_uncrt = Tycho V magnitude uncertianty
#         """
        
#         self.TBMag = TBMag
#         self.TBMag_uncrt = TBMag_uncrt
#         self.TVMag = TVMag
#         self.TVMag_uncrt = TVMag_uncrt
#         self.GMag = GMag
#         self.GMag_uncrt = GMag_uncrt
        

#     def colour_excess(self):
#         """Using Tycho B and Tycho V magnitudes to approximate (B-V) Johnson's 
#         colour excess.
        
#         INPUTS:
            
#         RETURNS:
#             B_V_array = array with the first entry being the (B-V) value, and the
#                         second being its uncertainty.
#             """
        
#         #Calculating the (B-V) Johnson's colour excess and corresponding uncertainty
#         B_V = 0.850 * (self.TBMag - self.TVMag) #BV = B-V
#         B_V_uncrt = np.sqrt((0.850**2 * self.TBMag_uncrt**2) + 
#                              ((-0.850)**2 * self.TVMag_uncrt**2))
        
#         #Storing (B-V) and its uncertainty into an array
#         B_V_array = np.array([B_V, B_V_uncrt])

#         return B_V_array

#     def Gmag_to_mv(self, B_V_array):
#         """Using (B-V) colour excess to determine m_v magnitude.
        
#         INPUTS:
#             B_V_array = array with the first entry being the (B-V) value, and the
#                         second being its uncertainty. Calculated using the
#                         colour_excess function.
                        
#         RETURNS:
#             """
        
#         #Coefficients for polynomial expression
#         a = -0.04749
#         b = -0.0124
#         c = -0.2901
#         d = 0.02008

#         #polynomial expression to convert from g_mag to m_mag
#         m_v = self.GMag - (a + b*(B_V_array[0]) + c*(B_V_array[0])**2 + d*(B_V_array[0])**3)
#         m_v_uncert = np.sqrt((self.GMag_uncrt**2) + ((b + (2*c)*B_V_array[0] + (3*d)*B_V_array[0]**2)**2
#                                                  * B_V_array[1]**2))
        
#         #Storing the apparent magnitude and its uncertainty into an array
#         m_v_array = np.array([m_v, m_v_uncert])
        
#         return m_v_array
    
#     def distance(self, P, P_uncrt):
#         """Computes distance to star in pc using Gaia parallax measurements.
        
#         INPUTS:
#             p = Gaia parallax (mas)
#             P_uncrt = Gaia parallax uncertainty (mas)
            
#         RETURNS:
#             d_array = array with the first entry being the distance to the star,
#                       and the second being its uncertainty.
#         """
        
#         #Converts units 'mas' into 'arcsec'
#         mas_to_arcsec = P * 0.00099999995874704 
#         mas_to_arcsec_uncert = P_uncrt * 0.00099999995874704
        
#         #Computing distance to star (and uncertainty) using the above parallax
#         #values in arcseconds
#         d = 1 / mas_to_arcsec 
#         d_uncrt = mas_to_arcsec_uncert / mas_to_arcsec**2 #Working in book  
        
#         #Storing the distance and distance uncertianty into an array
#         d_array = np.array([d, d_uncrt])

#         return d_array
    
#     def abs_magnitude(self, m_v_array, d_array, BC):
#         """Calculates the absolute bolometric magnitude of the star.
        
#         INPUTS:
#             m_v_array = array with the first entry being apparent magnitude, and
#                         the second being its uncertainty. Calculated using the
#                         Gmag_to_mv function.
#             d_array = array with the first entry being the distance to the star,
#                       and the second being its uncertainty. Calculated using the 
#                       distance function.
                      
#         RETURNS:
        
#         """
        
#         #Computing absolute magnitude of the star
#         abs_mag = m_v_array[0] - 5 * np.log10(d_array[0]) + 5 
#         abs_mag_uncert = np.sqrt(m_v_array[1]**2 + (-5/(np.log(10)*d_array[0]))**2
#                                  * d_array[1]**2)
        
#         #Calculating the absolute bolometric magnitude of the star
#         abs_BC_mag = BC - abs_mag
#         abs_BC_mag_uncrt = abs_mag_uncert
        
#         #Storing absolute bolometric magnitude and uncertianty into an array
#         abs_BC_mag_array = np.array([abs_BC_mag, abs_BC_mag_uncrt])
        
#         return abs_BC_mag_array
    