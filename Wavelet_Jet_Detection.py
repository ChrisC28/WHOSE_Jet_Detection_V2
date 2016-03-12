#========================================================================#
# Wave_Jet_Detection
# This class implements a simple 1D version of the wavelet edge detection 
# method described in Chapman 2014 
#
#========================================================================#

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, uniform, kurtosis, rankdata
#import PDF_Tools as PDF_Tools
#from scipy.signal import detrend
import pywt
import math
import scipy.signal
from scipy import interpolate
import peakutils

from geopy.distance import vincenty

#import iswt as invSWT
#from scipy.cluster.vq import kmeans2
#from scipy.interpolate import interp1d
#from scipy.signal import lfilter, firwin, filtfilt, freqz, convolve
#import Pycluster as pc
#import scipy.ndimage.filters as filters
#import scipy.ndimage as ndimage


class Jet_Detector:


    #Basic constructor 

    def __init__(self,wavelet_basis,n_deomp_levels,confidence_param):
    
       self.wavelet_basis = 'haar'
       self.wavelet_instance = pywt.Wavelet(self.wavelet_basis) 
       self.n_decomp_levels  = n_deomp_levels     
       self.confidence_param = confidence_param

    def detect_jets(self,lon_points,lat_points,dynamic_topo):
        #Remove nans from the track
        dynamic_topo = self.remove_nans(dynamic_topo)
        lon_points   = self.remove_nans(lon_points)
        lat_points   = self.remove_nans(lat_points)
        
        #Step 0: Test to determine if the track crosses 360 line
        n_points = len(lon_points)
        
        #delta_lon = np.diff(lon_points)
        #if np.any(np.abs(delta_lon)>300):
            
        #    crossing_index =np.nonzero(np.abs(delta_lon)>300)[0]
        #    delta_lon[crossing_index[0]] = np.nan
        #    #print np.nanmean(delta_lon)
        #    if np.nanmean(delta_lon)<0:
        #        lon_points[0:crossing_index[0]+1]= lon_points[0:crossing_index[0]+1]+360.0
        #    else:
        #        lon_points[crossing_index[0]+1:n_points]= lon_points[crossing_index[0]+1:n_points]+360.0
            
        
        #Step 1: Interpolate the along-track data onto a regularly spaced grid
        if self.is_odd(n_points):
            n_points = n_points-1
        n_points = self.nearest_power_of_two(n_points)
        
        south_index = 10
        north_index =  -10
        
        lat_grid = np.linspace(lat_points.min(),lat_points.max(),n_points)
        lon_grid = np.linspace(lon_points.min(),lon_points.max(),n_points)

        f = interpolate.interp1d(lat_points, dynamic_topo)   
        dynamic_topo_grid = f(lat_grid)
        
        #dynamic_topo_grid = np.interp(lon_grid, lon_points,dynamic_topo)
        
        if dynamic_topo_grid.max() <0:
            dynamic_topo_grid = dynamic_topo_grid-(dynamic_topo_grid.max()-0.1)
        baselines = peakutils.baseline(dynamic_topo_grid, 2.0)
        dynamic_topo_detrend = dynamic_topo_grid -baselines
        #dynamic_topo_detrend = scipy.signal.detrend(dynamic_topo_grid)
        
        #Step 2: Wavelet Decomposition
        wavelet_coeffs = np.asarray(pywt.swt(dynamic_topo_detrend, self.wavelet_instance,self.n_decomp_levels,start_level=0))
        
        
        #Step 3: Kurtosis Thresholding 
        wavelet_coeffs = self.Kurtosis_Thresholding(wavelet_coeffs,south_index,north_index)    
        
    
        #Step 4: Wavelet de-noising 
        wavelet_coeffs=self.Wavelet_Denoising(wavelet_coeffs,south_index,north_index)
        
        #Step 5: Signal Reconstruction
        denoised_signal  = self.iswt(wavelet_coeffs, self.wavelet_instance)
        
        #Step 6: Simple moving average filter to get rid of the occasional high 
        #frequency crap that appears due to wavelet reconstruction
        denoised_signal = (denoised_signal[0:n_points-2]+denoised_signal[1:n_points-1]+denoised_signal[2:n_points])/3.0
        n_points = denoised_signal.size
        denoised_signal = (denoised_signal[0:n_points-2]+denoised_signal[1:n_points-1]+denoised_signal[2:n_points])/3.0
        n_points = denoised_signal.size
        denoised_signal = (denoised_signal[0:n_points-2]+denoised_signal[1:n_points-1]+denoised_signal[2:n_points])/3.0

        
        #Step 7: Maxima Detection
        indicies = peakutils.indexes(np.abs(np.diff(denoised_signal)), thres = 0.20, min_dist=5)
    
        indicies = [index_element+3 for index_element in indicies]
        
        #Step 8: Rejection of negative velocities
        indicies_to_remove = []
        
        for i_jet in range(0,len(indicies)):
            if indicies[i_jet]>=len(denoised_signal)-1:
                indicies_to_remove.append(i_jet)
            elif np.diff(denoised_signal)[indicies[i_jet]-1]<0:
                indicies_to_remove.append(i_jet)
        
        for i in sorted(indicies_to_remove, reverse=True):
            del indicies[i]
        
        jet_lon_positions = lon_grid[indicies]
        jet_lat_positions = lat_grid[indicies]
        
        jet_lon_positions[jet_lon_positions>360] = jet_lon_positions[jet_lon_positions>360]-360.0 
        
        return jet_lon_positions, jet_lat_positions
        
    def Kurtosis_Thresholding(self,wavelet_coefficients,south_index,north_index):
        
        #Implement the Kurtosis thresholding to separate the "signal" from the
        #"noise" by looking for non gaussian features
        chebychev_thresh = np.sqrt(24.0/wavelet_coefficients[0][1][south_index:north_index].shape[0]) / np.sqrt(1-self.confidence_param)
        
        for iLevel in range(0,len(wavelet_coefficients)):
            if np.abs( kurtosis(wavelet_coefficients[iLevel][1][south_index:north_index],fisher=True, bias=True) ) < chebychev_thresh:
	        wavelet_coefficients[iLevel][1][:] = np.zeros(wavelet_coefficients[iLevel][1].shape,dtype='float64')
        return wavelet_coefficients

    def Wavelet_Denoising(self,wavelet_coefficients,southIndex,northIndex):

        for iLevel in range(0,len(wavelet_coefficients)):
            if not np.array_equal(wavelet_coefficients[iLevel][1], np.zeros(wavelet_coefficients[iLevel][1].shape[0],dtype='float64')):
                #Denoising threshold determined from Donoho and Johnson 
                thresholdValue = 2.0*( np.median(np.abs(wavelet_coefficients[iLevel][1][southIndex:northIndex])) / 0.6745) * np.sqrt(2.0*np.log10(wavelet_coefficients[iLevel][1][southIndex:northIndex].shape[0]) )
            
                for iY in range(0,wavelet_coefficients[iLevel][1].shape[0]):
                    if np.abs(wavelet_coefficients[iLevel][1][iY]) <= thresholdValue:
                        #print 'thresholding'
                        wavelet_coefficients[iLevel][1][iY] = 0.0
	
        return wavelet_coefficients


    def iswt(self, coefficients, wavelet):
        """
        M. G. Marino to complement pyWavelets' swt.
        Input parameters:

        coefficients
        approx and detail coefficients, arranged in level value 
        exactly as output from swt:
        e.g. [(cA1, cD1), (cA2, cD2), ..., (cAn, cDn)]

        wavelet
          Either the name of a wavelet or a Wavelet object

        """
        output = coefficients[0][0].copy() # Avoid modification of input data

        #num_levels, equivalent to the decomposition level, n
        num_levels = len(coefficients)
        for j in range(num_levels,0,-1): 
            step_size = int(math.pow(2, j-1))
            last_index = step_size
            _, cD = coefficients[num_levels - j]
            for first in range(last_index): # 0 to last_index - 1

                # Getting the indices that we will transform 
                indices = np.arange(first, len(cD), step_size)

                # select the even indices
                even_indices = indices[0::2] 
                # select the odd indices
                odd_indices = indices[1::2]

                # perform the inverse dwt on the selected indices,
                # making sure to use periodic boundary conditions
                x1 = pywt.idwt(output[even_indices], cD[even_indices], wavelet, 'per') 
                x2 = pywt.idwt(output[odd_indices], cD[odd_indices], wavelet, 'per')

                # perform a circular shift right
                x2 = np.roll(x2, 1)

                # average and insert into the correct indices
                output[indices] = (x1 + x2)/2.  

        return output

	
    def adaptive_extrema_finder(self,input_signal,neighbourhood_size,threshold):
	
        #Determine the local maxima in a given neighbourhood using a maximum filter
        #Filter determines the local max in a neighbourhood of size = neighbourhoodSize\
        #and replaces all values in that region by the maximum
        import scipy.ndimage.filters as filters

        maximum_filter_output    = filters.maximum_filter(input_signal, neighbourhood_size)

	
        #Find the maxima points. Returns a Boolean array of the points where the data is 
        #equal to the local maxima
        maxima_points = (input_signal == maximum_filter_output)
	
	
        #Determine the (weighted) average of values in the same neighbourhood to
        #give an idea of the background values. Sigma is determined to cover 99% of the spread
	
        sigma = (neighbourhood_size - 1) / 6.0
        background_filter_output = filters.gaussian_filter(input_signal, sigma)#, output=None, mode='reflect', cval=0.0)
	
        threshold_value = threshold*np.nanstd(input_signal)
	
        threshold_output = ( input_signal > threshold_value)
        maxima_points[np.logical_not(threshold_output)] = 0
	
        return maxima_points



    def is_odd(self,num):
        
        return num & 0x1  

    def nearest_power_of_two(self,x):
        return 1<<(x-1).bit_length()
        
    def remove_nans(self,input_array):
        return input_array[~np.isnan(input_array)]
