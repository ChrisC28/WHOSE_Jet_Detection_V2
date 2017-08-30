# -*- coding: utf-8 -*-
#========================================================================#
# Wave_Jet_Detection
# This class implements the1D version of the wavelet edge detection 
# method described in Chapman 2014 and used in Chapman 2017
#
# The class consists of a basic constructor that instantiates the class
# and a series of methods called from a the wrapper detect_jets that 
# that implement the jet detection methodology.
# 
# This program requires numpy, scipy and the pywavelets package
#
# The algorithm has been tested on both gridded data from AVISO, as well 
# as the along-track data, but when using the along track data, one must 
# be carefull near the poles, as the satellite tracks tend to become very
# zonally oriented. 
#========================================================================#

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import kurtosis
import pywt
import math
from scipy import interpolate


class Jet_Detector:
    
    
    #Basic class constructor 

    def __init__(self,n_deomp_levels=4,confidence_param=0.9,wavelet_basis='haar',grad_thresh=0.1):
    
       """Constructor for the wavelet jet detection class
       | This constructor instantiates the wavelet jet detection object
       | and initialises the wavelet basis, and other parmeters such as the 
       | number of decompostion levels. 
       |
       | Methods within this class implement the WHOSE jet detection method
       | 
       | Parameters
       | ----------
       | n_decomp_levels : integer
       |                   number of decompostion levels to use in the 
       |                   wavelet decomposition
       | confidence_param: float between 0 and 1 
       |                   the confidence parameter for the Bienaymé-
       |                   Chebyshev inequality
       |   
       | wavelet_basis: string
       |                the basis function to use for the discrete wavelet 
       |                transform (Haar is recomended)
       |                                   
       | grad_thresh:  float 
       |               The gradient threshold value for the eventual peak 
       |               detection
       |
       |  Returns
       |  -------
       |  self: instantiated Jet_Detection class 
       """
        
    
       self.wavelet_basis = wavelet_basis  #Initialise the wavelet transform
                                           #Default wavelet basis is the Haar 
                                           #wavelet, which is a particularly 
                                           #good choice when searching for 
                                           #step like features
                                            
       self.wavelet_instance = pywt.Wavelet(self.wavelet_basis)  
       self.n_decomp_levels  = n_deomp_levels      #number of wavelet decomposition levels   
       self.confidence_param = confidence_param    #The confidence parametre for the Bienaym-Chebyshev inequality 
                                                   # (Eqn 4 in Chapman 2014) 
       self.grad_thres       = grad_thresh         #



    def detect_jets(self,lon_points,lat_points,dynamic_topo,only_eastward=True):
    
        """ PUBLIC: detect_jets
        |
        |   This method implements the basic WHOSE jet detection methodology  
        |   described in Chapman (2014). The methodology consists of a filtering 
        |   step, designed to remove the noise and eddy like features, followed by
        |   a gradients threshold detection step. The full reference:
        |
        |   Chapman, C. C. (2014), Southern Ocean jets and how to find them: 
        |   Improving and comparing common jet detection methods, 
        |   J. Geophys. Res. Oceans, 119, 4318–4339, doi:10.1002/2014JC009810.
        |      
        |   Parameters
        |   ----------
        |
        |   lon_points: (n_points,) array_like 
        |               the longitude along the satellite ground track. In the 
        |               case of a meridional transect this parameter should be 
        |               by a 1d array with the same dimensions as the latitude
        |               array 
        |   lat_points: (n_points,) array_like
        |               the latitude along the satellite ground track. In the 
        |               case of a zonal transect this parameter should be 
        |               by a 1d array with the same dimensions as the longitude
        |               array 
        |        
        |   dynamic_topo: (n_points,) array_like
        |               A  1D array of the absolute dynamic topography along the 
        |               the ground track. The values must be absolute 
        |               (not anomalies) 
        |  
        |   Key Word Parameters 
        |   -------------------
        |   only_eastward: boolean, default True
        |                  Filters out westward jets. Usefull for those interested
        |                  in ACC jets or western boundary current extentions that 
        |                  have a primarily eastward character. 
        |  Returns 
        |  -------- 
        |  jet_lon_position_orig_grid: (n_jets,) array_like
        |                  Array of the longitudes of the detected jet locations
        |  jet_lat_position_orig_grid: (n_jets,) array_like
        |                  Array of the longitudes of the detected jet locations               
        |                      
        """
        
        
        EARTH_RADIUS = 6380.0
        DEG_2_RAD    = np.pi/180.0
        RAD_2_DEG    = 1.0/DEG_2_RAD

        #====================================================================#
        #First, we take care of NaNs (due to coastlines or bad data or whatever)
        #using a basic nan infilling routine: Regions of NaNs are infilled using  
        #linear interpolation. NaN infilling is purely to enable to wavelet 
        #decomposition to work and any jets detected in these regions are 
        #filtered out later.
        #====================================================================#   
        infilled_dynamic_topo, nan_mask = self.__treat_nans(dynamic_topo,lat_points)
        
        #Is the transect (almost) all NaNs? If yes, terminate here
        if nan_mask.sum() >= len(infilled_dynamic_topo)-5:
            return [],[]
        
        #==========================================================#
        #Step 0: Test to determine if the track crosses 360 line
        #If the track crosses, wrap it around.
        #==========================================================#
        n_points = len(lon_points)
        
        delta_lon = np.diff(lon_points)
        if np.any(np.abs(delta_lon)>300):
            
            crossing_index =np.nonzero(np.abs(delta_lon)>300)[0]
            delta_lon[crossing_index[0]] = np.nan
            if np.nanmean(delta_lon)<0:
                lon_points[0:crossing_index[0]+1]= lon_points[0:crossing_index[0]+1]+360.0
            else:
                lon_points[crossing_index[0]+1:n_points]= lon_points[crossing_index[0]+1:n_points]+360.0
            
        #====================================================================#
        # Step 1: Interpolate the along-track data onto a regularly spaced grid
        # with the number of grid points a power of 2 (nessecary for the 
        # wavelet decomposition to work). Along track data is routinely 
        #====================================================================# 
        
        #Find the nearest power of two.  
        if self.__is_odd(n_points):
            n_points = n_points-1
        
        n_points = self.__nearest_power_of_two(n_points)
        
        #Interpolate to a regular 
        south_index = 10
        north_index =  -10
        lat_grid = np.linspace(lat_points.min(),lat_points.max(),n_points)    #Set up the new lat grid 
        lon_grid = np.linspace(lon_points.min(),lon_points.max(),n_points)    #Set up the new lon grid 
       
        f = interpolate.interp1d(lat_points, infilled_dynamic_topo)           #Interpolate to the new grid
        dynamic_topo_grid = f(lat_grid)
        
        
        #====================================================================#
        #Step 2: Wavelet Decomposition
        #        The methodology uses wavelets to effectively remove noise
        #        See section 2.2 and Fig. 3b in Chapman (2014)
        #====================================================================#
        wavelet_coeffs = np.asarray(pywt.swt(dynamic_topo_grid, self.wavelet_instance,
                                    self.n_decomp_levels,start_level=0))
        
        #====================================================================#    
        #Step 3: Kurtosis Thresholding 
        #        The fourth statistical moment (Kurtosis - Eqn. (1) in 
        #        Chapman 2014) is used to separate noise and eddies
        #        (assumed to be normally distributed) from step features
        #        (assumed non-normally distributed). To do this, the 
        #        Bienaymé-Chebyshev inequality (Eqn. 4 in Chapman 2014) is 
        #        applied. This thresholding requires the specification of the 
        #        confidence parameter alpha that is between 0 and 1. An alpha
        #        of 0.9 indicates a 90% probability that the process is 
        #        normally distributed.  
        #        See Sections 2.1 and Fig 3c of Chapman (2014) for detail
        #====================================================================#
        
        wavelet_coeffs = self.__Kurtosis_Thresholding(wavelet_coeffs,south_index,north_index)    
        #====================================================================#
        #Step 4: Wavelet de-noising 
        #        Simple Wavelet shrinkage/denoising of Donoho & Johnstone (1994)
        #        is applied to further improve the signal-to-noise. 
        #        See Section 2.2 and Fig. 3d of Chapman (2014) for detail.  
        #====================================================================#
        wavelet_coeffs=self.__Wavelet_Denoising(wavelet_coeffs,south_index,north_index)
        
        #====================================================================#
        #Step 5: Signal Reconstruction
        #        The inverse wavelet transform of the denoised wavelet coeffs  
        #        is performed to reconstruct the denoised signal. 
        #====================================================================#
        denoised_signal  = self.__iswt(wavelet_coeffs, self.wavelet_instance)
                
        #====================================================================#                        
        #Step 5 1/2: Simple moving average filter to get rid of the occasional high 
        #frequency crap that appears due to wavelet reconstruction
        #====================================================================#
        
        denoised_signal = (denoised_signal[0:n_points-2]+denoised_signal[1:n_points-1]+denoised_signal[2:n_points])/3.0
        n_points = denoised_signal.size
        denoised_signal = (denoised_signal[0:n_points-2]+denoised_signal[1:n_points-1]+denoised_signal[2:n_points])/3.0
        n_points = denoised_signal.size
        denoised_signal = (denoised_signal[0:n_points-2]+denoised_signal[1:n_points-1]+denoised_signal[2:n_points])/3.0
        
        #====================================================================#
        # Step 6: Calculate the gradient (in m/km)
        #         As the WHOSE method relies on gradient detection, we calculate
        #         the central difference gradient here in order to estimate it.
        #         Note that the gradient is calculated in physical distance
        #         not per degree.
        #====================================================================#
        
        grad_denoised = np.gradient(denoised_signal,EARTH_RADIUS*DEG_2_RAD*(lat_grid[1]-lat_grid[0]))
        
        #====================================================================#
        #Step 7: Maxima Detection
        #        We use the detect_peaks function, written by Marcos Duarte, 
        # (https://github.com/demotu/BMC) to calculate the peaks. Changed
        # from the peak_utils toolbox, because I honestly couldn't work out
        # how it was calculating its thresholds. This function is cleaner and
        # includes thresholding operations.  
        #====================================================================#
        
        indicies = self.__detect_peaks(np.abs(grad_denoised), mph=self.grad_thres, mpd=5, threshold=0.01*self.grad_thres, 
                                        edge='rising', kpsh=False, valley=False)
        
        
        indicies = [index_element+3 for index_element in indicies]
        
        #====================================================================#
        #Step 8: Rejection of negative velocities
        #        Needs flag "only eastwards" 
        #====================================================================#
        
        if only_eastward:
            indicies_to_remove = []
            for i_jet in range(0,len(indicies)):
                if indicies[i_jet]>=len(denoised_signal)-1:
                    indicies_to_remove.append(i_jet)
                
                elif -np.sign(np.nanmean(lat_grid))*np.diff(denoised_signal)[indicies[i_jet]-1]<0:
                    indicies_to_remove.append(i_jet)
        
            for i in sorted(indicies_to_remove, reverse=True):
                del indicies[i]
        
                
        jet_lon_positions = lon_grid[indicies]
        jet_lat_positions = lat_grid[indicies]
        
        jet_lat_position_orig_grid = np.zeros_like(jet_lat_positions)
        jet_lon_position_orig_grid = np.zeros_like(jet_lon_positions)

        #Finally, we put the jet locations back on the original grid and
        #screen out any detected jets in masked regions. 
        
        for i_jet in range(0,len(jet_lat_positions)):
            idx = np.nonzero(lat_points>=jet_lat_positions[i_jet])[0][0]
            
            if not(nan_mask[idx]):
                jet_lat_position_orig_grid[i_jet] = lat_points[idx]
            idx = np.nonzero(lon_points>=jet_lon_positions[i_jet])[0][0]
            
            if not(nan_mask[idx]):
                jet_lon_position_orig_grid[i_jet] = lon_points[idx]
        
             
        
        #Finally, return the jet latitude and longitudes
        return jet_lon_position_orig_grid, jet_lat_position_orig_grid
        
    def __Kurtosis_Thresholding(self,wavelet_coefficients,south_index,north_index):
        
        '''
        PRIVATE: Kurtosis_Thresholding
        Implement the Kurtosis thresholding to separate the signal from the
        noise by looking for non gaussian features. 
        '''
        
        chebychev_thresh = np.sqrt(24.0/wavelet_coefficients[0][1][south_index:north_index].shape[0]) / np.sqrt(1-self.confidence_param)
        
        for iLevel in range(0,len(wavelet_coefficients)):
            if np.abs( kurtosis(wavelet_coefficients[iLevel][1][south_index:north_index],fisher=True, bias=True) ) < chebychev_thresh:
	        wavelet_coefficients[iLevel][1][:] = np.zeros(wavelet_coefficients[iLevel][1].shape,dtype='float64')
        return wavelet_coefficients

    def __Wavelet_Denoising(self,wavelet_coefficients,southIndex,northIndex):

        for iLevel in range(0,len(wavelet_coefficients)):
            if not np.array_equal(wavelet_coefficients[iLevel][1], np.zeros(wavelet_coefficients[iLevel][1].shape[0],dtype='float64')):
                #Denoising threshold determined from Donoho and Johnson 
                thresholdValue = ( np.median(np.abs(wavelet_coefficients[iLevel][1][southIndex:northIndex])) / 0.6745) * np.sqrt(2.0*np.log10(wavelet_coefficients[iLevel][1][southIndex:northIndex].shape[0]) )
            
                for iY in range(0,wavelet_coefficients[iLevel][1].shape[0]):
                    if np.abs(wavelet_coefficients[iLevel][1][iY]) <= thresholdValue:
                        #print 'thresholding'
                        wavelet_coefficients[iLevel][1][iY] = 0.0
	
        return wavelet_coefficients


    def __iswt(self, coefficients, wavelet):
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

    def __treat_nans(self,input_transect,y_grid):
        
        
        #Find NANs in the transect and check if it is all/mostly nans
        #if yes return an error. 
        
        transect_infilled = input_transect.copy()
        nan_mask          = np.isnan(input_transect)
        
        n_nan_points = nan_mask.sum()
        if n_nan_points == len(input_transect):
            return transect_infilled, nan_mask
        
        #Break up the nans into continuous chunks separated by good data

        start_nan_idx = np.nonzero(np.diff(1.0*nan_mask)==1)[0]
        end_nan_idx   = np.nonzero(np.diff(1.0*nan_mask)==-1)[0]
        
        n_start = start_nan_idx.size
        n_end   = end_nan_idx.size
    
        if n_start == n_end:
            for i_nan_zone in range(0,n_start):
                start_value = input_transect[start_nan_idx[i_nan_zone]]
                end_value   = input_transect[end_nan_idx[i_nan_zone]+1]
                
                start_y = y_grid[start_nan_idx[i_nan_zone]]
                end_y   = y_grid[end_nan_idx[i_nan_zone]+1]
                 
                slope = (end_value-start_value)/(end_y-start_y)
                infill_values = slope * (y_grid[start_nan_idx[i_nan_zone]+1:end_nan_idx[i_nan_zone]+1]-y_grid[start_nan_idx[i_nan_zone]-1]) + start_value 
                
                transect_infilled[start_nan_idx[i_nan_zone]+1:end_nan_idx[i_nan_zone]+1] = infill_values
            
        elif n_end==n_start-1:
            for i_nan_zone in range(0,n_start-1):
                
                start_value = input_transect[start_nan_idx[i_nan_zone]]
                end_value   = input_transect[end_nan_idx[i_nan_zone]+1]
                
                start_y = y_grid[start_nan_idx[i_nan_zone]]
                end_y   = y_grid[end_nan_idx[i_nan_zone]+1]
                 
                slope = (end_value-start_value)/(end_y-start_y)
                infill_values = slope * (y_grid[start_nan_idx[i_nan_zone]+1:end_nan_idx[i_nan_zone]+1]-y_grid[start_nan_idx[i_nan_zone]-1]) + start_value 
                
                transect_infilled[start_nan_idx[i_nan_zone]+1:end_nan_idx[i_nan_zone]+1] = infill_values
            infill_values = input_transect[start_nan_idx[-1]]   
            transect_infilled[start_nan_idx[-1]+1::] = infill_values
        
        elif n_end-1==n_start:
                 
            end_value = input_transect[end_nan_idx[0]]
            transect_infilled[0:end_nan_idx[0]] = end_value
            
            for i_nan_zone in range(0,n_start):
                start_value = input_transect[start_nan_idx[i_nan_zone]]
                end_value   = input_transect[end_nan_idx[i_nan_zone+1]+1]
                
                start_y = y_grid[start_nan_idx[i_nan_zone]]
                end_y   = y_grid[end_nan_idx[i_nan_zone+1]+1]
                 
                slope = (end_value-start_value)/(end_y-start_y)
                infill_values = slope * (y_grid[start_nan_idx[i_nan_zone]+1:end_nan_idx[i_nan_zone+1]+1]-y_grid[start_nan_idx[i_nan_zone]-1]) + start_value 
                
                transect_infilled[start_nan_idx[i_nan_zone]+1:end_nan_idx[i_nan_zone+1]+1] = infill_values
               
        else:
            print 'Error' 
        
        return transect_infilled, nan_mask
        
         
        
        
        #Infill the nans with linear interpolation between the end points or 
        #just a single constant value
        
        #return the infilled transect with a nan mask to later remove any jets
        #found at those indices. 

    def __is_odd(self,num):
        
        return num & 0x1  

    def __nearest_power_of_two(self,x):
        return 1<<(x-1).bit_length()
        
    def __remove_nans(self,input_array):
        return input_array[~np.isnan(input_array)]
    
    def __detect_peaks(self,x, mph=None, mpd=1, threshold=0, edge='rising',
                     kpsh=False, valley=False):

        """Detect peaks in data based on their amplitude and other features.

        Parameters
        ----------
        x : 1D array_like
            data.
        mph : {None, number}, optional (default = None)
            detect peaks that are greater than minimum peak height.
        mpd : positive integer, optional (default = 1)
            detect peaks that are at least separated by minimum peak distance (in
            number of data).
        threshold : positive number, optional (default = 0)
            detect peaks (valleys) that are greater (smaller) than `threshold`
            in relation to their immediate neighbors.
        edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
            for a flat peak, keep only the rising edge ('rising'), only the
            falling edge ('falling'), both edges ('both'), or don't detect a
            flat peak (None).
        kpsh : bool, optional (default = False)
               keep peaks with same height even if they are closer than `mpd`.
        valley : bool, optional (default = False)
            if True (1), detect valleys (local minima) instead of peaks.
        
        Returns
        -------
        ind : 1D array_like
            indeces of the peaks in `x`.

        Notes
        -----
        The detection of valleys instead of peaks is performed internally by simply
        negating the data: `ind_valleys = detect_peaks(-x)`
    
        The function can handle NaN's 

        See this IPython Notebook [1]_.

        References
        ----------
        .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

        Examples
        --------
        >>> from detect_peaks import detect_peaks
        >>> x = np.random.randn(100)
        >>> x[60:81] = np.nan
        >>> # detect all peaks and plot data
        >>> ind = detect_peaks(x, show=True)
        >>> print(ind)
    
        >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
        >>> # set minimum peak height = 0 and minimum peak distance = 20
        >>> detect_peaks(x, mph=0, mpd=20, show=True)
    
        >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
        >>> # set minimum peak distance = 2
        >>> detect_peaks(x, mpd=2, show=True)
    
        >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
        >>> # detection of valleys instead of peaks
        >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)
    
        >>> x = [0, 1, 1, 0, 1, 1, 0]
        >>> # detect both edges
        >>> detect_peaks(x, edge='both', show=True)
    
        >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
        >>> # set threshold = 2
        >>> detect_peaks(x, threshold = 2, show=True)
        """

        x = np.atleast_1d(x).astype('float64')
        if x.size < 3:
            return np.array([], dtype=int)
        if valley:
            x = -x
        # find indices of all peaks
        dx = x[1:] - x[:-1]
        # handle NaN's
        indnan = np.where(np.isnan(x))[0]
        if indnan.size:
            x[indnan] = np.inf
            dx[np.where(np.isnan(dx))[0]] = np.inf
        ine, ire, ife = np.array([[], [], []], dtype=int)
        if not edge:
            ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
        else:
            if edge.lower() in ['rising', 'both']:
                ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
            if edge.lower() in ['falling', 'both']:
                ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
        ind = np.unique(np.hstack((ine, ire, ife)))
        # handle NaN's
        if ind.size and indnan.size:
            # NaN's and values close to NaN's cannot be peaks
            ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
        # first and last values of x cannot be peaks
        if ind.size and ind[0] == 0:
            ind = ind[1:]
        if ind.size and ind[-1] == x.size-1:
            ind = ind[:-1]
        # remove peaks < minimum peak height
        if ind.size and mph is not None:
            ind = ind[x[ind] >= mph]
        # remove peaks - neighbors < threshold
        if ind.size and threshold > 0:
            dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
            ind = np.delete(ind, np.where(dx < threshold)[0])
        # detect small peaks closer than minimum peak distance
        if ind.size and mpd > 1:
            ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
            idel = np.zeros(ind.size, dtype=bool)
            for i in range(ind.size):
                if not idel[i]:
                    # keep peaks with the same height if kpsh is True
                    idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                        & (x[ind[i]] > x[ind] if kpsh else True)
                    idel[i] = 0  # Keep current peak
            # remove the small peaks and sort back the indices by their occurrence
            ind = np.sort(ind[~idel])

    
        return ind
