import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset
import Wavelet_Jet_Detection

#================================#
# CONSTANTS
EARTH_RADIUS = 6380.0e3

#================================#
# WAVELET PARAMETERS

N_DECOMP_LEVELS = 4
confidence_param = 0.8

START_YEAR = 2010
END_YEAR   = 2011

base_sla_path = '/home/cchlod/AVISO/AVISO_Gridded_2/'
base_mdt_path = '/home/cchlod/AVISO/'


adt_file_name = 'adt_Kuroshio_2010.nc'

grad_thres = 0.15

wavelet_jet_detector = Wavelet_Jet_Detection.Jet_Detector(N_DECOMP_LEVELS,confidence_param,wavelet_basis='haar',
                       grad_thresh=grad_thres)


time_counter = 0
start_lon = 0

for i_year in range(START_YEAR,END_YEAR):
    #print base_sla_path+output_file_name+'thres_' + str(grad_thres) + '_' + str(i_year)+'.nc'
    #dataset_out        = Dataset(base_sla_path+output_file_name+'thres_' + str(grad_thres) + '_' + str(i_year)+'.nc',
    #                            'w',clobber=True, format='NETCDF4')
    #dataset_out.createDimension('time', None)
    #var_time = dataset_out.createVariable('time', 'f8', ['time'])

    
    #print 'YEAR: ', i_year
    dataset_adt = Dataset(base_sla_path+adt_file_name,'r')

    if i_year == START_YEAR:
        lat_adt     = dataset_adt.variables['latitude'][:]
        lon_adt     = dataset_adt.variables['longitude'][:]
        n_lon = lon_adt.size
        n_lat = lat_adt.size
        jet_histogram = np.zeros([n_lat,n_lon],dtype='u4')
    
    #dataset_out.createDimension('lat', n_lat)
    #dataset_out.createDimension('lon', n_lon)
    #var_lat = dataset_out.createVariable('lat', 'f8', ['lat'])
    #var_lon = dataset_out.createVariable('lon', 'f8', ['lon'])
    #var_lat[:] = lat
    #var_lon[:] = lon
    #var_hist = dataset_out.createVariable('jet_loc_hist', 'f8', ['lat','lon'])
    #var_locations = .createVariable('jet_locations', 'f8', ['time','lat','lon'])

    adt         = dataset_adt.variables['adt'][:,:,:]
    time        = dataset_adt.variables['time'][:]
    nT  = time.shape[0]
    
    jet_locations = np.zeros([nT,n_lat,n_lon],dtype='u4')

    #var_time[time_counter:time_counter+nT] = time
    
    for iT in range(0,nT):
        print "time step: ", iT, " of ", nT
        for i_lon in range(start_lon,n_lon):
            #print i_lon
            adt_slice = adt[iT,:,i_lon]
            adt_slice[adt_slice.mask] = np.nan
            lat_slice = lat_adt.copy()
            #lat_slice[np.isnan(adt_slice)] = np.nan
        
            lon_positions, lat_positions = wavelet_jet_detector.detect_jets(lon_adt[i_lon]*np.ones(n_lat), lat_slice,adt_slice,only_westward=True)
            
            
            for i_jet in range(0,len(lat_positions)):
                index_y = np.nonzero(lat_adt>=lat_positions[i_jet])[0][0]     
                jet_histogram[index_y,i_lon] = jet_histogram[index_y,i_lon]+1
                jet_locations[iT,index_y,i_lon] = 1
    #var_locations[0:nT,:,:] =  jet_locations 
    #time_counter = time_counter+nT    
    
        
    #dataset_out.close()  
    dataset_adt.close()



#var_hist[:,:] = jet_histogram         
#jet_histogram = np.ma.masked_array(jet_histogram,mask=jet_histogram==0)    
plt.figure(1)
cs = plt.contourf(lon,lat,np.squeeze(mdt),15,cmap=plt.cm.jet)
plt.colorbar(cs) 


plt.figure(2)
cs = plt.contourf(lon,lat,np.squeeze(mdt+sla[0,:,:]),15,cmap=plt.cm.jet)
plt.colorbar(cs)
#plt.scatter(jet_lon_lats[:,0],jet_lon_lats[:,1],s=60,facecolors='none', edgecolors='k')
 

plt.figure(3)
cs = plt.contourf(lon,lat[1::],np.squeeze(adt_diff_y[0,...]),15,cmap=plt.cm.jet)
plt.colorbar(cs) 
#plt.scatter(jet_lon_lats[:,0],jet_lon_lats[:,1],s=60,facecolors='none', edgecolors='k')

plt.figure(4)
cs = plt.contourf(lon,lat,jet_histogram,15,cmap=plt.cm.hot_r)


plt.show()

dataset_mdt.close()