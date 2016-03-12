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
confidence_param = 0.90

START_YEAR = 2007
END_YEAR   = 2008

base_sla_path = '/home/cchlod/AVISO/AVISO_Gridded_2/'
base_mdt_path = '/home/cchlod/AVISO/'

sla_file_name_stem = 'sla_dt_SouthernOcean_'
mdt_file_name      = 'mdt_cnes_cls2013_global.nc'
output_file_name   = 'ssh_front_climatology_'

dataset_mdt = Dataset(base_mdt_path+mdt_file_name,'r')
lat         = dataset_mdt.variables['lat'][:]
lon         = dataset_mdt.variables['lon'][:]
mdt         = dataset_mdt.variables['mdt'][:,:,:]



wavelet_jet_detector = Wavelet_Jet_Detection.Jet_Detector('haar',N_DECOMP_LEVELS,confidence_param)


time_counter = 0
for i_year in range(START_YEAR,END_YEAR):
    
    dataset_out        = Dataset(base_sla_path+output_file_name+str(i_year)+'.nc','w',clobber=True, format='NETCDF4')
    dataset_out.createDimension('time', None)
    var_time = dataset_out.createVariable('time', 'f8', ['time'])

    
    print 'YEAR: ', i_year
    dataset_sla = Dataset(base_sla_path+sla_file_name_stem+str(i_year)+'.nc','r')

    if i_year == START_YEAR:
        lat_sla     = dataset_sla.variables['lat'][:]
        min_lat = lat_sla.min()
        max_lat = lat_sla.max()
        min_index = np.nonzero(lat>=min_lat)[0][0]
        max_index = np.nonzero(lat>=max_lat)[0][0]+1
        
        lat = lat[min_index:max_index]
        mdt = mdt[:,min_index:max_index,:]
        n_lon = lon.size
        n_lat = lat.size        
        jet_histogram = np.zeros([n_lat,n_lon],dtype='u4')
    
    dataset_out.createDimension('lat', n_lat)
    dataset_out.createDimension('lon', n_lon)
    var_lat = dataset_out.createVariable('lat', 'f8', ['lat'])
    var_lon = dataset_out.createVariable('lon', 'f8', ['lon'])
    var_lat[:] = lat
    var_lon[:] = lon
        #var_hist = dataset_out.createVariable('jet_loc_hist', 'f8', ['lat','lon'])
    var_locations = dataset_out.createVariable('jet_locations', 'f8', ['time','lat','lon'])

    sla         = dataset_sla.variables['sla'][:,:,:]
    time        = dataset_sla.variables['time'][:]
    nT  = time.shape[0]

    jet_locations = np.zeros([nT,n_lat,n_lon],dtype='u4')

    var_time[time_counter:time_counter+nT] = time
    adt = mdt + sla
    
    for iT in range(0,nT):
        print "time step: ", iT, " of ", nT
        for i_lon in range(0,n_lon):
            #print i_lon
            adt_slice = adt[iT,:,i_lon]
            adt_slice[adt_slice.mask] = np.nan
            lat_slice = lat.copy()
            lat_slice[np.isnan(adt_slice)] = np.nan
        
            lon_positions, lat_positions = wavelet_jet_detector.detect_jets(lon[i_lon]*np.ones(n_lat), lat_slice,adt_slice)

            
            for i_jet in range(0,len(lat_positions)):
                index_y = np.nonzero(lat>=lat_positions[i_jet])[0][0]     
                jet_histogram[index_y,i_lon] = jet_histogram[index_y,i_lon]+1
                jet_locations[iT,index_y,i_lon] = 1
    var_locations[0:nT,:,:] =  jet_locations 
    #time_counter = time_counter+nT    
    
    #Derivative of the adt
    adt_diff_y = np.diff(adt,axis=1)
        
    dataset_out.close()  
    dataset_sla.close()
    
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