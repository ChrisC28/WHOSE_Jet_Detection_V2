import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset
import Wavelet_Jet_Detection

#This is an example script showing how to use, and save, the WHOSE jet detection
#alogorithm. The example uses AVISO gridded altimetry (ADT) in the north Pacific region
#with a focus on the Kuroshio current. The example ADT file can be downloaded 
#from 



#=========================#
# WAVELET PARAMETERS
#=========================#
N_DECOMP_LEVELS = 4
confidence_param = 0.9

#============================#
#Set the gradient threshold
grad_thres = 0.001 #(units: m/km - reported in text as m/100km)
#============================#

#Start year and end year 
START_YEAR = 2010
END_YEAR   = 2011

#==================================================#
#Input and output paths of the files
base_sla_path = '/home/cchlod/AVISO/AVISO_Gridded_2/'
base_output_path = base_sla_path

adt_file_stem = 'sla_dt_SouthernOcean_'
output_file_stem = 'jet_detection_test_Southern_Ocean_'

base_mdt_path = '/home/cchlod/AVISO/'
mdt_file_name      = 'mdt_cnes_cls2013_global.nc'


#==================================================#


#==========================#
#Here we instantiate the
#wavelet jet detection class
#==========================#
wavelet_jet_detector = Wavelet_Jet_Detection.Jet_Detector(N_DECOMP_LEVELS,confidence_param,wavelet_basis='haar',
                       grad_thresh=grad_thres)
#==========================#                    
#some counters
time_counter = 0
start_lon = 0

#==========================#
#Get the MDT
#==========================#
dataset_mdt = Dataset(base_mdt_path+mdt_file_name,'r')
lat_mdt         = dataset_mdt.variables['lat'][:]
lon_mdt         = dataset_mdt.variables['lon'][:]
mdt             = dataset_mdt.variables['mdt'][:,:,:]


for i_year in range(START_YEAR,END_YEAR):
    
    #Load input data
    dataset_adt = Dataset(base_sla_path+adt_file_stem + str(i_year) + '.nc','r')
    sla         = dataset_adt.variables['sla'][:,:,:]
    time        = dataset_adt.variables['time'][:]    
    lat_adt     = dataset_adt.variables['lat'][:]
    lon_adt     = dataset_adt.variables['lon'][:]
    
    
    nT    = time.size   
    n_lon = lon_adt.size
    n_lat = lat_adt.size
    min_lat = lat_adt.min()
    max_lat = lat_adt.max()
    
    min_index = np.nonzero(lat_adt>=min_lat)[0][0]
    max_index = np.nonzero(lat_adt>=max_lat)[0][0]+1
        
    lat_mdt = lat_mdt[min_index:max_index]
    mdt     = mdt[:,min_index:max_index,:]
    adt = mdt + 1.0e-4*sla
    
    
    jet_histogram = np.zeros([n_lat,n_lon],dtype='u4')
    jet_locations = np.zeros([nT,n_lat,n_lon],dtype='u4')

    #============================#
    #Set up the output file
    #============================#
    print 'writting file to: ', base_output_path+output_file_stem + str(i_year)+'.nc'
    dataset_out        = Dataset(base_output_path+output_file_stem + str(i_year)+'.nc',
                                'w',clobber=True, format='NETCDF4')
                                                           
    dataset_out.createDimension('time', None)
    var_time = dataset_out.createVariable('time', 'f8', ['time'])

    dataset_out.createDimension('lat', n_lat)
    dataset_out.createDimension('lon', n_lon)
    var_lat = dataset_out.createVariable('lat', 'f8', ['lat'])
    var_lon = dataset_out.createVariable('lon', 'f8', ['lon'])
    var_time[:] = time

    var_lat[:] = lat_adt
    var_lon[:] = lon_adt
    var_hist      = dataset_out.createVariable('jet_loc_hist', 'f8', ['lat','lon'])
    var_locations = dataset_out.createVariable('jet_locations', 'f8', ['time','lat','lon'])
    #============================#
    
    for iT in range(0,nT):
        print "time step: ", iT, " of ", nT
        for i_lon in range(start_lon,n_lon):
            adt_slice = adt[iT,:,i_lon]
            adt_slice[adt_slice.mask] = np.nan
            
            #================================================================#
            #Here's where the magic happens
            #For each meridional transect, and at each time step, we apply the 
            #methodology.
            #================================================================#
            lon_positions, lat_positions = wavelet_jet_detector.detect_jets(lon_adt[i_lon]*np.ones(n_lat), lat_adt,adt_slice,only_eastward=True)
            
            
            for i_jet in range(0,len(lat_positions)):
                index_y = np.nonzero(lat_adt>=lat_positions[i_jet])[0][0]     
                jet_histogram[index_y,i_lon] = jet_histogram[index_y,i_lon]+1
                jet_locations[iT,index_y,i_lon] = 1
    var_locations[0:nT,:,:] =  jet_locations 
    time_counter = time_counter+nT    
    
        
    dataset_adt.close()
var_hist[:,:] = jet_histogram/float(time_counter)
dataset_out.close()  

#Let's make some plots
fig  = plt.figure(1)
ax   = fig.add_subplot(1,1,1)
cs = ax.contourf(lon_adt,lat_adt,adt[0,:,:],25,cmap=plt.cm.jet)
fig.colorbar(cs)
ax.contour(lon_adt,lat_adt,jet_locations[0,:,:],2,colors='k')
plt.show()