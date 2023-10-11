# Import python modules
from netCDF4 import Dataset,date2num
import numpy as np
from scipy.ndimage import gaussian_filter,uniform_filter
from scipy.ndimage import label
from scipy.stats import linregress,gaussian_kde
from scipy.signal import convolve2d
from skimage.morphology import square, dilation
from scipy.signal import find_peaks
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import RegularGridInterpolator,LinearNDInterpolator,NearestNDInterpolator
import os
from numba import njit
import scipy.special
from datetime import datetime
from sklearn.linear_model import LinearRegression
import sys

def yearday(year,month,day):
    '''Compute the day number in a year given the date'''
    nbdays = [31,28,31,30,31,30,31,31,30,31,30,31]
    #Special case for bissextile years
    if year%4==0 and year!=2000:
        nbdays = [31,29,31,30,31,30,31,31,30,31,30,31]
    return int(np.sum(nbdays[:month-1])+day)

def q_sat(T,p):
    '''
    T: temperature in K
    p: pressure in hPa
    Output: saturated mixing ration in kg/kg
    '''
    Rv = 461.0
    T0 = 273.15
    e0 = 6.13
    Lv = 2500000.
    epsilon = 0.622
    return epsilon*e0/p*np.exp(Lv/Rv*(1/T0-1/T))

def gamma_m(T,p):
    '''
    T: temperature in K
    p: pressure in hPa
    Output: moist adiabatic lapse rate in K/m
    '''
    Ls = 2500000.
    return 9.81*(1+Ls*q_sat(T,p)/284./T)/(1004.+(Ls**2)*q_sat(T,p)/461./(T**2))

def tropical_p(T):
    '''
    T: temperature in K
    output: estimated pressure in hPa
    '''
    les_p = np.linspace(50,1000,951)
    #load a typical temperature profile
    #(to be computed first, for example by integrating a moist adiabat)
    les_T = np.load('/users/bpoujol/tropical_T.npy')
    return np.interp(T,les_T,les_p)

def e_sat(T):
    '''
    T: temperature in K
    output: saturated water vapor pressure in hPa
    '''
    Rv = 461.0
    T0 = 273.15
    e0 = 6.13
    Lv = 2500000.
    return e0*np.exp(Lv/Rv*(1/T0-1/T))

#For the method to be faster, concavity of the temperature
#profile is computed in advance and saved
les_T = np.linspace(220,300,1001)
les_gamma_m = gamma_m(les_T,tropical_p(les_T))
derivative = np.gradient(les_gamma_m)/les_gamma_m/(les_T[1]-les_T[0])
np.save('dlngammadT.npy',derivative)

def dlngammadT(T):
    '''
    T: temperature in K
    output: d(ln(G))/dT in K-1 where G is the moist adiabatic lapse rate
    '''
    les_T = np.linspace(220,300,1001)
    values = np.load('/users/bpoujol/dlngammadT.npy')
    return np.interp(T,les_T,values)

def planck(lam,T):
    '''
    lam: wavelength in m
    T: temperature in K
    output: black body radiance in W/m2/sr/m
    '''
    h = 6.63e-34
    kb = 1.38e-23
    c = 299792458
    return (2*h*(c**2))/(lam**5)/(np.exp(h*c/(lam*kb*T))-1)
  
def invplanck(lam,B):
    '''
    lam: wavelength in m
    B: radiance in W/m2/sr/m
    output: brightness temperature in K
    '''
    h = 6.63e-34
    kb = 1.38e-23
    c = 299792458
    return (h*c)/(lam*kb*np.log(1+2*h*(c**2)/(B*(lam**5))))

def cap(A,valmax):
    '''
    This function caps the absolute value of the input array A to valmax
    '''
    A[A>valmax] = valmax
    A[A<-valmax]=-valmax
    return A

#Import OpenPIV for estimating horizontal winds
from openpiv import pyprocess,validation,filters

#Physical constants
Rv = 461.5
Rd = 287.
Ls = 2800000.
Lv = 2500000.
g = 9.81
cp = 1004.

h = 6.63e-34
kb = 1.38e-23
c = 299792458
C1 = 2*h*(c**2)
C2 = h*c/kb


def find_uv(Tb1,Tb2):
    '''
    Tb1: water vapor brightness temperature in frame 1
    Tb2: water vapor brightness temperature in frame 2
    Output (x,y,u0,v0) whith
    x,y : coordinates of the wind vectors
    u0,v0 : retrieved horizontal displacement in pixels and along the axes of the image
    '''
    winsize = 48 # pixels, interrogation window size in frame A
    searchsize = 56   # pixels, search area size in frame B
    overlap = 36  #overlap between interrogation windows
    dt = 1 # time interval between the two frames, fixed at 1

    #Retrieval of displacement
    u0, v0, sig2noise = pyprocess.extended_search_area_piv(
        Tb1,
        Tb2,
        window_size=winsize,
        overlap=overlap,
        dt=dt,
        search_area_size=searchsize,
        sig2noise_method='peak2peak',
        normalized_correlation=False
    )
    x, y = pyprocess.get_coordinates(
        image_size=Tb1.shape,
        search_area_size=searchsize,
        overlap=overlap,
    )
    #Standard quality control procedures recommended by OpenPIV
    u0, v0, mask = validation.global_std(
        u0, v0,
        std_threshold = 3,
    )
    u0, v0, mask = validation.local_median_val(
        u0, v0,
        u_threshold=3.,
        v_threshold=3.,
        size=4
    )
    u0, v0 = filters.replace_outliers(
        u0, v0,
        method='localmean',
        max_iter=1,
        kernel_size=1,
    )
    return x,y,u0,v0


@njit()
def retroadvect(exists,U,V,Tb2,Tb2_retroadvected,dt):
    '''
    This procedure advects back brightness temperature images onto their initial position
    exists: data mask
    U: wind along image x-axis in px/hr
    V: wind along image y-axis in px/hr
    Tb2: brightness temperature at time t+dt
    Tb2_retroadvected: brightness temperature at time t+dt advected back onto its position at t
    dt: time interval in hr
    '''
    for x in range(nx):
        for y in range(ny):
            if exists[y,x]:
                #first guess of the mean position between t and t+dt
                x_fg = round(x+U[y,x]/2.*dt)
                y_fg = round(y+V[y,x]/2.*dt)
                #asserts there is data at this position
                if x_fg>-1 and x_fg<nx and y_fg>-1 and y_fg<ny and exists[y_fg,x_fg]:
                    #compute precisely the initial position (order 2 scheme)
                    x_before = round(x+U[y_fg,x_fg]*dt)
                    y_before = round(y+V[y_fg,x_fg]*dt)
                    #asserts there is data at this position
                    if x_before>-1 and x_before<nx and y_before>-1 and y_before<ny and exists[y_before,x_before]:
                        #Advect back brightness temperature
                        Tb2_retroadvected[y,x]=Tb2[y_before,x_before]
                        
@njit()
def compute_derivative(exists,les_delta_T,les_t,dTbdt,err_dTbdt):
    '''
    Parallelized computation of the lagrangian time derivative of brightness temperature
    exists: data mask
    les_delta_T(time,x,y) : difference between brightness temperature advected back and 
                            brightness temperature at initial time
    les_t(time): time in hr
    dTbdt(x,y): time derivative of brightness temperature
    err_dTbdt(x,y): standard error on dTbdt
    '''
    for x in range(nx):
        for y in range(ny):
            if exists[y,x] :
                vals = les_delta_T[:,y,x]
                idx2 = np.isfinite(vals)
                n = np.sum(idx2)
                if n>2 : #we perform the regression only if there are more than two points
                    #Method with matrix to perform the regression, because it can be parallelized
                    A = np.vstack((les_t[idx2], np.ones(np.sum(idx2)))).T
                    a,b = np.linalg.lstsq(A, vals[idx2])[0]
                    dTbdt[y,x] = a
                    t_m = np.mean(les_t[idx2])
                    vals_pred = a*les_t[idx2]+b
                    #usual formula for the uncertainty on a least square linear regression
                    err_dTbdt[y,x]=np.sqrt(np.sum((vals_pred-vals[idx2])**2)/((n-2)*np.sum((les_t[idx2]-t_m)**2)))
                    
@njit()
def compute_advection(exists,les_err_Tadv,les_t,err_Tadv):
    '''
    Parallelized computation of the advective contribution u grad(Tb) to brightness temperature variations
    exists: data mask
    les_err_Tadv(time,x,y) : difference between brightness temperature advected back and 
                              actual brightness temperature
    les_t(time): time in hr
    err_Tadv(x,y): advective contribution in dTbdt
    '''
    for x in range(nx):
        for y in range(ny):
            if exists[y,x] :
                vals = les_err_Tadv[:,y,x]
                idx2 = np.isfinite(vals)
                n = np.sum(idx2)
                if n>2 :
                    A = np.vstack((les_t[idx2], np.ones(np.sum(idx2)))).T
                    a,b = np.linalg.lstsq(A, vals[idx2])[0]
                    err_Tadv[y,x] = a

def zenith(lon,lat,year,month,day,hour):
    '''
    Solar zenith angle (in radians) determined following NOAA's Global Monitoring Division
    lon(x,y),lat(x,y): latitude and longitude in degrees
    year, month, day: integers
    hour: float
    '''
    G = 2*np.pi/365.*(yearday(year,month,day)-1+(hour-12.)/24.)
    eqtime = 229.18*(0.000075 + 0.001868*np.cos(G) - 0.032077*np.sin(G) -0.014615*np.cos(2*G) - 0.040849*np.sin(2*G) )
    decl = 0.006918 - 0.399912*np.cos(G) + 0.070257*np.sin(G) - 0.006758*np.cos(2*G) + 0.000907*np.sin(2*G) - 0.002697*np.cos(3*G) + 0.00148*np.sin(3*G)
    tst = eqtime+4*lon+60*hour
    ha = np.pi/180*(tst/4.-180)
    return np.arccos(np.sin(lat*np.pi/180.)*np.sin(decl)+np.cos(lat*np.pi/180.)*np.cos(decl)*np.cos(ha))

def dadu(u):
    '''
    Derivative of the dimansionless absorptivity function from Chou (1986)
    '''
    return 0.5343*11.5*(1+64*(1-0.59)*(u**0.59))/((1+10.5*u+64*(u**0.59))**2)*np.exp(-11.5*u/(1+10.5*u+64*(u**0.59)))

file = 'a_typical_netcdf_radiance_file.nc'
lon = np.array(Dataset(file)['longitude'])[j0:j1,i0:i1] #load longitude
lat = np.array(Dataset(file)['latitude'])[j0:j1,i0:i1] #load latitude
zeta_sat = np.array(Dataset(file)['View_Zenith'])[j0:j1,i0:i1]*np.pi/180. #satellite zenith angle in degrees

#Limits of the bounding box where we want to compute vertical velocity
j0,j1 = 800,2900
i0,i1 = 500,3300

#Minutes over which brightness temperature is available at each hour
minutes = ['00','15','30','45']

#Radius of the buffer zone around convective clouds where the retrieval is masked out
convolve_radius = 30. #km

#Grid spacing of the satellite image
dx = 2.5 #km
kernel = square(round(convolve_radius/dx))
#Dimensions
ny,nx = lon.shape

#Name of the water vapor channel to retrieve and its wavelength
channel = 'WV_073'
lam = 7.3e-6

#Half-width of the kernel for the scale separation
scale_separation = 200 #km
sigma_scale = scale_separation//dx

#Enter here year and month for the retrieval
year = 2019
month = 8

#number of days in each month
nbdays = [0,31,28,31,30,31,30,31,31,30,31,30,31]
if year%4==0:
    nbdays[2]=29

for day in range(1,nbdays[month]+1):
    try: ncfile.close()  # just to be safe, make sure dataset is not already open.
    except: pass
    #Create a new netcdf file to contain the retrieval
    ncfile = Dataset('/output_directory/omega_'+channel+'_'+str(10000*year+100*month+day)+'.nc',mode='w',format='NETCDF4_CLASSIC') 
    #Add latitude and longitude, time
    lat_dim = ncfile.createDimension('lat', ny)     # latitude axis
    lon_dim = ncfile.createDimension('lon', nx)    # longitude axis
    time_dim = ncfile.createDimension('time', None) # unlimited axis (can be appended to).
    ncfile.title='Vertical velocity from GOES'
    lats = ncfile.createVariable('lats', np.float32, ('lat','lon'))
    lats.units = 'degrees_north'
    lats.long_name = 'latitude'
    lons = ncfile.createVariable('lons', np.float32, ('lat','lon'))
    lons.units = 'degrees_east'
    lons.long_name = 'longitude'
    time = ncfile.createVariable('time', np.float32, ('time',))
    time.units = 'hours since 1800-01-01'
    time.long_name = 'time'
    # Define a 3D variable to hold the retrieval and its standard error
    omega = ncfile.createVariable('omega',np.float32,('time','lat','lon')) # note: unlimited dimension is leftmost
    omega.units = 'hPa/hr' # degrees Kelvin
    omega.standard_name = 'pressure_velocity' # this is a CF standard name
    err_omega = ncfile.createVariable('err_omega',np.float64,('time','lat','lon')) # note: unlimited dimension is leftmost
    err_omega.units = 'hPa/hr' # degrees Kelvin
    err_omega.standard_name = 'uncertainty_on_pressure_velocity' # this is a CF standard name
    #Define variables to hold metadata: air temperature at the emission level and associated pressure
    ta = ncfile.createVariable('temp',np.float32,('time','lat','lon')) # note: unlimited dimension is leftmost
    ta.units = 'K' # degrees Kelvin
    ta.standard_name = 'temperature_at_emission_level'
    pressure = ncfile.createVariable('pres',np.float32,('time','lat','lon')) # note: unlimited dimension is leftmost
    pressure.units = 'hPa' # degrees Kelvin
    pressure.standard_name = 'pressure_at_emission_level'
    lats[:,:]=lat
    lons[:,:]=lon
    for hour in range(24):
        nc_timestep = hour
        date = str(hour+100*day+10000*month+1000000*year)
        print(date,end='\r') #This prints the date continuously so that one can follow progress fo the algorithm
        #Load the files between hour and hour+1
        files = ['/input_directory/'+str(year)+'/'+date[:4]+'_'+date[4:6]+'_'+date[6:8]+'/my_image_'+date+minute+'.nc' for minute in minutes]
        #Assert all files are actually present
        files_exist = np.array([os.path.exists(f) for f in files]).all()
        if files_exist :
            #Compute the emission level optical thickness
            eta = h*Rv*c/(kb*Ls*lam)
            correction = scipy.special.gamma(1+eta) #scipy.special.gamma is the Euler gamma function
            tau_star = 1+eta
            
            #Get brightness temperatures in the atmospheric window and the water vapor channel at initial and final time
            IR1 = np.array(Dataset(files[0])['IR_103'][j0:j1,i0:i1])
            IR1[IR1<100]=350
            FIR1 = np.array(Dataset(files[0])['IR_112'][j0:j1,i0:i1])
            FIR1[FIR1<100]=350
            IR2 = np.array(Dataset(files[-1])['IR_103'][j0:j1,i0:i1])
            IR2[IR2<100]=350
            FIR2 = np.array(Dataset(files[-1])['IR_112'][j0:j1,i0:i1])
            FIR2[FIR2<100]=350
            Tb1 = np.array(Dataset(files[0])[channel][j0:j1,i0:i1])
            Tb2 = np.array(Dataset(files[-1])[channel][j0:j1,i0:i1])
            #Mask regions where the difference is less than 10K or where there are semitransparent clouds
            #The dilation function adds a buffer zone around the mask
            to_mask = np.logical_or(dilation(np.logical_or(IR1<Tb1+10,IR2<Tb2+10),kernel),
                                    dilation(np.logical_or(IR1-FIR1>2.5,IR2-FIR2>2.5),kernel))
            
            #Prepare the tables to hold horizontal wind retrievals
            les_U = np.ones((3,ny,nx))*np.nan
            les_V = np.ones((3,ny,nx))*np.nan
            for k in range(3):
                #Select every couple of WV brightness temperature images
                file1,file2 = files[k:k+2]
                Tb1 = np.array(Dataset(file1)[channel][j0:j1,i0:i1])
                Tb2 = np.array(Dataset(file2)[channel][j0:j1,i0:i1])
                #Remove NaN values
                Tb1[Tb1<1]=np.nan
                Tb2[Tb2<1]=np.nan
                #Save the NaN mask
                to_reject = np.logical_or(np.isnan(Tb1),np.isnan(Tb2))
                #Replace NaN values by uniform value
                Tb1[to_reject]=np.mean(Tb1[~to_reject])
                Tb2[to_reject]=np.mean(Tb2[~to_reject])
                #High-pass and rejection band filtering to select appropriate WV features
                sigma1 = 0.5
                sigma2 = 8
                sigma3 = 15
                filtered_Tb1 = 8*(Tb1 - gaussian_filter(Tb1,sigma1)) + cap(gaussian_filter(Tb1 - gaussian_filter(Tb1,sigma3),sigma2),1)
                filtered_Tb2 = 8*(Tb2 - gaussian_filter(Tb2,sigma1)) + cap(gaussian_filter(Tb2 - gaussian_filter(Tb2,sigma3),sigma2),1)

                #Compute horizontal displacement from the WV images
                x,y,u0,v0 = find_uv(filtered_Tb1,filtered_Tb2)
                ny,nx = Tb1.shape
                xx,yy = np.meshgrid(range(nx),range(ny))
                #Interpolate the obtained vectors onto the satellite grid
                interp = RegularGridInterpolator((y[:,0],x[0]),u0,bounds_error=False)
                les_U[k] = interp((yy,xx))
                interp = RegularGridInterpolator((y[:,0],x[0]),v0,bounds_error=False)
                les_V[k] = interp((yy,xx))

            #Assert that some wind could be retrieved before continuing further
            if np.sum(np.isfinite(les_U))>0:
                #Put the wind speed in px/hr instead of px/frame 
                #HERE THERE IS A FACTOR 4 BUT YOU MAY NEED TO CHANGE IT DEPENDING ON TIME RESOLUTION OF YOUR DATA
                les_U = les_U*4
                les_V = les_V*4
                #Compute standard deviation and median
                std_U = np.nanstd(les_U,axis=0)
                std_V = np.nanstd(les_V,axis=0)
                U = np.nanmedian(les_U,axis=0)
                V = np.nanmedian(les_V,axis=0)
                #Set missing values if less than 3 wind retrievals are available on a given pixel
                to_remove = np.logical_or(np.sum(np.isnan(les_U),axis=0)>2,np.sum(np.isnan(les_V),axis=0)>2)
                U[to_remove]=np.nan
                V[to_remove]=np.nan
                std_U[to_remove]=np.nan
                std_V[to_remove]=np.nan
                del les_U
                del les_V

                #Load WV brightness temperature at initial and final time
                Tb1 = np.array(Dataset(files[0])[channel][j0:j1,i0:i1])
                Tb2 = np.array(Dataset(files[-1])[channel][j0:j1,i0:i1])
                Tb1[Tb1<1]=np.nan
                Tb2[Tb2<1]=np.nan
                #Same for temperature in the atmospheric window
                IR1 = np.array(Dataset(files[0])['IR_108'][j0:j1,i0:i1])
                IR1[IR1<1]=np.nan
                #Compute optical thickness at the surface
                tau = IR1/Tb1*e_sat(IR1)/e_sat(Tb1)*(Lv/Rv+Tb1)/(Lv/Rv+IR1)
                e_tau = np.exp(-tau)
                #Compute air temperature at the emission level
                Tb1 = invplanck(lam,(tau_star**eta)*(planck(lam,Tb1)-e_tau*planck(lam,IR1))/(correction-e_tau*(tau**eta)*(1+eta/tau)))

                #mask out U and V where there are clouds
                U[to_mask]=np.nan
                V[to_mask]=np.nan
                #Get the final data mask (clouds + where horizontal wind could not be retrieved)
                exists = np.isfinite(U)&np.isfinite(V)

                #Initialize the fields containing brightness temperature differences
                ny,nx = Tb1.shape
                les_delta_T = np.zeros((4,ny,nx))
                les_err_Tadv = np.zeros((4,ny,nx))

                for k in range(1,4):
                    dt = k/4. #Time interval between the image and the initial image
                    #Load brightness temperature in WV channel and infrared window
                    Tb2 = np.array(Dataset(files[k])[channel][j0:j1,i0:i1])
                    Tb2[Tb2<1]=np.nan
                    IR2 = np.array(Dataset(files[k])['IR_108'][j0:j1,i0:i1])
                    IR2[IR2<1]=np.nan
                    #Compute surface optical thickness
                    tau = IR2/Tb2*e_sat(IR2)/e_sat(Tb2)*(Lv/Rv+Tb2)/(Lv/Rv+IR2)
                    e_tau = np.exp(-tau)
                    #Deduce air temperature at the emission level
                    Tb2 = invplanck(lam,(tau_star**eta)*(planck(lam,Tb2)-e_tau*planck(lam,IR2))/(correction-e_tau*(tau**eta)*(1+eta/tau)))
                    #Advect it back onto its position at t
                    Tb2_retroadvected = np.ones(Tb1.shape)*np.nan
                    retroadvect(exists,U,V,Tb2,Tb2_retroadvected,dt)
                    #Save lagrangian variation and advective contribution to this variation
                    les_delta_T[k] = Tb2_retroadvected-Tb1
                    les_err_Tadv[k] = Tb2_retroadvected-Tb2

                #Initialize lagrangian derivative of brightness temperature
                dTbdt = np.zeros(Tb1.shape)
                err_dTbdt = np.zeros(Tb1.shape)
                err_Tadv = np.zeros(Tb1.shape)
                #Time axis of the different frames
                les_t = np.linspace(0,1-1/4.,4)
                #Compute derivative and estimate advective contribvution to this derivative
                compute_derivative(exists,les_delta_T,les_t,dTbdt,err_dTbdt)
                compute_advection(exists,les_err_Tadv,les_t,err_Tadv)

                #Find missing data
                mask_deriv = np.logical_or(dTbdt==0,np.isnan(dTbdt))
                dTbdt[np.isnan(dTbdt)]=0.
                
                #Apply gaussian filter to get large scale contribution
                division = gaussian_filter(1.-mask_deriv.astype('int'),sigma=sigma_scale)
                dTbdt_large = gaussian_filter(dTbdt,sigma=sigma_scale)/division
                #Deduce small scale contribution as the residual
                dTbdt_small = dTbdt-dTbdt_large

                #Put bacl NaNs at missing data places
                dTbdt_large[mask_deriv]=np.nan
                dTbdt_small[mask_deriv]=np.nan
                dTbdt[mask_deriv]=np.nan

                #Estimated pressure
                pres = tropical_p(Tb1)
                pres[to_mask]=np.nan
                #moist adiabatic lapse rate
                gamma = gamma_m(Tb1,pres)
                #Dimensionless quantities defined in the manuscript
                delta = g/(Rd*gamma)
                theta = Tb1*Rv/Lv
                psi = -dlngammadT(Tb1)*Rv*(Tb1**2)/Lv
                #Compute vertical velocity
                omega_arr = Lv*pres*( ((2+delta)*theta+1)/((1+delta)*theta+1) + psi/(delta+1)*(1-delta/(1+theta*(delta+1))) + delta*theta )* (dTbdt_small/(Rv*(Tb1**2)*(Lv*Rd/(Rv*Tb1*cp)-1)) + dTbdt_large/(Rv*(Tb1**2)*(Lv*Rd*gamma/(Rv*Tb1*g)-1)))
                #Remove values exceeding 100hPa/hr
                omega_arr[np.abs(omega_arr)>100]=np.nan

                #Compute correction for shortwave absorption
                #Determine the value of kappa (here deduced from figure S1 for the channel 7.3µm)
                kappa = Rd*pres*tau_star/((100000.)*0.622*1.106)
                #Constants from Chou (1986)
                p0 = 300. #hPa
                u0 = 10. #kg/m2
                #Solar zenith angle
                mu = 1./np.cos(zenith(lon,lat,year,month,day,hour+0.5))
                #Column water vapor from TOA to the emission level
                CWV = np.abs(mu)*tau_star/kappa*((1+delta)*theta+1)/(1+theta)
                #Estimate of the spurious vertical velocity due to shortwave absorption
                delta_omega = np.sign(mu)*((pres/p0)**0.8)*g*tau_star*Rv*np.cos(zeta_sat)/(Lv*kappa*cp*(theta**2))*(((1+(delta+1)*theta)*(1+(.8*delta+1)*theta))/((1+theta)*(1-delta*theta)))*1368.*dadu((CWV/u0)*((pres/p0)**0.8)*np.exp(0.00135*(Tb1-240.)))/u0
                #Correction of the vertical velocity estimate in hPa/hr
                omega_arr -= delta_omega*3600./100.
                
                #Save output in the netCDF file
                omega[nc_timestep,:,:] = omega_arr
                #Compute and save standard error
                err_omega[nc_timestep,:,:] = Lv*pres*( ((2+delta)*theta+1)/((1+delta)*theta+1) + psi/(delta+1)*(1-delta/(1+theta*(delta+1))) + delta*theta )/(Rv*(Tb1**2)*(Lv*Rd/(Rv*Tb1*cp)-1))*np.sqrt((std_U**2+std_V**2)/(U**2+V**2)*(err_Tadv**2) + err_dTbdt**2)
                #Save emission level data (temperature and pressure)
                ta[nc_timestep,:,:] = Tb1
                pressure[nc_timestep,:,:] = pres
                #Save timestep
                time[nc_timestep] = date2num(datetime(year,month,day,hour),time.units)
                
#Close the dataset
ncfile.close()
