import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from OS_denoiser import OverSubtractionDAS

"""
Created on Oct 18th, 2024 - last edit Jan 31st, 2025.

Author: Giulio Pascucci
Contact: giulio.pascucci@phd.unipi.it

This code provides an Over Subtraction denoising workflow for 2D DFOS data, as described in the paper:

    "Signal Enhancement of Distributed Fiber-Optic Sensing (DFOS) data using a Spectral Subtraction-based Approach".
                        Giulio Pascucci, Sonja Gaviano, Alice Pozzoli, Francesco Grigoli.


                        Submitted (XX XX 2025) to Seismological Research Letters (SRL)

A full description of the method is available in the article (doi: XX.XX00000000)

"""

class DAS:

    def __init__(self, file, dx, gl, formato): # fname,skip
       
        print('Reading : ' + file)
        if formato=='tdms':
            from nptdms import TdmsFile
            tdms_file = TdmsFile(file)
            self.freq = tdms_file.properties['SamplingFrequency[Hz]']
            self.starttime = tdms_file.properties['CPUTimeStamp']
            self.gl=tdms_file.properties['GaugeLength']
            traces = (tdms_file.as_dataframe().to_numpy()).T
            # traces=traces[::skip]

            #print(type(self.starttime))
            #print(traces.shape)
            self.ntrs,self.npts = traces.shape
            self.dt=1./self.freq
            # self.fname = fname
            self.sam_rate_raw = tdms_file.properties['SamplingFrequency[Hz]']
            del tdms_file
        
        elif formato=='segy':
            from obspy.io.segy.core import  _read_segy
            das_data = _read_segy(file,format='segy',unpack_trace_headers=True)
            traces = np.stack([trace.data for trace in das_data])
            traces = np.array(traces,  dtype=np.float32)
                                    
            self.ntrs, self.npts = traces.shape
            self.freq = das_data[0].stats.sampling_rate
            self.dt = das_data[0].stats.delta
            # self.fname = fname
            self.starttime = das_data[0].stats.starttime
            print(type(self.starttime))
            self.sam_rate_raw = das_data[0].stats.sampling_rate
                        
        elif formato=='h5':
            import h5py
            h5file = h5py.File(file)
            traces = h5file['Acoustic']

            # PRINT ALL METADATA
            for key1 in traces.attrs.keys():
                print(key1, traces.attrs[key1])

            self.sam_rate_raw = traces.attrs['InterrogationRate(Hz)']
            self.freq = traces.attrs['InterrogationRate(Hz)']
            self.dt=1./self.sam_rate_raw
            self.starttime = traces.attrs['CPUTimeStamp(UTC)']
            self.gl=traces.attrs['GaugeLength']
            self.dx=traces.attrs['SpatialSamplingInterval']
            traces = np.array(traces).T

            self.ntrs,self.npts = traces.shape

            # self.fname = fname
            del h5file


        elif formato=='npz':
            loadfile = np.load(file)
            traces = loadfile['data']
            self.ntrs, self.npts = traces.shape
            self.dt = loadfile['dt']
            self.dx = loadfile['dx']
            self.sam_rate_raw = 1/self.dt
            self.freq = self.sam_rate_raw

        else:                
            print("Only tdms, segy, h5 and npz file formats are supported")
        
        self.traces = traces #[::5]
        self.dx = dx
        self.gl = gl



    def __downsample(self, data, sampling_rate):
        """
          Downsample the data to the new specified sampling_rate

          INPUT:
            data          --> (numpy matrix) data to downsample
            sampling_rate --> (int) new sampling rate

          OUTPUT
            data2 --> numpy matrix. Downsampled data

          NOTE: This operation is performed in place on the actual data arrays
        """
        from scipy.signal import resample
        
        sampling_rate = int(sampling_rate)        
        new = int(sampling_rate*self.npts/self.freq)
        data2 = resample(data, new, window='hann', axis= 1) 
        self.traces = data2
        self.ntrs, self.npts = data2.shape
        self.freq = sampling_rate
        self.dt = 1/self.freq
        return data2


    def __filter(self, data, ftype, fmin, fmax, order=4, tapering=False):
        """ 
        Filter the input data with a butterworth filter, based on the specified filter type (ftype) and frequencies

        INPUT PARAMETERS
            data       --> (numpy matrix) data to filter. Dimension order channels-time
            ftype      --> (str) type of filter to apply. Possible values: 'bandpass', 'highpass', 'lowpass'
            fmin, fmax --> minimum and maximum corner frequencies for filtering. 
            order      --> order of the filter 

        OUTPUT
            data2     -->  numpy matrix with dimension order channels-time

        NOTE: This operation is performed in place on the actual data arrays
        """
        from scipy.signal import butter, sosfilt, windows

        if ftype=='bandpass':
            sos = butter(order, [fmin,fmax], 'bandpass', fs=self.freq, output='sos')
        
        elif ftype=='highpass':
            sos = butter(order, fmin, 'highpass', fs=self.freq, output='sos')
                
        elif ftype=='lowpass':
            sos = butter(order, fmax, 'lowpass', fs=self.freq, output='sos')

        if tapering:
            taper_win=windows.tukey(data.shape[1], alpha=0.01)
            data_taper=data*taper_win

            data2 = sosfilt(sos, data_taper, axis=1)

        else:
            data2 = sosfilt(sos, data, axis=1)
        
        self.traces = data2
        self.ntrs,self.npts = data2.shape
        return data2
        
    
    def preprocess(self, data, sampling_rate=None, ftype='bandpass', fmin=2, fmax=250, fk=True, fkmax=250, order=4, K_out=None, K_inn=None, perc_fk=1, tapering=False):
        """
           This method applies several approaches to denoise data.
           It removes noisy traces based on their energy compared to the traces mean average.
           It lowpass filters and downsamples the data, based on the user specifications. It allows to
           quicker process the data later.
           Trace mean removal, bandpass filtering and trace amplitude normalization.
           Filtering in the FK domain.

           INPUT PARAMETERS
            data           --> numpy matrix. Data to denoise
            sampling_rate  --> (int) resampling frequency
            ftype          --> (str). Type of filter to apply to the data. Possible values are 'lowpass', 'highpass', 'bandpass'
            fmin, fmax     --> (int) minimum and maximum frequency for filtering the data
            fkmax          --> In the FK filtering, fkmax represents the maximum frequency of the filter. Higher frequencies are filtered out
            order          --> (int) Order of the butterworth filter to apply
            K_out          --> (float) in the definition of the FK filter, it changes the extent of 
                                        the filtering for k_max, k_min. The smaller the value, the 
                                        higher the filtering
            K_inn          --> (float) in the definition of the FK filter, it changes the extent of 
                        the cone filtering around k=0. The higher the value, the  narrower the filtering cone
  
            perc_fk        --> parameter for smoothing the FK filter at the edges

                   
           OUTPUT
            denoised traces --> (numpy matrix)

           NOTE
            Denoised traces are provided inplace in object.traces
        """

        traces = data

        if sampling_rate is not None:
            traces = self.__filter(traces, 'lowpass', fmin, fmax, 4, False)   # lowpass before downsampling
            traces = self.__downsample(traces, sampling_rate)       # downsampling of the data

        ### DEMEAN EACH TRACE SINGULARLY
        mean=np.mean(traces, axis=1)
        mean.shape=(mean.shape[0],1)
        traces=traces-mean
        
        traces = self.__filter(traces, ftype, fmin, fmax, order, tapering)           # Filtering the data based on the user specified parameters
        traces = self.__trace_normalization(traces)                                  # Normalize the data (optional)

        if fk:
            fkmax = fkmax
            K_out=K_out 
            K_inn=K_inn
            perc_fk=perc_fk
            
            traces = self.__fk_filt(traces, fmin, fkmax, K_out, K_inn, perc_fk, sampling_rate) #FK filtering the data 
        
        else:
            self.traces=traces

            self.ntrs,self.npts = traces.shape


    def data_select(self, tini, tend, xini, xend, rewrite=True):
        """
        This method provides a slice of data, based on the selected channels and time interval.

        INPUT
            tini, tend   --> (float) Start and end time of the selected chunk of data (in seconds)
            xini, xend   --> (int) Start and end channels of the selected chunk of data (channels)
            rewrite      --> (boolean) Define whatever rewriting the new traces inplace in the object.traces or not
        OUTPUT
            traces       --> (numpy matrix) Selected chunk of data
        """
        itini=int(tini/self.dt)    # Identify the time sample of the starttime
                
        if tend==-1:
            
            itend=int(self.traces.shape[-1] - self.traces.shape[-1]%self.dt)
        else:
            itend=int(tend/self.dt)
        
        ixini=int(xini)
        
        if xend==-1:
            ixend=int(self.traces.shape[0])
        else:
            ixend=int(xend)
        
        traces=self.traces[ixini:ixend,itini:itend]
        
        if rewrite:
            self.traces=traces
            self.ntrs,self.npts = traces.shape
        
        return traces
        
        
    #@vectorize(float32)
    def __trace_normalization(self, data, demean=True):
        """
             Normalize input data along axis 1 based on the trace maximum
             If demean is True, remove also the mean value

            INPUT
                data  --> numpy matrix of data to normalize
                demean--> (boolean) define whatever remove mean value
            OUTPUT
                data   --> numpy matrix of normalized data
        """

        if demean:
            data=data-np.mean(data)
        ntrs,npts = np.shape(data)
        for i in range(ntrs):
            nf=np.max(np.abs(data[i,:]))
            data[i,:]=data[i,:]/nf
        return data
    

    def __fk_filt(self, data, fmin, fmax, out_win=None, inn_win=None, perc=1, samp_rate=None): 
 
        """
            This method calculates the FK spectra of the data, creates the FK filter and apply it
             based on the user preferences.

             INPUT PARAMTERS
                data       --> numpy matrix of data (channel-time) to which apply the filtering
                fmin, fmax --> (float) minimum and maximum frequencies for bandpass filtering
                out_wind   --> (float) in the definition of the FK filter, it changes the extent of 
                                        the filtering for k_max, k_min. The smaller the value, the 
                                        higher the filtering
                inn_wind   --> (float) in the definition of the FK filter, it changes the extent of 
                        the cone filtering around k=0. The higher the value, the  narrower the filtering cone
                perc       --> smooth the filter at the edges 
                samp_rate  --> sampling_rate. Specify it if previously downsampling the data

            OUTPUT
                data_filt  --> filtered data
            
            NOTE
                Filtered data is substituted inplace to the object.traces attribute
        """

        from scipy.signal import windows
        
        # Calculates the FK data spectra
        f = np.fft.rfftfreq(self.npts, d=self.dt)   # Discrete Fourier Transform sample frequencies f-axis
        k = np.fft.fftfreq(self.ntrs, d=self.dx)    # Discrete Fourier Transform sample frequencies k-axis
        fk= np.fft.rfft2(data)    # Calculates FK spectra of the data

        # Creation of the FK filter to apply to the data 
        n,m = np.shape(fk)        # shape of the FK filter
        filt=np.ones([n,m])          

        num_samples=self.npts

        if samp_rate==None:
            samp_rate=self.freq
        signal_len=(num_samples/samp_rate)
        high_pass=fmax #frequecy Hz
        low_pass=fmin #frequency Hz
        high_pass=int(high_pass*signal_len)
        low_pass=int(low_pass*signal_len)

              
        # smooth the filter at the edges 
        delta_filt=int(perc*signal_len)
 
        # k=0 removal
        filt[0:16,:]=0.5
        filt[n-16:,:]=0.5
        filt[:,high_pass-delta_filt:]=0.5
        filt[:,0:low_pass+delta_filt]=0.5

        filt[0:8,:]=0.
        filt[n-8:,:]=0.
        filt[:,high_pass:]=0.
        filt[:,0:low_pass]=0.

        ##Define the shape of the outer and inner triangular windows of the FK filter and scale them by the number of considered traces
        if out_win!=None:
            max_value_outer_trian=int(m/out_win) 
            outer_window = (windows.triang(n) * max_value_outer_trian)
            for i in range(filt.shape[0]):
                filt[i,:int(outer_window[i])] = 0.
        
        if inn_win!=None:
            max_value_inner_trian=int(m*inn_win)  
            inn_window = (windows.triang(n) * max_value_inner_trian)  
            for i in range(filt.shape[0]):
                filt[i,int(inn_window[i]):] = 0.
            

        #print(np.abs(fk).shape, filt.shape, np.exp(1j*np.angle(fk)).shape)
                       
        fkfilt=np.abs(fk)*filt*np.exp(1j*np.angle(fk))      # Apply the filter to the data  
        data_filt=np.fft.irfft2(fkfilt)                     # Inverse transform

        self.traces=data_filt
        return data_filt      

    
    def run_OS_DAS_denoiser(self,
                            data,
                            dt, 
                            frame_duration, 
                            noise_duration=2.0,
                            window='hamming', 
                            G=0.9, 
                            Thresh=4, 
                            beta=0.005):
        """
        Apply Over Subtraction denoiser to DAS data: 
        (apply to self.traces and store in self.denoised).
        """
        denoised = OverSubtractionDAS(data,
                                    dt, 
                                    frame_duration=frame_duration,  
                                    noise_duration=noise_duration, 
                                    window=window, 
                                    G=G, 
                                    Thresh=Thresh, 
                                    beta=beta)

        self.denoised = denoised

        #denoised = self.__trace_normalization(denoised)   #optional
        return denoised


        
    def plotfk(self, data, filename, fmin=None, fmax=None, kmin=None, kmax=None, savefig=False):
        """
        Plot FK spectrum of arr.traces
        
        PARAMETERS:
            filename --> string. Name of the file where to save the plot
            fmin     --> float. Minimum frequency to display (default: None, i.e., plot all)
            fmax     --> float. Maximum frequency to display (default: None, i.e., plot all)
            kmin     --> float. Minimum wavenumber to display (default: None, i.e., plot all)
            kmax     --> float. Maximum wavenumber to display (default: None, i.e., plot all)
            savefig  --> boolean. Define whatever to save the plot or not
        """
        from matplotlib import colors
        
        # Tapering
        hann = np.hanning(self.traces.shape[1])
        data = data * hann
        
        # FK spectra calculation
        f = np.fft.rfftfreq(self.npts, d=self.dt)
        k = np.fft.fftfreq(self.ntrs, d=self.dx)
        fk = np.fft.rfft2(data) + 1
        
        # Normalize the FK spectrum
        fk = np.abs(fk)
        fk /= np.max(fk)
        
        # Apply frequency and wavenumber limits
        fmin = fmin if fmin is not None else min(f)
        fmax = fmax if fmax is not None else max(f)
        kmin = kmin if kmin is not None else min(k)
        kmax = kmax if kmax is not None else max(k)
        
        # Find the indices that correspond to the desired fmin, fmax, kmin, and kmax
        f_indices = np.where((f >= fmin) & (f <= fmax))[0]
        k_indices = np.where((k >= kmin) & (k <= kmax))[0]
        
        # Select the portion of the FK spectrum to plot
        fk_plot = fk[np.ix_(k_indices, f_indices)]
        
        # Plot the selected FK spectrum
        plt.figure(figsize=[8, 6])
        plt.imshow(np.fft.fftshift(fk_plot, axes=(0,)).T, extent=[kmin, kmax, fmin, fmax],
                aspect='auto', cmap='plasma', interpolation=None, origin='lower', norm=colors.LogNorm())
        
        # Add colorbar and labels
        h = plt.colorbar()
        h.set_label('Amplitude Spectra  (rel. 1 $(\epsilon/s)^2$)')
        plt.ylabel('Frequency [Hz]', fontsize=18)
        plt.xlabel('Wavenumber [1/m]', fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        
        # Save the figure, if required
        if savefig:
            plt.savefig(filename + '.pdf', dpi=200)


    def visualize(self, data, filename, vmin=None,vmax=None, cmap='seismic'):

        time=np.arange(data.shape[1])*self.dt
        channels=np.arange(data.shape[0])*self.dx

        plt.figure(figsize=[20,5], dpi=200)
        
        if vmin!=None and vmax!=None:
            plt.imshow(data, extent=[min(time), max(time), max(channels),min(channels)], cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        else:
            plt.imshow(data, extent=[min(time), max(time), max(channels), min(channels)], cmap=cmap, aspect='auto')
        

        plt.xlabel('Relative time [s]', fontsize=18)
        plt.ylabel('Distance along the fiber [m]', fontsize=18)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.colorbar(pad=0.01)
        #plt.close()
        plt.savefig(filename+'.pdf', dpi=200)
        plt.show()