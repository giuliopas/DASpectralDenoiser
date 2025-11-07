import numpy as np
from tqdm import tqdm

#######################

def set_alpha(SNR):
    """
    Set the alpha parameter for Over Subtraction
    """
    if 0 <= SNR <= 20:
        return 3 - (2/20)*SNR
    if SNR < 0:
        return 5
    if SNR > 20:
        return 1


def OverSubtractionDAS(data,
                       dt, 
                       frame_duration=0.06, 
                       noise_duration=2.0,
                       window='hamming', 
                       G=0.9, 
                       Thresh=4, 
                       beta=0.005):
    """
    Spectral Over Subtraction Denoising for 2D DFOS data.

    Parameters:
    ----------
    data : ndarray
        2D array with raw DFOS data [n_channels, n_samples]
    dt : float
        Sampling interval in seconds
    frame_duration : float
        Frame length in seconds
    noise_duration : float
        Time duration (in seconds) for estimating the noise spectrum at the start of each trace
    window : str
        Type of window to use ('hamming' or 'hanning')
    G : float
        Noise update factor (default=0.9)
    Thresh : float
        SNR threshold for updating noise spectrum (default=4)
    beta : float
        Minimum allowed noise level (default=0.005)

    Returns:
    -------
    denoised_arr : ndarray
        2D array of denoised DFOS traces
    """

    ntrs, npts = data.shape
    sam_rate_raw = 1 / dt

    # Frame size (even)
    fSize = int(frame_duration * sam_rate_raw / 2) * 2
    overlap_perc = 50 #50% overlap
    len1 = int(fSize * (overlap_perc / 100))
    len2 = fSize - len1
    Nframes = int(npts / len2) - 1

    # Set window
    if window == 'hamming':
        win = np.hamming(fSize)
    elif window == 'hanning':
        win = np.hanning(fSize)
    else:
        raise ValueError("Unsupported window type. Use 'hamming' or 'hanning'.")


    # Compute number of noise frames from 'noise_duration'
    tot_noise_samp = int(noise_duration * sam_rate_raw)
    noise_N_frames = tot_noise_samp // fSize

    if noise_N_frames < 1:
        raise ValueError("Noise duration is too short to fit at least 1 frame. Please increase 'noise_duration' param.")

    print(f"Using first {noise_N_frames} frames (~{noise_duration:.2f}s) for noise spectrum estimation.")


    # FFT size (next power of 2)
    nFFT = 2 * 2**8
    denoised_arr = np.zeros(data.shape)

    for trace_idx in tqdm(range(ntrs), desc="Processing Channel: "):
        x = data[trace_idx, :]

        x_old = np.zeros(len1)
        x_final = np.zeros(Nframes * len2)

        # Estimate average noise spectrum from first N frames
        noise = np.zeros(nFFT)
        j = 1
        for k in range(1, noise_N_frames + 1):
            noise += np.abs(np.fft.fft(win * x[j:j+fSize], nFFT))
            j += fSize
        noise_avg = noise / noise_N_frames

        if beta >= 1:
            raise ValueError('Beta must be between 0 and <1.')

        k = 0
        for n in range(Nframes):
            segm_x = win * x[k:k+fSize]
            x_fft = np.fft.fft(segm_x, nFFT)
            x_mag = np.abs(x_fft)
            theta = np.angle(x_fft)

            SSNR = 10 * np.log10(np.sum(x_mag ** 2) / np.sum(noise_avg ** 2))
            alpha = set_alpha(SSNR)

            ss = x_mag**2 - alpha * noise_avg**2
            diff = ss - beta * noise_avg**2
            ss[diff < 0] = beta * noise_avg[diff < 0]**2

            if SSNR < Thresh:
                noise_temp = G * noise_avg**2 + (1 - G) * x_mag**2
                noise_avg = np.sqrt(noise_temp)

            x_phase = np.sqrt(ss) * np.exp(1j * theta)
            x_ifft = np.real(np.fft.ifft(x_phase))

            x_final[k:k+len2] = x_old + x_ifft[:len1]
            x_old = x_ifft[len1:fSize]
            k += len2

        denoised_arr[trace_idx, :len(x_final)] = x_final

    return denoised_arr