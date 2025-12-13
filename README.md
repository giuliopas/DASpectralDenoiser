# DA(S)pectralDenoiser: A Spectral Subtraction-based Denoiser for Distributed Acoustic Sensing (DAS) data.  

![DAS denoising workflow using a Spectral Subtraction-based approach](https://github.com/giuliopas/DASpectralDenoiser/blob/main/fig/fig1.png)

This repository implements a spectral subtraction-based denoising algorithm for DAS data, applied to the Utah FORGE Enhanced Geothermal System (EGS) dataset.

A detailed description of the method can be found in the paper:

*G. Pascucci, S. Gaviano, A. Pozzoli, F. Grigoli (2025). Signal Enhancement of Distributed Acoustic Sensing data using a Spectral Subtraction-based Approach. Seismological Research Letters (SRL). doi:10.1785/0220250105*

---

## Setup / Installation

### 1. Clone the repository
```
git clone https://github.com/giuliopas/DASpectralDenoiser.git
cd DASpectralDenoiser
```

### 2. Create and activate a Python environment
*Option A: Using conda (recommended)*
```
conda create -n das_env
conda activate das_env
```
*Option B: using venv*
```
python -m venv das_env
# macOS/Linux
source das_env/bin/activate
# Windows (PowerShell)
.\das_env\Scripts\Activate.ps1
```

### 3. Install required packages
```
conda install pip

pip install --upgrade pip #optional step
pip install -r requirements.txt
```

## Usage:
The file `Example_Denoising_FORGE.ipynb` contains a tutorial on how to run the code on a test DAS dataset from Utah FORGE (April 2022 stimulation).

### Additional Notes:
The example DAS data (.tdms format) used in `Example_Denoising_FORGE.ipynb` can be obtained in two ways:
1. Download manually from Zenodo: https://doi.org/10.5281/zenodo.17554490
2. Download directly from the notebook using Python:
   ```
   import urllib.request
   fname = 'FORGE_DFIT_UTC_20220421_144609.398.tdms'
   url = "https://zenodo.org/records/17554490/files/FORGE_DFIT_UTC_20220421_144609.398.tdms"
   urllib.request.urlretrieve(url, fname)
   ```

