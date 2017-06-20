# BCD - Bayesian Collaborative Denoiser for Monte-Carlo Rendering #

Reference implementation for the paper "Bayesian Collaborative Denoising 
for Monte-Carlo Rendering" by Malik Boughida and Tamy Boubekeur.

Please cite the following paper in case you are using this code:
Bayesian Collaborative Denoising for Monte-Carlo Rendering. Malik Boughida and Tamy Boubekeur. Computer Graphics Forum (Proc. EGSR 2017), vol. 36, no. 4, p. 137-153, 2017

Copyright(C) 2014-2017
Malik Boughida and Tamy Boubekeur
                                                                           
All rights reserved. 

You can refer to the project page for more information: http://www.telecom-paristech.fr/~boubek/papers/BCD/

### Building ###

This program uses CMake. It has been tested on Linux. It should work on Windows too, but building on MacOSX could require more efforts.

This program has several dependencies that need to be installed first:
Required:

* OpenEXR
* Eigen

Optional:

* OpenMP
* CUDA

To build on Linux, go to the directory containing this README file, then:

```
mkdir build
cd build
cmake ../src
make
```

You can also use the graphical user interface for cmake (cmake-gui). 
Use the following cmake option to compile without CUDA support (multi-core CPU execution only then): 
```
-DBCD_USE_CUDA=OFF
```

### Execution ###

Usage:

```
cd bin/
./BayesianCollaborativeDenoiser <arguments list>
```
Required arguments list:

* -o <output>          The file path to the output image
* -i <input>           The file path to the input image
* -h <hist>            The file path to the input histograms buffer
* -c <cov>             The file path to the input covariance matrices buffer

Optional arguments list:

* -d <float>           Histogram patch distance threshold (default: 1)
* -b <int>             Radius of search windows (default: 6)
* -w <int>             Radius of patches (default: 1)
* -r <0/1>             1 for random pixel order (in case of grid artifacts) (default: 0)
* -p <0/1>             1 for a spike removal prefiltering (default: 0)
* --p-factor <float>   Factor that is multiplied by standard deviation to get the threshold for classifying spikes during prefiltering. Put lower value to remove more spikes (default: 2)
* -m <float in [0,1]>  Probability of skipping marked centers of denoised patches. 1 accelerates a lot the computations. 0 helps removing potential grid artifacts (default: 1)
* -s <int>             Number of Scales for Multi-Scaling (default: 3)
* --ncores <nbOfCores> Number of cores used by OpenMP (default: environment variable OMP_NUM_THREADS)
* --use-cuda <0/1>     1 to use cuda, 0 not to use it (default: 1)
* -e <float>           Minimum eigen value for matrix inversion (default: 1e-08)

Example: 
./BayesianCollaborativeDenoiser -o filtered-image.exr -i noisy-image.exr -h noisy-image_hist.exr -c noisy-image_cov.exr

Precompiled MS Windows binaries are provided in the bin/win64 directory.

Only EXR images are supported.

### Conversion from raw full sampling images to proper inputs ###

The raw2bcd command line tool allows to convert raw binary many-samples per pixel (i.e. all the samples that get average to the final pixel color, before averaging them) files to the 3 EXR files required by BCD (per-pixel color, distribution/histogram, covariance matrix).

Usage: 
```
raw2bcd <input> <outputPrefix>.
```

Converts a raw file with all samples into the inputs for the Bayesian Collaborative Denoiser (BCD) program.

Required arguments list:
* <input>           The file path to the input image,
* <outputPrefix>    The file path to the output image, without .exr extension.

RAW sample images follow the following header structure:
<pre><code>typedef struct {
   int version;
   int xres;
   int yres;
   int num_samples;
   int num_channels;
   float data[1];
} Header;
</code></pre>

Depending on num_channels value you might get RGB (3) or RGBA (4) values.

The input file "test.raw" is provided as an example in the data/raw directory.