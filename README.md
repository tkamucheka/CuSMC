
<!-- README.md is generated from README.Rmd. Please edit that file -->

# CuSMC

<!-- badges: start -->
<!-- badges: end -->

CuSMC is an R package for drawing random samples for a posterior
probability distribution in Bayesian inference. CuSMC stands for CUDA
Sequential Monte Carlo. CuSMC supports Metropolis-Hastings sampler and
Multivariate Normal and Student-T distributions.

## Prerequisites

You do not need a GPU to install and run CuSMC library. However, to
enjoy the best perfomance, make sure you are running Linux and an Nvidia
GPU (Graphics Processing Unit) that supports CUDA (most if not all
modern GPUs do) and the CUDA SDK installed. Details on how to set up the
CUDA SDK kit can be found
[here](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#installing-cuda-development-tools).
Windows and MacOS generally will not support GPU acceleration. See more
details on MacOS support below.

### Linux

If there is no NVidia GPU and CUDA SDK installed, CuSMC will install
with CPU support only. If an NVidia GPU and CUDA SDK are found during
installation then full GPU acceleration becomes available. You will need
the GCC compiler and tool chain installed to build the library on your
computer. If you are on Debian based distro like Ubuntu, you can install
the compiler with:

``` sh
$ sudo apt get install build-essential -y
```

### MacOS

MacOS supports NVidia GPUs and CUDA up to MacOS version 10.13.6. If you
have a version of MacOS after 10.13.6 then GPU acceleration will not be
available. There are two ways to get a compiler toolchain installed on
MacOS. Option 1, is through `xcode-select` and the other is via
Homebrew:

``` sh
# Xcode
$ xcode-select --install

# Homebrew
$ brew install gcc
```

### Windows

Currently, GPU acceleration is not available on Windows. Before
attempting to install, you will need to install RTOOLS40 to get the
MSYS2 MINGW64 environment and GCC compiler and toolchain installed and
available in R. RTOOLS40 can be foundR
[here](https://cran.r-project.org/bin/windows/Rtools/)

When you have installed the GCC compiler and toolchain, next you will
need to install the `devtools`, `Rcpp` and `RcppEigen` packages from CRAN as a last
step before installing `CuSMC`

``` r
## Install the following packages from CRAN
install.packages("devtools")
install.packages("Rcpp")
install.packages("RcppEigen")

## If you are using an old version of R (R 3.6 or older),
## you could also need to install RcppArmadillo package
install.packages("RcppArmadillo")
```

## Installation

You can install CuSMC from this repository with:

``` r
devtools::install_github("tkamucheka/cusmc")
library(CuSMC)
```

## Examples

``` r
## basic example code
```
