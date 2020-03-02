
<!-- README.md is generated from README.Rmd. Please edit that file -->

# CuMCMC

<!-- badges: start -->

<!-- badges: end -->

CuMCMC is an R package for statisical learning. CuMCMC stands for Cuda
Markov Chain Monte Carlo. CuMCMC uses a CUDA capable GPU to generate
particles in the posterior distribution. Currently, CuMCMC support the
Metropolis-Hastings sampler and Multivariate Normal and Student-T
distributions.

## Installation

You can install the released version of CuMCMC from
[CRAN](https://CRAN.R-project.org) with:

``` r
install.packages("CuMCMC")
```

## Example

This is a basic example which shows you how to solve a common problem:

``` r
library(CuMCMC)
## basic example code
```

What is special about using `README.Rmd` instead of just `README.md`?
You can include R chunks like so:

``` r
summary(cars)
#>      speed           dist       
#>  Min.   : 4.0   Min.   :  2.00  
#>  1st Qu.:12.0   1st Qu.: 26.00  
#>  Median :15.0   Median : 36.00  
#>  Mean   :15.4   Mean   : 42.98  
#>  3rd Qu.:19.0   3rd Qu.: 56.00  
#>  Max.   :25.0   Max.   :120.00
```

You’ll still need to render `README.Rmd` regularly, to keep `README.md`
up-to-date.

You can also embed plots, for example:

<img src="man/figures/README-pressure-1.png" width="100%" />

In that case, don’t forget to commit and push the resulting figure
files, so they display on GitHub\!
