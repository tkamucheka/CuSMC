% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{step}
\alias{step}
\title{Simulations step}
\usage{
step(N, d, timeSteps, Y, F, df, resampler, distribution)
}
\arguments{
\item{N}{[integer]: Number of particles is solution.}

\item{d}{[integer]: Number of parameters.}

\item{timeSteps}{[integer]: Total time steps.}

\item{Y}{[matrix]: Input observations.}

\item{F}{[matrix]: Covariant matrix for scaling particle samples}

\item{distribution}{[string]: Distribution for sampling particles}

\item{m0}{[vector]: Initial parameters at t=0}

\item{C0}{[matrix]: Initial covariant matrix at t=0}

\item{sampler}{[string]: Resampling sampler}
}
\description{
Simulations step
}
