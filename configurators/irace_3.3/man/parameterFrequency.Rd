% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/parameterAnalysis.R
\name{parameterFrequency}
\alias{parameterFrequency}
\title{Plot of histogram of parameter values}
\usage{
parameterFrequency(configurations, parameters, rows = 4, cols = 3,
  filename = NULL, pdf.width = 12, col = "gray")
}
\arguments{
\item{configurations}{Data frame containing target algorithms configurations 
in the format used by \pkg{irace}.}

\item{parameters}{List of target algorithm parameters in the \pkg{irace} format.}

\item{rows}{Number of plots per column.}

\item{cols}{Number of plots per row.}

\item{filename}{Filename prefix to generate the plots. If \code{NULL} the plot 
displayed but not saved.}

\item{pdf.width}{Width for the pdf file generated.}

\item{col}{Color of the bar plot.}
}
\description{
\code{parameterFrequency} plots the frequency of the parameters values in a
 set of target algorithm configurations. It generates plots showing the
 frequency of parameter values for each parameter, with \code{rows} *
 \code{cols} parameters being shown per plot.  If a filename is provided the
 plots are saved in one or more files.
}
\examples{
\donttest{
 ## To use data obtained by irace

 # First, load the data produced by irace.
 irace.logfile <- file.path(system.file(package="irace"), "exdata", "irace-acotsp.Rdata")
 load(irace.logfile)
 attach(iraceResults)
 parameterFrequency(allConfigurations, parameters)
}

}
\seealso{
\code{\link{readParameters}} to obtain a valid parameter structure from a parameters file.
 \code{\link{readConfigurationsFile}} to obtain a set of target algorithm configurations from 
   a configurations file.
}
\author{
Manuel López-Ibáñez and Leslie Pérez Cáceres
}
