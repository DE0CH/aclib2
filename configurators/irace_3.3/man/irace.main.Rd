% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/main.R
\name{irace.main}
\alias{irace.main}
\title{irace.main}
\usage{
irace.main(scenario = defaultScenario(), output.width = 9999)
}
\arguments{
\item{scenario}{\code{\link{defaultScenario}}() The scenario setup of 
\pkg{irace}.}

\item{output.width}{9999 The width that must be used for the screen
output.}
}
\description{
\code{irace.main} is a higher-level interface to invoke \code{\link{irace}}.
}
\details{
The function \code{irace.main} checks the correctness of the
scenario, prints it, reads the parameter space from
\code{scenario$parameterFile}, invokes \code{\link{irace}} and
prints its results in various formatted ways. If you want a
lower-level interface, please see function \code{\link{irace}}.
}
\seealso{
\code{\link{irace.cmdline}} a higher-level command-line interface to
 \code{irace.main}.
 \code{\link{readScenario}} to read the scenario setup from  a file.
 \code{\link{defaultScenario}} to provide a default scenario for \pkg{irace}.
}
\author{
Manuel López-Ibáñez and Jérémie Dubois-Lacoste
}
