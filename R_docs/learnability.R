library(foreign)
library(ggplot2, quietly=TRUE)
library(reshape2, quietly=TRUE)
library(dplyr, quietly=TRUE)
library(plyr, quietly=TRUE)
library(car, quietly=TRUE)
library(caret, quietly=TRUE)
library(fastDummies, quietly=TRUE)
library(dummies, quietly=TRUE)
library(devtools, quietly=TRUE)
library(roxygen2)
document(file.path(getwd(),'loc.R.utils'))


vargrp <- function(varname) {
  split2char(sprintf('%s:1,%s:2,%s:3,%s:4',varname,varname,varname,varname))
}

rank_ <- function(x) {
  rank(x, ties.method='max')
}

get_alts <- function(data, varname, altmask) {
  unrolled <- as.vector(t(data[vargrp(varname)]))
  to_append <- matrix(unrolled[altmask], ncol=3, byrow=TRUE)
  colnames(to_append) <- split2char(sprintf('%s:1,%s:2,%s:3',varname,varname,varname))
  return(to_append)
}

rownorm <- function(row) {
  if (min(row) == max(row)) {
    row / 10
  } else {
    (row-min(row))/(max(row)-min(row))
  }
}

STRINGFACT <- FALSE
NORMAL_SELF_REPORTS <- FALSE

# Load data
mdata <- read.csv('cmnr_data30cols.csv', check.names = FALSE)

reltdata <- read.csv('cmnr_mdata.csv', check.names = FALSE)[split2char('relt:1,relt:2,relt:3,relt:4')]
mdata <- cbind(mdata, reltdata)
mdata <- mdata[, c(2:28, 36:39, 29:35)]

xdata <- read.csv('cmnr_xdata.csv', check.names = FALSE)

fulldata <- mdata

# Discard incomplete cases
fulldata <- fulldata[complete.cases(fulldata), ]

# Convert categorical data to factors
factor_col_names <- c('current', 'nxt')
fulldata[, factor_col_names] <- lapply(fulldata[, factor_col_names], factor)

# May be convert number-coded factors to explicit names
if (STRINGFACT) {
  fulldata[, factor_col_names] <- lapply(fulldata[, factor_col_names], function(x) {
    mapvalues(x, from = as.character(1:4), to = split2char('1D,I1D,2D,R'))}
  )
}

# Create 'current' dummies
dummies <- dummy_cols(fulldata$current)[c('.data_1', '.data_2', '.data_3', '.data_4')]
colnames(dummies) <- split2char('currentd:1,currentd:2,currentd:3,currentd:4')
fulldata <- cbind(fulldata, dummies)
rm(dummies)

# Match unique xdata rows with corresponding rows in mdata and join datasets
inds <- match(fulldata$sid, xdata$sid) # get row indices of xdata for each mdata sid
fulldata <- cbind(fulldata, xdata[inds, c(-1,-2,-3)]) # concatenate datasets
rm(xdata)

# May be normalize self reports
if (NORMAL_SELF_REPORTS) {
  for (varg in split2char('lrn:,int:,comp:,time:,prog:,rule:,lrn2:')) {
    normcols <- apply(fulldata[, grepl(varg, names(fulldata))], 1, rownorm)
    fulldata[, grepl(varg, names(fulldata))] <- t(normcols)
    rm(normcols)
  }
}

# Rearrange structure
fulldata <- fulldata[c(1,2,4:7,8:35,39:74)]

# pvals
# =================
pvalend <- fulldata[fulldata$trial==200, split2char('pval:1,pval:2,pval:3,pval:4')]
Ss <- fulldata[fulldata$trial==250, 'sid']
mask <- fulldata$trial==61 & fulldata$sid!=setdiff(fulldata$sid, Ss)

pvalbeg <- fulldata[mask, split2char('pval:1,pval:2,pval:3,pval:4')]

pvaldiff <- pvalbeg - pvalend
pvaldiff[pvaldiff < 0] <- 0

cor(1 - pvalend[,'pval:2'], fulldata[mask,'prog:2'])
colMeans(fulldata[mask,split2char('prog:1,prog:2,prog:3,prog:4')])
