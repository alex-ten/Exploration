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

main <- function(filter_switches=FALSE, 
                 remove_NA=FALSE) {
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
  
  # Set up some flags
  BASELINE <- FALSE
  CONVERT2LONG <- TRUE
  STRINGFACT <- FALSE
  NORMAL_SELF_REPORTS <- TRUE
  
  # Load data
  mdata <- read.csv('cmnr_data30cols.csv', check.names = FALSE)
  
  reltdata <- read.csv('cmnr_mdata.csv', check.names = FALSE)[split2char('relt:1,relt:2,relt:3,relt:4')]
  mdata <- cbind(mdata, reltdata)
  mdata <- mdata[, c(1:28, 36:39, 29:35)]
  
  xdata <- read.csv('cmnr_xdata.csv', check.names = FALSE)
  
  # Get switch trials (and remove pseudoswitch trials)
  if (filter_switches) {
    lmdata <- mdata[mdata$sw_act == 1, -1]
    lmdata <- lmdata[!(lmdata$current == lmdata$nxt), ]
  } else {
    lmdata <- mdata[, -1]
  }
  rm(mdata)
  
  # Discard incomplete cases
  if (remove_NA) {lmdata <- lmdata[complete.cases(lmdata), ]}
  
  # Convert categorical data to factors
  factor_col_names <- c('current', 'nxt')
  lmdata[, factor_col_names] <- lapply(lmdata[, factor_col_names], factor)
  
  # May be convert number-coded factors to explicit names
  if (STRINGFACT) {
    lmdata[, factor_col_names] <- lapply(lmdata[, factor_col_names], function(x) {
      mapvalues(x, from = as.character(1:4), to = split2char('1D,I1D,2D,R'))}
    )
  }
  
  # Square SC values
  ssc_col_names <- split2char('scsq:1,scsq:2,scsq:3,scsq:4')
  lmdata[ssc_col_names] <- lapply(lmdata[, split2char('sc:1,sc:2,sc:3,sc:4')], function(x) {x^2})
  
  # Create 'current' dummies
  dummies <- dummy_cols(lmdata$current)[c('.data_1', '.data_2', '.data_3', '.data_4')]
  colnames(dummies) <- split2char('currentd:1,currentd:2,currentd:3,currentd:4')
  lmdata <- cbind(lmdata, dummies)
  rm(dummies)
  
  # Match unique xdata rows with corresponding rows in mdata and join datasets
  inds <- match(lmdata$sid, xdata$sid) # get row indices of xdata for each mdata sid
  lmdata <- cbind(lmdata, xdata[inds, c(-1,-2,-3)]) # concatenate datasets
  rm(xdata)
  
  # May be normalize self reports
  if (NORMAL_SELF_REPORTS) {
    rownorm <- function(row) {
      if (min(row) == max(row)) {
        row / 10
      } else {
        (row-min(row))/(max(row)-min(row))
      }
    }
    for (varg in split2char('lrn:,int:,comp:,time:,prog:,rule:,lrn2:')) {
      normcols <- apply(lmdata[, grepl(varg, names(lmdata))], 1, rownorm)
      lmdata[, grepl(varg, names(lmdata))] <- t(normcols)
      rm(normcols)
    }
  }
  
  # Rearrange structure
  lmdata <- lmdata[c(1,2,4:7,8:35,39:74)]
  
  # Nominal data ready
  d <- lmdata
  
  # # Create ordinal data (alternative format)
  # d2 <- lmdata[1:5]
  # 
  # currmask <- as.logical(t(dummy(d2$current)))
  # unrolled <- rep(c(1,2,3,4), times=nrow(d2))
  # unrolled[currmask] <- 10
  # rolled <- matrix(unrolled, ncol=4, byrow=TRUE)
  # ranked <- t(apply(rolled, 1, rank_))
  # 
  # nxtmask <- as.logical(t(dummy(lmdata$nxt)))
  # unrolled <- as.vector(t(ranked))
  # d2$nxt <- unrolled[nxtmask]
  # 
  # for (varname in split2char('currentd,tpc,pc,rpc,pval,sc,relt,scsq,lrn,int,comp,time,prog,rule,lrn2')) {
  #   d2 <- cbind(d2, get_alts(lmdata, varname, !currmask))
  # }
}

d <- main()