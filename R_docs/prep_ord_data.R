library(foreign)
library(ggplot2, quietly=TRUE)
library(reshape2, quietly=TRUE)
library(dplyr, quietly=TRUE)
library(plyr, quietly=TRUE)
library(car, quietly=TRUE)
library(caret, quietly=TRUE)
library(fastDummies, quietly=TRUE)
library(devtools, quietly=TRUE)

document(file.path(getwd(),'loc.R.utils'))


# Set up some flags
BASELINE <- FALSE
CONVERT2LONG <- TRUE
STRINGFACT <- FALSE
NORMAL_SELF_REPORTS <- TRUE

# Load data
mdata <- read.csv('cmnr_mdata.csv', check.names = FALSE)
mdata <- mdata[, -length(colnames(mdata))]
xdata <- read.csv('cmnr_xdata.csv', check.names = FALSE)

# Get switch trials (and remove pseudoswitch trials)
lmdata <- mdata[mdata$sw_act == 1, -1]
rm(mdata)
lmdata <- lmdata[!(lmdata$current == lmdata$nxt), ]

# Discard incomplete cases
lmdata <- lmdata[complete.cases(lmdata), ]

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
ssc_col_names <- split2char('ssc:1,ssc:2,ssc:3,ssc:4')
lmdata[ssc_col_names] <- lapply(lmdata[, split2char('sc:1,sc:2,sc:3,sc:4')], function(x) {x^2})

# Create 'current' dummies
dummies <- dummy_cols(lmdata$current)[c('.data_1', '.data_2', '.data_3', '.data_4')]
colnames(dummies) <- split2char('currentd:1,currentd:2,currentd:3,currentd:4')
lmdata <- cbind(lmdata, dummies)
rm(dummies)

# Match unique xdata rows with corresponding rows in mdata and join datasets
inds <- match(lmdata$sid, xdata$sid) # get row indices of xdata for each mdata sid
lmdata <- cbind(lmdata, xdata[inds, c(-1,-2,-3)]) # concatenate datasets

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
lmdata <- lmdata[c(1,2,4,5,6,7,8:23,26:61)]

