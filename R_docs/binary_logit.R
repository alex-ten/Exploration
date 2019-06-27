library(foreign)
library(ggplot2, quietly=TRUE)
library(reshape2, quietly=TRUE)
library(dplyr, quietly=TRUE)
library(plyr, quietly=TRUE)
library(car, quietly=TRUE)
library(caret, quietly=TRUE)
library(fastDummies, quietly=TRUE)
library(devtools, quietly=TRUE)

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
NORMAL_SELF_REPORTS <- TRUE

# Load data
mdata <- read.csv('cmnr_data30cols.csv', check.names = FALSE)

reltdata <- read.csv('cmnr_mdata.csv', check.names = FALSE)[split2char('relt:1,relt:2,relt:3,relt:4')]
mdata <- cbind(mdata, reltdata)
mdata <- mdata[, c(2:28, 36:39, 29:35)]

# xdata <- read.csv('cmnr_xdata.csv', check.names = FALSE)
d <- mdata[, -36]
rm(mdata)

# Convert categorical data to factors
factor_col_names <- c('current', 'nxt')
d[, factor_col_names] <- lapply(d[, factor_col_names], factor)

# May be convert number-coded factors to explicit names
if (STRINGFACT) {
  d[, factor_col_names] <- lapply(d[, factor_col_names], function(x) {
    mapvalues(x, from = as.character(1:4), to = split2char('1D,I1D,2D,R'))}
  )
}

# Square SC values
ssc_col_names <- split2char('scsq:1,scsq:2,scsq:3,scsq:4')
d[ssc_col_names] <- lapply(d[, split2char('sc:1,sc:2,sc:3,sc:4')], function(x) {x^2})

# Create 'current' dummies
dummies <- dummy_cols(d$current)[c('.data_1', '.data_2', '.data_3', '.data_4')]
colnames(dummies) <- split2char('currentd:1,currentd:2,currentd:3,currentd:4')
d <- cbind(d, dummies)
rm(dummies)

d <- d[c(1,2,4:35,38:45,36:37)]

# Sample from stay trials
set.seed(2)
switch_data <- d[d$sw_act == 1, ]
switch_data <- switch_data[!(switch_data$current == switch_data$nxt), ]
stay_data <- d[d$sw_act == 0, ]
stay_sample <- stay_data[sample(nrow(stay_data), nrow(switch_data), replace = FALSE), ]

d <- rbind(stay_sample, switch_data)

# Scramble switch and stay data
d <- d[sample(nrow(d)), ]
rm(stay_data, switch_data, stay_sample)

# Factorize switch column
d$sw_act <- factor(d$sw_act)

selector <- cbind(1:nrow(d), d$current)
cleand <- d[ ,c(1:6,42:43)]
for (varname in split2char('pct,pc,pcr,pval,relt')) {
  cleand <- cbind(cleand, d[vargrp(varname)][selector])
  names(cleand)[length(names(cleand))] <- varname
}
cleand$current <- factor(cleand$current)
# d <- cbind(d, d[vargrp('currentd')])
# names(d)[c(20,21,22,23)] <- split2char('cur_1,cur_2,cur_3,cur_4')
# d[, c(20,21,22,23)] <- factor(d[, c(20,21,22,23)])

# Fit linear model
model <- glm('sw_act ~ grp + trial + blkt + pc + pcr + pval + relt', 
             data = cleand, maxit=1000, family='binomial')
results <- summary(model)

# Report results
results
exp(coef(results))

# Evaluate prediction accuracy
probs <- predict(model, cleand[split2char('grp,trial,blkt,pc,pcr,pval,relt')], type='response')

# results
# exp(coef(results))
# confint(model)

y_hat <- as.integer(probs > .5)
confusionMatrix(data=factor(y_hat), reference=factor(d$sw_act))

summary(probs)
