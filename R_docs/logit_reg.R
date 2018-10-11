require(foreign)
require(nnet)
require(ggplot2)
require(reshape2)
require(dplyr)
require(plyr)
require(car)
require(caret)
require(dummies)

# Load data
mdata <- read.csv('mnr_mdata.csv')
xdata <- read.csv('mnr_xdata.csv')

# Get switch trials (and remove pseudoswitch trials)
lmdata <- mdata[mdata$sw_act == 1, -1]
lmdata <- lmdata[!(lmdata$t0 == lmdata$t1)]

# Discard incomplete cases
# lmdata <- lmdata[complete.cases(lmdata), ]

# Convert categorical data to factors
factor_col_names <- c('t0', 't1')
lmdata[, factor_col_names] <- lapply(lmdata[, factor_col_names], factor)
lmdata[, factor_col_names] <- lapply(lmdata[, factor_col_names], function(x) { 
    mapvalues(x, from = as.character(1:4), to = split2char('1D,I1D,2D,R'))}
  )

# Square SC values
ssc_col_names <- split2char('ssc1,ssc2,ssc3,ssc4')
lmdata[ssc_col_names] <- lapply(lmdata[, split2char('sc1,sc2,sc3,sc4')], function(x) {x^2})

# Create t0 dummies
lmdata <- cbind(lmdata, dummy(lmdata$t0, sep='_'))

# Match unique xdata rows with corresponding rows in mdata and join datasets
inds <- match(lmdata$sid, xdata$sid) # get row indices of xdata for each mdata sid
lmdata = cbind(lmdata, xdata[inds, c(-1,-2,-3)]) # concatenate datasets

# Select training data
predictors = split2char('grp,trial,blkt,lmdata_1D,lmdata_I1D,lmdata_2D,lmdata_R,p1,p2,p3,p4,ssc1,ssc2,ssc3,ssc4,lrn_1,lrn_2,lrn_3,lrn_4')
linear_formula <- formula(paste("t1 ~ ", paste(predictors, collapse=" + ")))

# Fit linear model
model <- multinom(linear_formula, data = lmdata)

# Evaluate prediction accuracy
y_hat <- predict(model, lmdata[predictors])
confusionMatrix(y_hat, lmdata$t1)
