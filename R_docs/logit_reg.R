library(foreign)
library(nnet, quietly=TRUE)
library(ggplot2, quietly=TRUE)
library(reshape2, quietly=TRUE)
library(dplyr, quietly=TRUE)
library(plyr, quietly=TRUE)
library(car, quietly=TRUE)
library(caret, quietly=TRUE)
library(fastDummies, quietly=TRUE)
library(devtools, quietly=TRUE)
library(mlogit, quietly=TRUE)

document(file.path(getwd(),'loc.R.utils'))

# Set up some flags
BASELINE <- FALSE
CONVERT2LONG <- TRUE
STRINGFACT <- FALSE
NORMAL_SELF_REPORTS <- TRUE

# Load data
mdata <- read.csv('cmnr_mdata.csv', check.names = FALSE)
xdata <- read.csv('cmnr_xdata.csv', check.names = FALSE)

# mdata <- mdata[mdata$grp==0, ]
# xdata <- xdata[xdata$grp==0, ]

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

# Add variables
rownorm <- function(row) {scale(row, scale=FALSE)}
lmdata$ruleR <- apply(lmdata[, split2char('rule:1,rule:2,rule:3,rule:4')], 1, rownorm)[4, ]

# May be normalize self reports
if (NORMAL_SELF_REPORTS) {
  for (varg in split2char('lrn:,int:,comp:,time:,prog:,rule:,lrn2:')) {
    normcols <- apply(lmdata[, grepl(varg, names(lmdata))], 1, rownorm)
    lmdata[, grepl(varg, names(lmdata))] <- t(normcols)
    rm(normcols)
  }
}

# Rearrange structure
lmdata <- lmdata[c(1,2,4,5,6,7,63,8:23,27:62)]

# May be convert to long format
if (CONVERT2LONG) {
  longlmdata <- mlogit.data(lmdata, shape="wide", varying=8:59, choice='nxt', sep=':') 
}

# Get linear formula
cond_formula <- mFormula(nxt ~ pc+p+ssc+relt+lrn+int+rule+prog+time+lrn2+comp|blkt+trial+grp)
cond_formula <- mFormula(nxt ~ pc+p+ssc+relt+lrn+prog+time|grp)
cond_formula <- mFormula(nxt ~ pc+p+ssc+relt+lrn|grp)



# Fit linear model
pivot <- 4 # select reference category
model <- mlogit(cond_formula, data = longlmdata, maxit=1000, reflevel=pivot)
results <- summary(model)

# Evaluate prediction accuracy
nd <- cbind(longlmdata[attr(terms(cond_formula), 'term.labels')], factor(longlmdata$nxt))
y_hat <- factor(apply(predict.mlogit_(model, nd), 1, which.max))

cm <- confusionMatrix(data=y_hat, reference=lmdata$nxt)
results
exp(coef(results))

# Get marginal predictions
z <- with(longlmdata, data.frame(pc=tapply(pc, index(model)$alt, mean),
                                 p=tapply(p, index(model)$alt, mean),
                                 ssc=tapply(ssc, index(model)$alt, mean),
                                 relt=tapply(relt, index(model)$alt, mean),
                                 lrn=tapply(lrn, index(model)$alt, mean))
          )
effects(model, covariate='pc', data=z)

# Baselines
if (BASELINE) {
  # Baseline 1: Accuracy of random predictions
  y_rand <- matrix(sample(unique(lmdata$nxt), length(lmdata$nxt)*100, replace=TRUE), nrow=length(lmdata$nxt))
  matches <- y_rand == lmdata$nxt
  acc <- mean(colSums(matches)/dim(matches)[1])
  print(paste('Random selection accuracy =', acc), quote = FALSE)
  
  # Baseline 2: Accuracy of constant predictions
  y_const <- matrix(rep(unique(lmdata$nxt), length(lmdata$nxt)), ncol = 4, byrow = TRUE)
  matches <- y_const == lmdata$nxt
  acc <- colSums(matches)/dim(matches)[1]
  acc <- gettext(paste(unique(lmdata$nxt), acc, sep='='))
  cat('Constant selection accuracy:\n', acc)
}
