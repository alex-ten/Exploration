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
library(HistogramTools)
document(file.path(getwd(),'loc.R.utils'))

dummify <- function(r) {
  dum <- integer(length(ivs))
  r <- strsplit(strsplit(r, ' ~ ', fixed=TRUE)[[1]][2], ' | ', fixed=TRUE)[[1]]
  altspec <- strsplit(r[1], ' + ', fixed=TRUE)[[1]]
  indspec <- strsplit(r[2], ' + ', fixed=TRUE)[[1]]
  inds <- na.omit(c(pmatch(altspec, ivs), pmatch(indspec, ivs)))
  dum[inds] <- 1
  return(dum)
}

# Total models attempted: 524,287
ivs <- split2char('currentd,tord,pct,pcr,pc,pval,relt,sc,scsq,lrn,int,comp,time,prog,rule,lrn2,grp,trial,blkt')

df <- read.csv('complete-ds_31-10-2018.csv')
df <- df[, 3:ncol(df)]

# histogram(df$AIC, main='AIC', xlab='AIC', col='#008fd5')
# histogram(df$BIC, main='BIC', xlab='BIC', col='#008fd5')
# histogram(df$accuracy, main='Accuracy', xlab='Accuracy', col='#008fd5', xlim=c(.25, .5))
# histogram(df$loglik, main='Log likelihood', xlab='log L', col='#008fd5')

# first_aic_crit <- 5000
# identical(df[df$AIC < first_aic_crit, ], df[df$BIC < first_aic_crit, ])
# df <- df[df$AIC < first_aic_crit, ]

# histogram(df$AIC, main='AIC', xlab='AIC', col='#008fd5')
# histogram(df$BIC, main='BIC', xlab='BIC', col='#008fd5')
# histogram(df$accuracy, main='Accuracy', xlab='Accuracy', col='#008fd5', xlim=c(.36, .48))
# histogram(df$loglik, main='Log likelihood', xlab='log L', col='#008fd5')
# histogram(df$nvars, main='Number of variables', xlab='N vars', col='#008fd5')

# as.character(df[1, 'AIC'])
# qplot(df$AIC, df$BIC)

# head(df[order(df$AIC), c('form', 'nvars', 'loglik', 'accuracy', 'AIC', 'BIC')], 10)
# head(df[order(df$BIC), c('form', 'nvars', 'loglik', 'accuracy', 'AIC', 'BIC')], 10)
# tail(df[order(df$accuracy), c('form', 'nvars', 'loglik', 'accuracy', 'AIC', 'BIC')], 10)

varmat <- as.data.frame(do.call(rbind, lapply(as.character(df$form), dummify)))
colnames(varmat) <- ivs

df <- cbind(df, varmat)

