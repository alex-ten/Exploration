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

df <- read.csv('complete-ds_09-11-2018.csv')
df <- df[, 2:ncol(df)]

# histogram(df$AIC, main='AIC', xlab='AIC', col='#008fd5', nint=30, lty="blank")
# histogram(df$BIC, main='BIC', xlab='BIC', col='#008fd5', nint=30, lty="blank")
# histogram(df$accuracy, main='Accuracy', xlab='Accuracy', col='#008fd5', xlim=c(.25, .5), ylim=c(0,1.1), nint=498, lty="blank")
# histogram(df$loglik, main='Log likelihood', xlab='log L', col='#008fd5', nint=300, lty="blank", ylim=c(0,2.9))

# first_aic_crit <- 5000
# identical(df[df$AIC < first_aic_crit, ], df[df$BIC < first_aic_crit, ])
# df <- df[df$AIC < first_aic_crit, ]

# histogram(df$AIC, main='AIC', xlab='AIC', col='#008fd5', nint=300, lty="blank", ylim=c(0,1.5))
# histogram(df$BIC, main='BIC', xlab='BIC', col='#008fd5', nint=300, lty="blank", ylim=c(0,1.1))
# histogram(df$accuracy, main='Accuracy', xlab='Accuracy', col='#008fd5', ylim=c(0, 1.5), xlim=c(.36, .5), nint=274, lty="blank")
# histogram(df$loglik, main='Log likelihood', xlab='log L', col='#008fd5', nint=300, lty="blank", ylim=c(0,1.5))

# as.character(df[1, 'AIC'])
# qplot(df$AIC, df$BIC)
# 
head(df[order(df$AIC), c('form', 'nvars', 'loglik', 'accuracy', 'AIC', 'BIC')], 10)
head(df[order(df$BIC), c('form', 'nvars', 'loglik', 'accuracy', 'AIC', 'BIC')], 10)
head(df[order(df$accuracy, decreasing=T), c('form', 'nvars', 'loglik', 'accuracy', 'AIC', 'BIC')], 10)

varmat <- as.data.frame(do.call(rbind, lapply(as.character(df$form), dummify)))
colnames(varmat) <- ivs

df <- cbind(df, varmat)

form_AIC <- formula(paste('AIC ~', paste(ivs, collapse=' + ')))
form_BIC <- formula(paste('BIC ~', paste(ivs, collapse=' + ')))
form_acc <- formula(paste('accuracy ~', paste(ivs, collapse=' + ')))

fit_aic <- lm(form_AIC, data=df)
fit_bic <- lm(form_BIC, data=df)
fit_acc <- lm(form_acc, data=df)

res_aic <- summary(fit_aic)
res_bic <- summary(fit_bic)
res_acc <- summary(fit_acc)

coef_aic <- as.data.frame(coefficients(res_aic))
coef_bic <- as.data.frame(coefficients(res_bic))
coef_acc <- as.data.frame(coefficients(res_acc))

coef_aic[order(coef_aic$Estimate), ]
coef_bic[order(coef_bic$Estimate), ]
coef_acc[order(coef_acc$Estimate, decreasing = T), ]


## Coefficient weights (by AIC)
by_aic <- df[order(df$AIC, decreasing=F), c('form', 'nvars', 'loglik', 'accuracy', 'AIC', 'BIC')]
by_aic$deltaBest <- by_aic$AIC - min(by_aic$AIC)
by_aic$rlike <- exp(-by_aic$deltaBest/2)
by_aic$ER <- exp(0) / by_aic$rlike
by_aic$w <- by_aic$rlike / sum(by_aic$rlike)

varbools <- varmat[order(df$AIC, decreasing=FALSE), ]
varbools <- cbind(integer(nrow(varmat))+1, varbools)
names(varbools)[1] <- 'Intercept'
predw <- sapply(varbools, function(x) { sum(by_aic$w[as.logical(x)]) } )

histogram(by_aic$deltaBest, main='Delta scores', xlab='Delta_i', col='#008fd5', 
          nint=300, lty="blank", ylim=c(0,5))

barplot(height = predw[order(predw, decreasing=T)], main='Parameter weights',
        names.arg = names(predw[order(predw, decreasing=T)]), las=2)


weights <- predw[c(1,1,1,2:16,17,17,17,18,18,18,19,19,19)]

