library(mlogit, quietly=TRUE)
library(nnet, quietly=TRUE)
library(devtools, quietly=TRUE)
document(file.path(getwd(),'loc.R.utils'))

pivot <- 1 # select reference category

source('prep_data.R')

altspec <- numeric()
for (varname in split2char('currentd,tord,pct,pcr,pc,pval,relt,sc,scsq,lrn,int,comp,time,prog,rule,lrn2')) {
  altspec <- append(altspec, grep(paste(varname,':', sep=''), names(d)))
}

# Convert to long format
longd <- mlogit.data(d, shape="wide", varying=altspec, choice='nxt', sep=':')
ivs <- split2char('currentd,tord,pct,pcr,pc,pval,relt,sc,scsq,lrn,int,comp,time,prog,rule,lrn2,grp,trial,blkt')


varinds <- 1:19
varinds <- varinds[-8]
    
if (any(varinds >= 17)) {
  indspec <- paste(ivs[varinds[varinds >= 17]], collapse='+')
} else {indspec <- 1}

if (any(varinds < 17)) {
  altspec <- paste(ivs[varinds[varinds < 17]], collapse='+')
} else {altspec <- 1}
    
f <- mFormula(as.formula(
  paste( c('nxt ~ ', altspec, ' | ', indspec), collapse='') ))

nosc <- mlogit(f, data = longd, maxit=1000, reflevel=pivot)
weighted <- coefficients(nosc) * weights
wnosc <- update(nosc, start = weighted, data = longd, iterlim = 0, print.level = 0)


nd <- cbind(longd[attr(terms(f), 'term.labels')], factor(longd$nxt))

y_hat1 <- factor(apply(nosc$probabilities, 1, which.max))
y_hat2 <- factor(apply(wnosc$probabilities, 1, which.max))

cm1 <- confusionMatrix(data=y_hat1, reference=factor(d$nxt))
cm2 <- confusionMatrix(data=y_hat2, reference=factor(d$nxt))


altlongd$pval <- (longd$pval)^(1/3)

f <- as.character(df$form[order(df$AIC, decreasing=F)][1])

best <- mlogit(formula(f), longd, reflevel=1)
trans <- mlogit(formula(f), altlongd, reflevel=1)
AIC(best)
AIC(trans)

y_hat_best  <- factor(apply(best$probabilities, 1, which.max))
y_hat_trans <- factor(apply(trans$probabilities, 1, which.max))

cm_best  <- confusionMatrix(data=y_hat_best, reference=factor(d$nxt))
cm_trans <- confusionMatrix(data=y_hat_trans, reference=factor(d$nxt))

cm_trans
