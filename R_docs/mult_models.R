library(mlogit, quietly=TRUE)
library(nnet, quietly=TRUE)
library(devtools, quietly=TRUE)
document(file.path(getwd(),'loc.R.utils'))

pivot <- 1 # select reference category

source('prep_data.R')

out <- data.frame(
  form=character(),
  nvars=integer(),
  sing=logical(),
  loglik=numeric(),
  mfR2=numeric(),
  accuracy=numeric(), 
  AIC=numeric(),
  BIC=numeric(),
  xtime=numeric(),
  stringsAsFactors=FALSE)

altspec <- numeric()
for (varname in split2char('currentd,tord,pct,pcr,pc,pval,relt,sc,scsq,lrn,int,comp,time,prog,rule,lrn2')) {
  altspec <- append(altspec, grep(paste(varname,':', sep=''), names(d)))
}

# Convert to long format
longd <- mlogit.data(d, shape="wide", varying=altspec, choice='nxt', sep=':')
all_ivs <- split2char('currentd,tord,pct,pcr,pc,pval,relt,sc,scsq,lrn,int,comp,time,prog,rule,lrn2,grp,trial,blkt')
ivs <- split2char('currentd,tord,pct,pcr,pc,pval,relt,scsq')
BICN <- log(nrow(longd) / 4)
# Start loop
i <- 1
nbp <- length(ivs)
z <- vector()
for (j in 1:nbp) {
  combs <- combn(nbp, j)
  for (k in 1:ncol(combs)) {
    varinds <- combs[, k]
    
    if (any(varinds >= 17)) {
      indspec <- paste(ivs[varinds[varinds >= 17]], collapse='+')
    } else {indspec <- 1}
    
    if (any(varinds < 17)) {
      altspec <- paste(ivs[varinds[varinds < 17]], collapse='+')
    } else {altspec <- 1}
    
    f <- mFormula(as.formula(
        paste( c('nxt ~ ', altspec, ' | ', indspec), collapse='') ))
    print(f)
    t0 <- Sys.time()
    good <- tryCatch(model <- mlogit(f, data = longd, maxit=1000, reflevel=pivot),
                     error=function(e) {return(FALSE)})
    t1 <- Sys.time()
    
    if (is.logical(good)) {  
      out[i, ] <- c(
        paste(as.character(f)[2], as.character(f)[1], as.character(f)[3])[1],
        length(attr(terms(f), 'term.labels')),
        TRUE,
        NA_real_,
        NA_real_,
        NA_real_,
        NA_real_,
        NA_real_,
        round(as.numeric(t1-t0), digits=3)
      )
    } else {
      results <- summary(model)
      # Evaluate prediction accuracy
      nd <- cbind(longd[attr(terms(f), 'term.labels')], factor(longd$nxt))
      
      y_hat <- factor(apply(model$probabilities, 1, which.max))
      
      cm <- confusionMatrix(data=y_hat, reference=factor(d$nxt))
      t1 <- Sys.time()
      
      out[i, ] <- c(
        paste(as.character(f)[2], as.character(f)[1], as.character(f)[3])[1],
        length(attr(terms(f), 'term.labels')),
        FALSE,
        as.numeric(results$logLik),
        as.numeric(results$mfR2),
        as.numeric(cm$overall['Accuracy']),
        AIC(model),
        AIC(model, k=BICN),
        round(as.numeric(t1-t0), digits=3)
      )
    }
    # if (i %% 1000 == 0) {
    #   print(sprintf('i = %d, elapsed time: %.3f s', i, sum(as.numeric(out$xtime))), quote = FALSE)
    #   write.csv(x=out, file='remaining.csv')
    #   }
    i <- i + 1
  }
}

write.csv(x=out, file='smaller_ds.csv')

best <- mlogit(nxt ~ currentd + pcr + pval + relt + scsq | 1, data = longd, maxit=1000, reflevel=pivot)

y_hat <- factor(apply(best$probabilities, 1, which.max))

cm <- confusionMatrix(data=y_hat, reference=factor(d$nxt))
