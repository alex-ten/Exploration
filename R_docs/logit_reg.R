library(mlogit, quietly=TRUE)
library(nnet, quietly=TRUE)

BASELINE <- FALSE

source('prep_data.R')

altspec <- numeric()
for (varname in split2char('currentd,pc,pval,relt,sc,scsq,lrn,int,comp,time,prog,rule,lrn2')) {
  altspec <- append(altspec, grep(paste(varname,':', sep=''), names(d)))
}

# Baselines
if (BASELINE) {
  # Baseline 1: Accuracy of random predictions
  y_rand <- matrix(sample(unique(d$nxt), length(d$nxt)*100, replace=TRUE), nrow=length(d$nxt))
  matches <- y_rand == d$nxt
  acc <- mean(colSums(matches)/dim(matches)[1])
  print(paste('Random selection accuracy =', acc), quote = FALSE)
  
  # Baseline 2: Easiest unlearned
  unrolled <- as.vector(t(d[vargrp4('pc')]))
  unlearned <- !(as.vector(t(d[vargrp4('pval')])) < .01)
  unrolled[!unlearned] <- NA
  rolled <- matrix(unrolled, ncol=4, byrow=TRUE)
  allNA <- apply(rolled, 1, function(row){all(is.na(row))})
  y_hat <- apply(rolled[!allNA, ], 1, which.max)
  matches <- y_hat == d$nxt[!allNA]
  acc <- mean(matches)
  print(paste('Baseline challenge predictions accuracy =', acc), quote = FALSE)
}

# d <- d[d$grp == 1, ]

# Convert to long format
longd <- mlogit.data(d, shape="wide", varying=altspec, choice='nxt', sep=':')

# Get linear formula
t0 <- Sys.time()
cond_formula <- mFormula(nxt ~ currentd + pc + pval + scsq + relt + lrn + int + rule + prog + time + lrn2 + comp | blkt + trial + grp)

# Fit linear model
pivot <- 1 # select reference category
model <- mlogit(cond_formula, data = longd, maxit=1000, reflevel=pivot)
results <- summary(model)

# Evaluate prediction accuracy
nd <- cbind(longd[attr(terms(cond_formula), 'term.labels')], factor(longd$nxt))
probs <- predict.mlogit_(model, nd)
y_hat <- factor(apply(probs, 1, which.max))

cm <- confusionMatrix(data=y_hat, reference=factor(d$nxt))
t1 <- Sys.time()
t1-t0
results
exp(coef(results))
cm$table
cm$overall['Accuracy']

# Get marginal predictions
# z0 <- with(longd, data.frame(pc=tapply(pc, index(model)$alt, mean),
#                              pval=tapply(pval, index(model)$alt, mean),
#                              relt=tapply(relt, index(model)$alt, mean),
#                              lrn=tapply(lrn, index(model)$alt, mean),
#                              currentd=c(1,0,0,0),
#                              trial=tapply(trial, index(model)$alt, mean))
#           )
# 
# effects(model, covariate='pval', data=z0)
# effects(model, covariate='pc', data=z0)
# effects(model, covariate='relt', data=z0)
# effects(model, covariate='lrn', data=z0)

