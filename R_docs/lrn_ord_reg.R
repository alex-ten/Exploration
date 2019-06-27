require(foreign)
require(ggplot2)
require(MASS)
require(Hmisc)
require(reshape2)

fit_ord_reg <- function(f, data, summary=FALSE) {
  # fit ordinal regression
  om <- polr(f, data = data, Hess=TRUE)
  summary(om)
  
  # Coefficient p-values
  ctable <- coef(summary(om))
  p <- pnorm(abs(ctable[, "t value"]), lower.tail = FALSE) * 2
  ctable <- cbind(ctable, "p value" = round(p, 5))
  ctable <- cbind(ctable, "exp(B)" = round(exp(ctable[, 1]), 5))
  fulltable <- ctable[c(-7,-8,-9), ]
  
  # Confidence intervals
  ci <- confint(om)
  fulltable <- cbind(fulltable, "CI 2.5%" = round(ci[, 1], 5))
  fulltable <- cbind(fulltable, "CI 97.5%" = round(ci[, 2], 5))
  
  # Odds ratios
  expB <- exp(fulltable[, 1])
  fulltable <- cbind(fulltable, "Odds ratio" = round(expB, 5))
  ORci <- exp(ci)
  fulltable <- cbind(fulltable, "ORCI 2.5%" = round(ORci[, 1], 5))
  fulltable <- cbind(fulltable, "ORCI 97.5%" = round(ORci[, 2], 5))
  
  print(ctable)
  # print(fulltable)
  
  if (summary) {print(summary(om))}
}

d <- read.csv('lrn_reg_data.csv')[, -1]
d$tid <- factor(d$tid)

# Get ranks by group data
for (grp in c(0,1)) {
  for (cnd in c(0,1)) {
    print(sprintf('Group: %d, condition: %d', grp, cnd), quote = FALSE)
    sd = d[d$grp == grp & d$cnd == cnd, ]
    # crosstab = data.frame(ftable(xtabs(~ tid + lrn, data = sd)))
    # write.csv(x=crosstab,
    #           file=sprintf('lrnranks_by_task_g%dc%d.csv', grp, cnd),
    #           quote=FALSE)
    
    # Any effect of task on LRN?
    # for (refl in c(1,2,3,4)) {
    #   sd <- within(sd, tid <- relevel(tid, ref = refl))
    #   fit_ord_reg(factor(lrn) ~ tid, sd, summary=FALSE)
    # }
    fit_ord_reg(factor(lrn) ~  tid + ord + pc + dpc, sd)
  }
}
