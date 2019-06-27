library(ez)
library(reshape2)

df <- read.csv('pre_post.csv', check.names = FALSE)[, c(2,3,4,5)]

longdf <- melt(df, id = c('grp', 'sid'), measured = c('pre', 'post'))
longdf$grp <- factor(longdf$grp)

colnames(longdf) <- c('grp','sid', 'time', 'score')
options(contrasts=c("contr.sum","contr.poly"))
ezANOVA(data = longdf,
        dv = score,
        within = time,
        between = grp,
        wid = sid,
        type = 3)
