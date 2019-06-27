library(dplyr)
library(ggplot2)
library(tidyr)
library(compute.es)
library(effects)
library(multcomp)

# Test effects of group and condition on LRN-PC and LRN-DPC correlations (on individual level)
data <- read.csv('lrn_pc_dpc_data.csv', check.names=F)

data$grp <- factor(data$grp, levels = c(0,1), labels = c('F','S'))
data$cnd <- factor(data$cnd, levels = c(0,1), labels = c('i-','i+'))

data$grp_cnd <- with(data, interaction(grp,  cnd))

kruskal.test(rpc ~ cnd, data = data[data$range >= 6, ])
kruskal.test(rdpc ~ cnd, data = data[data$range >= 6, ])

# Test effects of group, condition and task on relationship between LRN and PC on group level
data <- read.csv('LRNraw_PCraw.csv', check.names=F)

tvar <- data.frame(matrix(rep(1:4, each=nrow(data)), nrow=nrow(data)))
names(tvar) <- c('t.1', 't.2', 't.3', 't.4')
data <- cbind(data, tvar)

data <- reshape(data = data,
        direction = "long",
        varying = list(c('pc1','pc2','pc3','pc4'), c('lrn1','lrn2','lrn3','lrn4'), c('t.1','t.2','t.3','t.4')),
        v.names = c('pc', 'lrn', 'tid'))

data$grp <- factor(data$grp)
data$cnd <- factor(data$cnd)
data$tid <- factor(data$tid)

data$grp <- relevel(data$grp, ref=0)
data$cnd <- relevel(data$cnd, ref=0)
data$tid <- relevel(data$tid, ref=1)

regr <- lm(lrn ~ pc + grp + cnd + tid + pc*grp + pc*cnd + pc*tid, data=data)
summary(regr)


# Self-report analyses
data <- read.csv('../pipeline_data/longdata/long3.csv')
data$sid <- factor(data$sid)
data$tord <- factor(data$tord)

# Progress rating
regr <- lm(nprog ~ pc_grand + dpc + grp + cnd + tid + pc_grand*grp + pc_grand*cnd + pc_grand*tid + dpc*grp + dpc*cnd + dpc*tid, data=data)
summary(regr)

# Learnability 2 rating
regr <- lm(nlrn2 ~ (pc_grand + dpc + nrule) * (grp + cnd + tid), data=data)
summary(regr)

# Interest rating and freetime
regr <- lm(nint ~ (pc_grand + dpc + freetime) * (grp + cnd + tid), data=data)
summary(regr)

# Interest rating and rule
regr <- lm(lrn2 ~ (pc_grand + dpc + nrule) * (grp + cnd + tid), data=data)
summary(regr)


# Zoom in on 2D and R
data <- data[data$tid=='4_R' | data$tid=='1_1D', ]
data['task'] <- data$tid=='4_R'
data$task <- factor(data$task, levels = c(F,T), labels = c('not_R','R'))
data$tid <- relevel(data$tid, ref='4_R')

regr <- lm(nint ~ (pc_grand+dpc+nlrn+nrule+nprog+ntime+nlrn2)*(tid+grp), data=data)
summary(regr)

regr <- lm(nint ~ tid*(grp + cnd), data=data)
summary(regr)

regr <- lm(freetime ~ (pc_grand+dpc+nlrn+nrule+nprog+nint)*grp*tid, data=data)
summary(regr)
