library(ez)
library(multcomp)
library(dplyr)
library(nlme)

# Load data
data <- read.csv('../pipeline_data/clear_data/long_ntm_data.csv')
data$sid <- factor(data$sid)
data$tord <- factor(data$tord)
data$ntm <- factor(data$ntm)
data$grp <- factor(data$grp)

# Reshape to wide format
data <- reshape(data, idvar=c('sid', 'grp', 'cnd'), timevar='tid', direction='wide')

# # Effects on PC during training
summary(aov(pc_first ~ grp*tid + Error(sid), data=data)) # Mixed 3-way ANOVA
# summary(aov(pc_first ~ grp*ntm*tid*fam*tord + Error(sid), data=data)) # Mixed 3-way ANOVA
# 
# aggregate(pc_first ~ tid, data=data[data$fam=='Bear', ], mean)
# aggregate(pc_first ~ tid, data=data[data$fam=='Bunny', ], mean)
# aggregate(pc_first ~ tid, data=data[data$fam=='Green', ], mean)
# aggregate(pc_first ~ tid, data=data[data$fam=='Squid', ], mean)

# Effects on self-reports
y = 'nlrn2'
summary(aov(formula(sprintf('%s ~ grp*ntm*tid + Error(sid)', y)), data=data)) # Mixed 3-way ANOVA
aggregate(formula(sprintf('%s ~ grp', y)), data=data, mean)
aggregate(formula(sprintf('%s ~ ntm', y)), data=data, mean)
aggregate(formula(sprintf('%s ~ tid', y)), data=data, mean)
aggregate(formula(sprintf('%s ~ tid*grp', y)), data=data, mean)

aggregate(prog ~ grp, data=data[data$tid=='1_1D',  ], mean)
aggregate(prog ~ grp, data=data[data$tid=='2_I1D', ], mean)
aggregate(prog ~ grp, data=data[data$tid=='3_2D',  ], mean)
aggregate(prog ~ grp, data=data[data$tid=='4_R',   ], mean)
