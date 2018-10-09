# Prepare data
setwd('~/Projects/Exploration/')
master = read.csv('pipeline_data/scdata/joint_data2.csv')
colnames(master)[colnames(master)=="X..grp"] <- "grp"
master$grp = factor(master$grp, levels = c(0,1), labels=c('F', 'S'))
master = master[ , c(1:5)]