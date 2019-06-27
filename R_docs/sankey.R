library(RColorBrewer)
library(riverplot)
library(stringr)

# edges = data.frame(N1 = paste0(rep(LETTERS[1:4], each = 4), rep(1:5, each = 16)),
#                      N2 = paste0(rep(LETTERS[1:4], 4), rep(2:6, each = 16)),
#                      Value = runif(80, min = 2, max = 5) * rep(c(1, 0.8, 0.6, 0.4, 0.3), each = 16),
#                      stringsAsFactors = F)
# 
# edges = edges[sample(c(TRUE, FALSE), nrow(edges), replace = TRUE, prob = c(0.8, 0.2)),]

edges = read.csv('flowdata.csv', check.names=FALSE)
head(edges)
edges$N1 <- as.character(edges$N1)
edges$N2 <- as.character(edges$N2)

nodes = data.frame(ID = unique(c(edges$N1, edges$N2)), stringsAsFactors = F)
nodes$x = as.integer(substr(nodes$ID, 2, 2))
# # nodes$y = as.integer(sapply(substr(nodes$ID, 1, 1), charToRaw)) - 65
# 
# rownames(nodes) = nodes$ID
# head(nodes)
# 
# palette = paste0(brewer.pal(4, "Set1"), "60")
# styles = lapply(nodes$y, function(n) {
#     list(col = palette[n+1], lty = 0, textcol = "black")
# })
# names(styles) = nodes$ID


rp <- list(nodes = nodes, edges = edges, styles = styles)
class(rp) <- c(class(rp), "riverplot")
plot(rp, plot_area = 0.95, yscale=0.06)
