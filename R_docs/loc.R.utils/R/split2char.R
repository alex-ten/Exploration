#' @export
split2char <- function(s, sep=',') {
  unlist(strsplit(s, sep))
}