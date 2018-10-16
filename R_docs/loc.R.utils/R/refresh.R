#' @export
lut.refresh <- function() {
  print(sys.frame())
  rm(list=ls())
}
