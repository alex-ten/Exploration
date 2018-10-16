predict.mlogit_ <- function(object, newdata = NULL, returnData = FALSE, ...){
  # if no newdata is provided, use the mean of the model.frame
  if (is.null(newdata)) newdata <- mean(model.frame(object))
  # if newdata is not a mlogit.data, it is coerced below
  if (! inherits(newdata, "mlogit.data")){
    rownames(newdata) <- NULL
    lev <- colnames(object$probabilities)
    J <- length(lev)
    choice.name <- attr(model.frame(object), "choice")
    if (nrow(newdata) %% J)
      stop("the number of rows of the data.frame should be a multiple of the number of alternatives")
    attr(newdata, "index") <- data.frame(chid = rep(1:(nrow(newdata) %/% J ), each = J), alt = lev)
    attr(newdata, "class") <- c("mlogit.data", "data.frame")
    if (is.null(newdata[['choice.name']])){
      newdata[[choice.name]] <- FALSE
      newdata[[choice.name]][1] <- TRUE # probit and hev requires that one (arbitrary) choice is TRUE
    }
  }
  # if the updated model requires the use of mlogit.data, suppress all
  # the relevant arguments
  m <- match(c("choice", "shape", "varying", "sep",
               "alt.var", "chid.var", "alt.levels",
               "opposite", "drop.index", "id", "ranked"),
             names(object$call), 0L)
  if (sum(m) > 0) object$call <- object$call[ - m]
  # update the model and get the probabilities
  newobject <- update(object, start = coef(object, fixed = TRUE), data = newdata, iterlim = 0, print.level = 0)
  #    newobject <- update(object, start = coef(object), data = newdata, iterlim = 0, print.level = 0)
  
  result <- newobject$probabilities
  if (nrow(result) == 1){
    result <- as.numeric(result)
    names(result) <- colnames(object$probabilities)
  }
  if (returnData) attr(result, "data") <- newdata
  result
}