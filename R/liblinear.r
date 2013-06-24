modelDescriptions <- function(type=NA) {
    model_types = character(14)
    model_types[1+0] = "L2-regularized logistic regression (primal)"
    model_types[1+1] = "L2-regularized L2-loss support vector classification (dual)"
    model_types[1+2] = "L2-regularized L2-loss support vector classification (primal)"
    model_types[1+3] = "L2-regularized L1-loss support vector classification (dual)"
    model_types[1+4] = "support vector classification by Crammer and Singer"
    model_types[1+5] = "L1-regularized L2-loss support vector classification"
    model_types[1+6] = "L1-regularized logistic regression"
    model_types[1+7] = "L2-regularized logistic regression (dual)  for regression"
    model_types[1+11] = "L2-regularized L2-loss support vector regression (primal)"
    model_types[1+12] = "L2-regularized L2-loss support vector regression (dual)"
    model_types[1+13] = "L2-regularized L1-loss support vector regression (dual)"
    if (is.na(type)) {
        return(model_types)
    }else{
        return(model_types[type+1])
    }
}

#' @useDynLib sparsity sparsity_liblinearTrain 
#' @export
liblinear <- function(inputMatrix, labels, solver_type=5, cost=1, epsilon=0.001) {
    model = .Call('sparsity_liblinearTrain', PACKAGE = 'sparsity', inputMatrix, labels, solver_type, cost, epsilon)
    class(model) <- "liblinear"
    model
}

#'
predict.liblinear <- function(fit, newdata) {
    if (dim(newdata)[2] != fit$nr_features) {
        stop(paste0("Input data has wrong dimension. Expected columns: ",
                    fit$nr_features, " Given columns:", ncol(newdata)))
    }
    newdata %*% matrix(fit$w)
}

print.liblinear <- function(obj) {
    cat("---------------\n")
    cat("LIBLINEAR model","\n")
    cat("---------------\n")
    cat("\n")
    cat("       Model type:", modelDescriptions(obj$solver_type), "\n")
    cat("Number of weights:", length(obj$w), "\n")
    cat("Number of classes:", obj$nr_class, "\n")
    cat("\n")
}
