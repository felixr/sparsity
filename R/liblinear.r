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


#' Creates a LIBLINEAR problem data structure from a sparse matrix and labels.
#'
#' @useDynLib sparsity sparsity_createProblemInstance
#' @export
liblinear.new <- function(inputMatrix, labels) {
    p = .Call('sparsity_createProblemInstance', PACKAGE = 'sparsity', inputMatrix, labels)
    class(p) <- "liblinearProblem"
    p
}
#' Trains a model using LIBLINEAR
#'
#' @export
liblinear <- function(data, labels=NULL, solver_type=5, cost=1, epsilon=0.001) {
    UseMethod("liblinear", data)
}

#' Trains a model using LIBLINEAR
#' 
#' @method liblinear liblinearProblem 
#' @S3method liblinear liblinearProblem 
#' @useDynLib sparsity sparsity_liblinearTrain 
liblinear.liblinearProblem <- function(data, labels, solver_type=5, cost=1, epsilon=0.001, quiet=FALSE) {
    model = .Call('sparsity_liblinearTrain', PACKAGE = 'sparsity', 
                  data, solver_type, cost, epsilon, quiet)

    class(model) <- "liblinear"
    model
}

#' @method liblinear dgCMatrix
#' @S3method liblinear dgCMatrix
#' @useDynLib sparsity sparsity_liblinearTrain 
liblinear.dgCMatrix <- function(data, labels, solver_type=5, cost=1, epsilon=0.001, quiet=FALSE) {
    problem = liblinear.new(data, labels)
    model = .Call('sparsity_liblinearTrain', PACKAGE = 'sparsity', problem, solver_type, cost, epsilon, quiet)
    rm(problem)
    class(model) <- "liblinear"
    model
}

#' @method predict liblinear 
#' @S3method predict liblinear 
predict.liblinear <- function(fit, newdata) {
    if (dim(newdata)[2] != fit$nr_features) {
        warning(paste0("Input data has wrong dimension. Expected columns: ",
                    fit$nr_features, " Given columns:", ncol(newdata)))
        newdata[,1:fit$nr_features] %*% matrix(fit$w)
    }else{
        newdata %*% matrix(fit$w)
    }
}

#' @method print liblinear 
#' @S3method print liblinear 
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
