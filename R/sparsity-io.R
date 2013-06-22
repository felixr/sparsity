#' Write a sparse matrix to a file in  SVMlight format
#'
#' @param sparseMatrix 
#' @param labelVector 
#' @param file 
#' @export
#' @useDynLib sparsity sparsity_writesvmlight 
write.svmlight <- function(sparseMatrix, labelVector, file) {
    if (!inherits(sparseMatrix, "dgCMatrix")) {
        stop("Matrix should have type dgCMatrix")
    }
    if (dim(sparseMatrix)[1] != length(labelVector)) {
        stop("Row number of matrix must match.")
    }
    .Call('sparsity_writeSvmLight', PACKAGE = 'sparsity',
          sparseMatrix, labelVector, file)
}

#' Read sparse matrix from file in SVMlight format
#'
#'
#' @export 
#' @useDynLib sparsity sparsity_readsvmlight  
read.svmlight <- function(file) {
    if (!file.exists(file)) {
        stop("File does not exist.")
    }
    .Call('sparsity_readSvmLight', PACKAGE = 'sparsity', file)
}
