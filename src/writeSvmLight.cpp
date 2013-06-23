// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <Rcpp.h>
#include <fstream>
#include <iostream>

using namespace Rcpp;

//' Writes a sparse matrix to a SVMlight compatible file
//'
//' @param inputMatrix sparse matrix
//' @param labels list of numeric labels for each row in the matrix 
//' @param fileName  output file name
//' @return list with debug information
// [[Rcpp::export]]
List writeSvmLight(Eigen::MappedSparseMatrix<double> inputMatrix, 
                        NumericVector labels, std::string fileName){
    typedef Eigen::MappedSparseMatrix<double> SpCMat;
    typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpRMat;
    typedef Eigen::Triplet<double> Triplet; 

    std::vector<Triplet> triplets;
    triplets.reserve(inputMatrix.nonZeros());

    // create list of triplets
    for (int k=0; k < inputMatrix.outerSize(); ++k) { // foreach col
        for (SpCMat::InnerIterator it(inputMatrix,k); it; ++it) { // foreach row
            triplets.push_back(Triplet(it.row(), it.col(), it.value()));
        }
    }

    // create sparse matrix with RowMajor
    SpRMat Y(inputMatrix.rows(), inputMatrix.cols());
    Y.setFromTriplets(triplets.begin(), triplets.end());

    // open output file
    std::ofstream out;
    out.open(fileName.c_str());

    // now outer corresponds to rows (RowMajor)
    for (int k=0; k < Y.outerSize(); ++k) { // foreach row
        out << labels[k];
        for (SpRMat::InnerIterator it(Y,k); it; ++it) { // foreach col
            out << " " << it.col() << ":" << it.value();
        }
        out << std::endl;
    }
    out.close();

    return List::create( 
            _["fileName"]  = fileName, 
            _["entriesWritten"] = Y.nonZeros() 
            ) ;
}

