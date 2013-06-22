// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <Rcpp.h>
#include <fstream>
#include <iostream>

using namespace Rcpp;

// [[Rcpp::export]]
SEXP writeSvmLight(SEXP sparseMatrix, SEXP labelVector, SEXP fname){
    typedef Eigen::MappedSparseMatrix<double> SpCMat;
    typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpRMat;
    typedef Eigen::Triplet<double> Triplet; 

    SpCMat X(Rcpp::as<SpCMat>(sparseMatrix));
    std::string fileName(as<std::string>(fname));
    NumericVector labels(as<NumericVector>(labelVector));

    std::vector<Triplet> triplets;
    triplets.reserve(X.nonZeros());


    // create list of triplets
    for (int k=0; k < X.outerSize(); ++k) { // foreach col
        for (SpCMat::InnerIterator it(X,k); it; ++it) { // foreach row
            triplets.push_back(Triplet(it.row(), it.col(), it.value()));
        }
    }

    // create sparse matrix with RowMajor
    SpRMat Y(X.rows(), X.cols());
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

