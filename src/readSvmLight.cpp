// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <Rcpp.h>

#include <Eigen/Sparse>

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>

using namespace Rcpp;

// [[Rcpp::export]]
SEXP readSvmLight(SEXP fname){
    typedef Eigen::Triplet<double> Triplet; 
    typedef Eigen::SparseMatrix<double> SpCMat;
    typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpRMat;

    std::vector<Triplet> *triplets = new std::vector<Triplet>();
    std::vector<double> targetValues;

    triplets->reserve(100000);
    targetValues.reserve(50000);

    std::string matrixfile(Rcpp::as<std::string>(fname));
    std::ifstream inputFile(matrixfile.c_str());
    std::string line;
    int nrow = 0;
    int ncol = 0;
    // foreach line
    while (!inputFile.eof()) {
        std::getline(inputFile, line);
        std::istringstream tokenStream( line );
        std::string token;

        // first token = target value
        std::getline(tokenStream, token, ' ');
        double target;
        sscanf(token.c_str(), "%lf", &target);
        targetValues.push_back(target);

        // foreach token (id:value) in line 
        while (!tokenStream.eof()) {
            std::getline(tokenStream, token, ' ');
            int id;
            double value;
            sscanf(token.c_str(), "%d:%lf", &id, &value);
            ncol = std::max(id, ncol); 
            triplets->push_back(Triplet(nrow, id, value));
        }
        nrow++;
    }
    ncol++;

    SpCMat mat(nrow, ncol);
    mat.setFromTriplets(triplets->begin(), triplets->end());
    mat.makeCompressed();
    delete triplets;
    return List::create( 
            _["matrix"] = mat, 
            _["labels"] = targetValues 
            ) ;
}
