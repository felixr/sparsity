// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <Rcpp.h>
#include <fstream>
#include <iostream>

#define INF HUGE_VAL
#include "liblinear-1.93/linear.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

using namespace Rcpp;

static const char *solver_type_table[]=
{
    "L2R_LR", "L2R_L2LOSS_SVC_DUAL", "L2R_L2LOSS_SVC", "L2R_L1LOSS_SVC_DUAL", "MCSVM_CS",
    "L1R_L2LOSS_SVC", "L1R_LR", "L2R_LR_DUAL",
    "", "", "",
    "L2R_L2LOSS_SVR", "L2R_L2LOSS_SVR_DUAL", "L2R_L1LOSS_SVR_DUAL", NULL
};

// [[Rcpp::export]]
List liblinearTrain(Eigen::MappedSparseMatrix<double> inputMatrix, 
        std::vector<double> labels,
        int solver_type,
        double cost,
        double epsilon){
    typedef Eigen::MappedSparseMatrix<double> SpCMat;
    typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpRMat;
    typedef Eigen::Triplet<double> Triplet; 

    set_print_string_function(NULL);
    std::vector<Triplet> triplets;
    triplets.reserve(inputMatrix.nonZeros());

    struct model* model_;
    struct feature_node *x_space;
    struct parameter param;
    struct problem prob;
    int flag_cross_validation;
    int nr_fold;

    // default values
    param.solver_type = solver_type;
    param.C = cost;
    param.eps = epsilon;
    param.p = 0.1;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;
    prob.bias = -1;


    // create list of triplets
    int elements = 0;
    for (int k=0; k < inputMatrix.outerSize(); ++k) { // foreach col
        for (SpCMat::InnerIterator it(inputMatrix,k); it; ++it) { // foreach row
            triplets.push_back(Triplet(it.row(), it.col(), it.value()));
            elements++;
        }
    }
    int max_index = inputMatrix.cols();

    elements += (prob.bias >= 0 ? 1 : 0);

    prob.l = labels.size();
    /* prob.y = new double[prob.l]; */
    prob.y = Malloc(double, prob.l);
    std::copy(labels.begin(), labels.end(), prob.y);

    /* prob.x = new feature_node*[prob.l]; */
    /* x_space =new feature_node[elements + prob.l]; */

    prob.x = Malloc(struct feature_node *, prob.l);
    x_space = Malloc(struct feature_node, elements+prob.l);

    // create sparse matrix with RowMajor
    SpRMat *rowMatrix = new SpRMat(inputMatrix.rows(), inputMatrix.cols());
    rowMatrix->setFromTriplets(triplets.begin(), triplets.end());

    // now outer corresponds to rows (RowMajor)
    int j = 0;
    for (int i=0; i < rowMatrix->outerSize(); ++i) { // foreach row
        prob.x[i] = &x_space[j];
        for (SpRMat::InnerIterator it(*rowMatrix, i); it; ++it) { // foreach col
            //    out << " " << it.col() << ":" << it.value();
            x_space[j].index = it.col()+1;
            x_space[j].value = it.value();
            ++j;
        }

        // add bias value at the end of row
        if(prob.bias >= 0) {
            x_space[j].index = max_index+1;
            x_space[j].value = prob.bias;
            ++j;
        }
        // terminate with -1
        x_space[j].index = -1;
        ++j;
    }
    // free up memory
    delete rowMatrix;
    triplets.clear();

    if(prob.bias >= 0) {
        prob.n = max_index+1;
        for(int i = 1; i < prob.l; i++) {
            (prob.x[i]-2)->index = prob.n;
        }
        x_space[j-2].index = prob.n;
    } else {
        prob.n = max_index;
    }

    // prob initialized
    model_ = train(&prob, &param);
    
    // copy weights
    std::vector<double> w(&model_->w[0], &model_->w[prob.n]);

    List ret = List::create(
            _["w"] = w,
            _["solver_name"] = solver_type_table[param.solver_type],
            _["solver_type"] = param.solver_type,
            _["nr_class"] = model_->nr_class,
            _["nr_features"] =  inputMatrix.cols()
            ) ;
    destroy_param(&param);
    free_and_destroy_model(&model_);

    // delete[] prob.x;
    // delete[] x_space;
    free(prob.x);
    free(x_space);

    return ret;
}

