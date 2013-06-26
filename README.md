# sparsity

*sparsity* is an R package with functions for sparse matrices. 

## Why use sparsity

### Reading and writing SVMlight format

`read.svmlight()` and `write.svmlight()` read/write sparse matrices in SVMlight format.
You will find other functions for this on the internet, but the ones I found were either slow or handled only dense (=normal) matrices.

### LIBLINEAR integration

The [LiblineaR CRAN package](http://cran.r-project.org/web/packages/LiblineaR/) provides an R interface to the [LIBLINEAR library](http://www.csie.ntu.edu.tw/~cjlin/liblinear/), but uses a dense representation. *sparsity*'s functions use sparse matrices (from the Matrix package) instead. In addition it gives you a pointer to LIBLINEAR's internal representation of the data, which means you can train multiple models without the overhead of transforming the input data.

## Installation

```r
# install.packages("devtools")
library(devtools)
install_github("sparsity", "felixr")
```
