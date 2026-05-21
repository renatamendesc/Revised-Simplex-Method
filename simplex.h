#ifndef SIMPLEX_H
#define SIMPLEX_H

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/src/Core/Matrix.h"

#include "data.h"
#include <umfpack.h>
#include <cstdlib>
#include <iostream>

class Simplex
{
public:
    Simplex(Data &data, Eigen::SparseMatrix<double> &B, void *Symbolic, void *Numeric, double *null);

    Eigen::VectorXd BTRAN (); // solve B*y = c (basic part of reduced costs)
    Eigen::VectorXd FTRAN (); // calculate B*d = a

    void calculate_entering_variable(Eigen::VectorXd &y);
    void calculate_leaving_variable(Eigen::VectorXd &d);

    Eigen::VectorXd x_values;

    Data &data;
    Eigen::SparseMatrix <double> &B; // initial basic matrix
    void *Symbolic;
    void *Numeric;
    double *null;

    int entering_variable_idx;
    int entering_direction; // 1 for increase, -1 for decrease
    Eigen::VectorXd entering_column;
    
    int leaving_variable_idx;

    void update_basic_matrix();

};

#endif