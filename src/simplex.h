#ifndef SIMPLEX_H
#define SIMPLEX_H

#include "eigen/Dense"
#include "eigen/Sparse"
#include "eigen/src/Core/Matrix.h"

#include "data.h"
#include <umfpack.h>
#include <cstdlib>
#include <iostream>

class Simplex
{
public:
    Simplex(Data &data, Eigen::SparseMatrix<double> &B, void *Symbolic, void *Numeric, double *null, int phase);

    Eigen::VectorXd BTRAN (); // solve B*y = c
    Eigen::VectorXd FTRAN (); // calculate B*d = a

    int calculate_entering_variable (Eigen::VectorXd &y); // returns 1 if entering variable found, 0 if solution is optimal
    void calculate_leaving_variable (Eigen::VectorXd &d);

    Eigen::VectorXd x_values;

    Data &data;
    Eigen::SparseMatrix <double> &B; // initial basic matrix
    void *Symbolic = nullptr;
    void *Numeric = nullptr;
    double *null;

    int entering_variable_idx;
    int entering_direction; // 1 for increase, -1 for decrease
    Eigen::VectorXd entering_column;
    
    int leaving_variable_idx;
    double min_step_size;

    void update_basis (Eigen::VectorXd &d);
    std::vector<std::pair <int, Eigen::VectorXd>> eta_matrix_col;

    void refactorization ();

    int phase; // 1 for phase 1, 2 for phase 2
};

#endif