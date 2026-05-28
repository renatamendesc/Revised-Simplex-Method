#ifndef DATA_H
#define DATA_H

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/src/Core/Matrix.h"

#include <iostream>
#include <string>
#include <vector>

class Data
{
public:

    Data(Eigen::SparseMatrix <double> &A, Eigen::VectorXd &b, Eigen::VectorXd &c, Eigen::VectorXd &ub, Eigen::VectorXd &lb, int m, int n);
    void print_data(std::string instance_name);

    void restore_original_data(Eigen::VectorXd &c,Eigen::VectorXd &ub, Eigen::VectorXd &lb);

    int m;
    int n;

    Eigen::SparseMatrix <double> A;
    Eigen::VectorXd b;
    Eigen::VectorXd c;
    Eigen::VectorXd ub;
    Eigen::VectorXd lb;

    std::vector<int> basic_indices;
    std::vector<int> non_basic_indices;
};

#endif