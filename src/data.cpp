#include "data.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::SparseMatrix;

Data::Data(SparseMatrix <double> &A, VectorXd &b, VectorXd &c, VectorXd &ub, VectorXd &lb, int m, int n)
{
    this->m = m; // rows (constraints)
    this->n = n; // columns (variables)

    this->A = A;
    this->b = b;
    this->c = -c; // because we want to maximize the objective function (mps standard form is to minimize)
    this->ub = ub;
    this->lb = lb;

    // initialize vector with basic and non-basic indices
    for (int i = 0; i < n; i++) 
    {
        if (i >= n-m)
            this->basic_indices.push_back(i);
        else
            this->non_basic_indices.push_back(i);
    }
}

void Data::restore_original_data(VectorXd &c,VectorXd &ub, VectorXd &lb)
{
    this->c = -c;
    this->ub = ub;
    this->lb = lb;
}   

void Data::print_data(string instance_name)
{
    std::cout << "\n============ Instance " << instance_name << " data ============\n";

    // std::cout << "\nA (" << m << " x " << n << ") = \n" << MatrixXd(this->A) << "\n\n";

    // std::cout << "b = " << this->b.transpose() << "\n";
    // std::cout << "c = " << this->c.transpose() << "\n\n";


    // std::cout << "lb = " << this->lb.transpose() << "\n";
    // std::cout << "ub = " << this->ub.transpose() << "\n\n";
    std::cout << "================================================\n\n";
}