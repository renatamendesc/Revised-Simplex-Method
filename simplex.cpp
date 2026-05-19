#include "simplex.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::SparseMatrix;

Simplex::Simplex(Data &data)
{
    this->data = data;
}

void Simplex::BTRAN()
{
    // calculate B*y = c

    
}

void Simplex::FTRAN()
{
    // calculate B*d = a
}