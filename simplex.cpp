#include "simplex.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::SparseMatrix;

Simplex::Simplex(Data &data, SparseMatrix <double> &B, void *Symbolic, void *Numeric, double *null) : data(data), B(B), Symbolic(Symbolic), Numeric(Numeric), null(null) {}

VectorXd Simplex::BTRAN()
{
    // calculate y*B = c

    VectorXd y(this->data.m); // vector with dual multipliers
    VectorXd c_basic(this->data.m);

    for (int i = 0; i < this->data.m; i++)
    {
        c_basic[i] = this->data.c[this->data.basic_indices[i]];
    }
    (void) umfpack_di_solve(UMFPACK_A, B.outerIndexPtr(), B.innerIndexPtr(), B.valuePtr(), y.data(), c_basic.data(), Numeric, null, null);

    if ((B * y - c_basic).norm() > 0.0000001)
    {
        cout << "Error: B*y != c_basic. Solver failed." << std::endl;
        exit(0);
    }
    else
    {
        cout << "y = " << y.transpose() << endl;
    }

    return y;
}

void Simplex::calculate_entering_variable(VectorXd &y)
{
    // calculate reduced costs (s_j = c_j - y*A_j)
    int k = this->data.n - this->data.m;
    VectorXd reduced_costs(k);
    for (int i = 0; i < k; ++i) {
        int j = data.non_basic_indices[i];
        reduced_costs(i) = data.c(j) - data.A.col(j).dot(y);
    }

    cout << "reduced costs = " << reduced_costs.transpose() << endl;

    // index in non_basic_indices with biggest reduced cost (maximize)
    int most_positive_reduced_cost_idx = 0;
    double most_positive_reduced_cost = reduced_costs(0);
    for (int i = 1; i < k; ++i) {
        if (reduced_costs(i) > most_positive_reduced_cost) {
            most_positive_reduced_cost = reduced_costs(i);
            most_positive_reduced_cost_idx = i;
        }
    }

    this->entering_variable_idx = data.non_basic_indices[most_positive_reduced_cost_idx];
    this->entering_column = VectorXd(data.A.col(this->entering_variable_idx));

    cout << "most positive reduced cost = " << most_positive_reduced_cost << endl;
    cout << "entering column index = " << this->entering_variable_idx << endl;
    cout << "entering column = " << this->entering_column.transpose() << endl;

    // if the most positive reduced cost is 0, the solution is optimal
    if (most_positive_reduced_cost == 0) {
        cout << "Found optimal." << endl;
        exit(0);
    }
}

VectorXd Simplex::FTRAN()
{
    // calculate B*d = a

    VectorXd d(this->data.m); // direction vector

    (void) umfpack_di_solve(UMFPACK_A, B.outerIndexPtr(), B.innerIndexPtr(), B.valuePtr(), d.data(), this->entering_column.data(), Numeric, null, null);

    if ((B * d - this->entering_column).norm() > 0.0000001)
    {
        cout << "Error: B*d != a. Solver failed." << std::endl;
        exit(0);
    }
    else
    {
        cout << "d = " << d.transpose() << endl;
    }

    return d;
}

void Simplex::calculate_leaving_variable(VectorXd &d)
{
    // calculate leaving variable (l_i = b_i / d_i)

    // to-do: implement version with explicit bounds
}