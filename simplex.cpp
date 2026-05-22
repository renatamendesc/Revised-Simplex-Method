#include "simplex.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::SparseMatrix;

Simplex::Simplex(Data &data, SparseMatrix <double> &B, void *Symbolic, void *Numeric, double *null) : data(data), B(B), Symbolic(Symbolic), Numeric(Numeric), null(null) {}

VectorXd Simplex::BTRAN()
{
    // calculate y*B = c
    // to-do: use eta matrix

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

int Simplex::calculate_entering_variable(VectorXd &y)
{
    // calculate reduced costs (s_j = c_j - y*A_j)
    int k = this->data.n - this->data.m;
    VectorXd reduced_costs(k);
    for (int i = 0; i < k; ++i) {
        int j = data.non_basic_indices[i];
        reduced_costs(i) = data.c(j) - data.A.col(j).dot(y);
    }

    cout << "reduced costs = " << reduced_costs.transpose() << endl;

    // explore non-basic variables
    for (int i = 0; i < k; ++i) {
        int j = data.non_basic_indices[i];

        cout << "reduced cost for x_" << j << " = " << reduced_costs(i) << endl;
        cout << "x_" << j << " = " << this->x_values(j) << endl;
        cout << "ub_" << j << " = " << this->data.ub(j) << endl;
        cout << "lb_" << j << " = " << this->data.lb(j) << endl;

        // explicit bounds: if the reduced cost is positive and variable still can be increased
        if (reduced_costs(i) > 0 && this->x_values(j) < this->data.ub(j))
        {
            this->entering_variable_idx = j;
            this->entering_column = VectorXd(this->data.A.col(j));
            this->entering_direction = 1;

            cout << "entering variable index = " << this->entering_variable_idx << endl;
            cout << "entering column = " << this->entering_column.transpose() << endl;
            return 1;
        }
        // explicit bounds: if the reduced cost is negative and variable still can be decreased
        else if (reduced_costs(i) < 0 && this->x_values(j) > this->data.lb(j))
        {
            this->entering_variable_idx = j;
            this->entering_column = VectorXd(this->data.A.col(j));
            this->entering_direction = -1;

            cout << "entering variable index = " << this->entering_variable_idx << endl;
            cout << "entering column = " << this->entering_column.transpose() << endl;
            return 1;
        }
    }

    cout << "No entering variable found. Solution is optimal!" << endl;
    return 0;
}

VectorXd Simplex::FTRAN()
{
    // calculate B*d = a
    // to-do: use eta matrix

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
    // calculate leaving variable

    // explore basic variables
    double step_size = 0;
    double min_step_size = numeric_limits<double>::infinity();
    for (int i = 0; i < this->data.m; i++)
    {
        // calculate the change in the basic variables: x_b_new = x_b_old - d * step_size
        int j = this->data.basic_indices[i];
        if (d(i) * -this->entering_direction > 0) // variable increases
        {
            step_size = (this->data.ub(j) - this->x_values(j)) / abs(d(i));
        }
        else if (d(i) * -this->entering_direction < 0) // variable decreases
        {
            step_size = (this->x_values(j) - this->data.lb(j)) / abs(d(i));
        }

        if (step_size < min_step_size)
        {
            min_step_size = step_size;
            this->leaving_variable_idx = j;
        }
    }

    if (min_step_size == numeric_limits<double>::infinity())
    {
        cout << "No leaving variable found." << endl;
        exit(0);
    }

    cout << "leaving variable index = " << this->leaving_variable_idx << endl; 
    cout << "leaving column = " << this->data.A.col(this->leaving_variable_idx).transpose() << endl;
}

void Simplex::update_basis(VectorXd &d)
{
    // update the basic matrix
    for (int i = 0; i < this->data.m; i++) 
    {
        if (this->data.basic_indices[i] == this->leaving_variable_idx)
        {
            this->data.basic_indices[i] = this->entering_variable_idx;
        }
    }

    for (int i = 0; i < this->data.n - this->data.m; i++) 
    {
        if (this->data.non_basic_indices[i] == this->entering_variable_idx)
        {
            this->data.non_basic_indices[i] = this->leaving_variable_idx;
        }
    }

    // store eta matrix
    this->eta_matrix_col.push_back(d);
}