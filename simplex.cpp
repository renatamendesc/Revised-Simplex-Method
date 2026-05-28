#include "simplex.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::SparseMatrix;

#define REFACTOR 20 // refactorization every 20 iterations

double EPSILON_1 = 1e-5;

Simplex::Simplex (Data &data, SparseMatrix <double> &B, void *Symbolic, void *Numeric, double *null, int phase) : data(data), B(B), Symbolic(Symbolic), Numeric(Numeric), null(null), phase(phase) {}

void Simplex::refactorization ()
{
    cout << endl << "Refactoring..." << endl;
    // update current basis
    for (int i = 0; i < this->data.m; i++)
    {
        this->B.col(i) = this->data.A.col(this->data.basic_indices[i]);
    }

    // delete eta matrices
    this->eta_matrix_col.clear();

    // delete symbolic and numeric factorization
    umfpack_di_free_symbolic(&this->Symbolic);
    umfpack_di_free_numeric(&this->Numeric);

    // create new symbolic and numeric factorization
    umfpack_di_symbolic(this->data.m, this->data.m, this->B.outerIndexPtr(), this->B.innerIndexPtr(), this->B.valuePtr(), &this->Symbolic, this->null, this->null);
    umfpack_di_numeric(this->B.outerIndexPtr(), this->B.innerIndexPtr(), this->B.valuePtr(), this->Symbolic, &this->Numeric, this->null, this->null);
}

VectorXd Simplex::BTRAN ()
{
    // calculate y*B = c

    // ==============================
    // example: y * B_3 = c

    // u * E_3 = c
    // v * E_2 = u
    // y * E_1 = v
    // or
    // u_0 * E_3 = c
    // u_1 * E_2 = u_0
    // y * E_1 = u_1
    // ==============================

    VectorXd y = VectorXd::Zero(this->data.m); // vector with dual multipliers
    VectorXd c_basic(this->data.m);

    for (int i = 0; i < this->data.m; i++)
    {
        c_basic[i] = this->data.c[this->data.basic_indices[i]];
    }

    cout << "c basic: " << c_basic.transpose() << endl;

    vector <VectorXd> u;
    u.push_back(c_basic);
    for (int k = this->eta_matrix_col.size(); k >= 1; k--)
    {
        // solving the linear system to calculate new u
        VectorXd last_u = u.back();
        VectorXd new_u(this->data.m);

        int idx_eta = this->eta_matrix_col[k-1].first;
        double aux = last_u(idx_eta);

        for (int i = 0; i < this->data.m; i++)
        {
            if (i != idx_eta)
            {
                new_u(i) = last_u(i);
                aux -= last_u(i) * this->eta_matrix_col[k-1].second(i);
            }
        }
        new_u(idx_eta) = aux / this->eta_matrix_col[k-1].second(idx_eta);
        u.push_back(new_u);
    }

    (void) umfpack_di_solve(UMFPACK_At, this->B.outerIndexPtr(), this->B.innerIndexPtr(), this->B.valuePtr(), y.data(), (u.back()).data(), this->Numeric, this->null, this->null);
    // if ((B.transpose() * y - u.back()).norm() > EPSILON_1)
    // {
    //     cout << "Error: B^t*y != c_basic. Solver failed." << std::endl;
    //     exit(0);
    // }

    return y;
}

int Simplex::calculate_entering_variable (VectorXd &y)
{
    bool found_entering_variable = false;
    
    // calculate reduced costs (s_j = c_j - y*A_j)
    int k = this->data.n - this->data.m;
    VectorXd reduced_costs(k);
    for (int i = 0; i < k; ++i) {
        int j = data.non_basic_indices[i];
        reduced_costs(i) = data.c(j) - data.A.col(j).dot(y);
    }

    cout << "Reduced costs (y): " << reduced_costs.transpose() << endl;

    // explore non-basic variables
    for (int i = 0; i < k; ++i) {
        int j = data.non_basic_indices[i];

        // explicit bounds: if the reduced cost is positive and variable still can be increased
        if (reduced_costs(i) > EPSILON_1 && this->x_values(j) + EPSILON_1 < this->data.ub(j))
        {
            if (!found_entering_variable || (found_entering_variable && this->entering_variable_idx > j))
            {
                found_entering_variable = true;

                this->entering_variable_idx = j;
                this->entering_column = VectorXd(this->data.A.col(j));
                this->entering_direction = 1;
            }
        }
        // explicit bounds: if the reduced cost is negative and variable still can be decreased
        else if (reduced_costs(i) < -EPSILON_1 && this->x_values(j) - EPSILON_1 > this->data.lb(j))
        {
            if (!found_entering_variable || (found_entering_variable && this->entering_variable_idx > j))
            {
                found_entering_variable = true;

                this->entering_variable_idx = j;
                this->entering_column = VectorXd(this->data.A.col(j));
                this->entering_direction = -1;
            }
        }
    }

    if (!found_entering_variable)
    {
        cout << "No entering variable found." << endl << endl;
        return 0;
    }

    cout << "Entering variable index = " << this->entering_variable_idx << endl;
    cout << "Entering column = " << this->entering_column.transpose() << endl << endl;

    return 1;
}

VectorXd Simplex::FTRAN ()
{
    // calculate B*d = a
    
    // ==============================
    // example: B_3 * d = a

    // E_1 * u = a
    // E_2 * v = u
    // E_3 * d = v
    // or
    // E_1 * u_0 = a
    // E_2 * u_1 = u_0
    // E_3 * d = u_1
    // ==============================

    VectorXd d(this->data.m); // direction vector
    VectorXd d_initial(this->data.m); 

    (void) umfpack_di_solve(UMFPACK_A, this->B.outerIndexPtr(), this->B.innerIndexPtr(), this->B.valuePtr(), d_initial.data(), this->entering_column.data(), this->Numeric, this->null, this->null);
    // if ((B * d_initial - this->entering_column).norm() > EPSILON_1)
    // {
    //     cout << "Error: B*d != a. Solver failed." << std::endl;
    //     exit(0);
    // }

    vector<VectorXd> u;
    u.push_back(d_initial);
    for (int k = 1; k <= this->eta_matrix_col.size(); k++)
    {
        // solving the linear system to calculate new u
        VectorXd last_u = u.back();
        VectorXd new_u(this->data.m);

        int idx_eta = this->eta_matrix_col[k-1].first;
        new_u(idx_eta) = last_u(idx_eta) / this->eta_matrix_col[k-1].second(idx_eta);

        for (int i = 0; i < this->data.m; i++)
        {
            if (i != idx_eta)
            {
                new_u(i) = last_u(i) - this->eta_matrix_col[k-1].second(i) * new_u(idx_eta);
            }
        }
        u.push_back(new_u);
    }
    d = u.back();

    cout << "Direction vector (d): " << d.transpose() << endl;

    return d;
}

void Simplex::calculate_leaving_variable (VectorXd &d)
{
    // calculate leaving variable

    this->min_step_size = numeric_limits<double>::infinity();

    // explore basic variables
    for (int i = 0; i < this->data.m; i++)
    {
        double step_size = numeric_limits<double>::infinity();
        // calculate the change in the basic variables: x_b_new = x_b_old - d * step_size
        int j = this->data.basic_indices[i];
        if ((abs(d(i)) < EPSILON_1))
        {
            // d is zero, so the step is infinite
            step_size = numeric_limits<double>::infinity();
        }
        if (d(i) * -this->entering_direction > EPSILON_1) // variable increases
        {
            step_size = (this->data.ub(j) - this->x_values(j)) / abs(d(i));
        }
        else if (d(i) * -this->entering_direction < -EPSILON_1) // variable decreases
        {
            step_size = (this->x_values(j) - this->data.lb(j)) / abs(d(i));
        }

        if (step_size <= this->min_step_size)
        {
            if (step_size == this->min_step_size)
            {
                if (j < this->leaving_variable_idx)
                {
                    this->leaving_variable_idx = j;
                }
            }
            else
            {
                this->leaving_variable_idx = j;
            }
            this->min_step_size = step_size;
        }
    }

    if (this->min_step_size == numeric_limits<double>::infinity())
    {
        cout << "No leaving variable found." << endl;
        cout << "Solution is unbounded." << endl << endl;
        exit(0);
    }

    cout << "Leaving variable index = " << this->leaving_variable_idx << endl; 
    cout << "Leaving column = " << this->data.A.col(this->leaving_variable_idx).transpose() << endl;
}

void Simplex::update_basis (VectorXd &d)
{
    cout << "Updating basis..." << endl;
    // update the x values
    for (int i = 0; i < this->data.m; i++)
    {
        this->x_values(this->data.basic_indices[i]) += this->min_step_size * -this->entering_direction * d(i);
    }
    this->x_values(this->entering_variable_idx) += this->min_step_size * this->entering_direction;
    cout << "Variables values: " << this->x_values.transpose() << endl;

    // store the entering variable index in the basis and the direction vector
    pair<int, VectorXd> eta_matrix_col_entry;
    eta_matrix_col_entry.second = d;

    // update the basic matrix
    for (int i = 0; i < this->data.m; i++) 
    {
        if (this->data.basic_indices[i] == this->leaving_variable_idx)
        {
            this->data.basic_indices[i] = this->entering_variable_idx;
            eta_matrix_col_entry.first = i;
        }
    }

    for (int i = 0; i < this->data.n - this->data.m; i++) 
    {
        if (this->data.non_basic_indices[i] == this->entering_variable_idx)
        {
            this->data.non_basic_indices[i] = this->leaving_variable_idx;
        }
    }

    cout << "Basic indices: ";
    for (int i = 0; i < this->data.m; i++) 
    {
        cout << this->data.basic_indices[i] << " ";
    }
    cout << endl;
    cout << "Non-basic indices: ";
    for (int i = 0; i < this->data.n - this->data.m; i++) 
    {
        cout << this->data.non_basic_indices[i] << " ";
    }
    cout << endl;

    // store eta matrix
    this->eta_matrix_col.push_back(eta_matrix_col_entry);

    if (this->eta_matrix_col.size() == REFACTOR)
    {
        this->refactorization();
    }
}