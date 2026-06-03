#include <iostream>
#include <cstdlib>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/src/Core/Matrix.h"

#include <numeric>
#include <ostream>
#include <utility>
#include <vector>
#include <algorithm>
#include <map>
#include <stdio.h>
#include <umfpack.h>
#include <chrono>

#include "mpsReader.h"
#include "data.h"
#include "simplex.h"

double pInf = numeric_limits<double>::infinity();
double nInf = -numeric_limits<double>::infinity();

using Eigen::MatrixXd;
using namespace std;

bool apply_penalties (mpsReader &mps, Data &data_phase_1, Eigen::VectorXd &x_values)
{
	data_phase_1.restore_original_data(mps.c, mps.ub, mps.lb);

	double violation_value = 0;
	data_phase_1.c = Eigen::VectorXd::Zero(data_phase_1.n);

	// verify if basic variable violate the bounds
	for (int i = 0; i < data_phase_1.basic_indices.size(); i++)
	{
		int j = data_phase_1.basic_indices[i];
		if (x_values[j] < mps.lb[j])
		{
			// cout << "Basic variable " << j << " violates the lower bound!" << endl;
			data_phase_1.ub[j] = mps.lb[j];
			data_phase_1.lb[j] = nInf;
			data_phase_1.c[j] = 1;
			violation_value += mps.lb[j] - x_values[j];
		}
		else if (x_values[j] > mps.ub[j])
		{
			// cout << "Basic variable " << j << " violates the upper bound!" << endl;
			data_phase_1.lb[j] = mps.ub[j];
			data_phase_1.ub[j] = pInf;
			data_phase_1.c[j] = -1;
			violation_value += x_values[j] - mps.ub[j];
		}
	}

	if (violation_value < 1e-5)
		return true;
	else
		return false;
}

void generate_initial_basic_solution(Data &data, Simplex &simplex, Eigen::SparseMatrix<double> &B, void *Symbolic, void *Numeric, double *null)
{
	Eigen::VectorXd x_N(data.n - data.m);
    Eigen::MatrixXd N = Eigen::MatrixXd::Zero(data.m, data.n - data.m);

    for (int i = 0; i < data.n - data.m; i++)
    {
        int j = data.non_basic_indices[i];
        N.col(i) = MatrixXd(data.A.col(j));
        if (data.ub[j] == pInf && data.lb[j] == nInf) // free variable, set to zero
            x_N[i] = 0;
        else if (data.lb[j] == -pInf) 				  // variable with only upper bound, set to upper bound
            x_N[i] = data.ub[j];
        else 									      // variable with lower and upper bound, set to lower bound
            x_N[i] = data.lb[j];
    }

    // solving B * x_B = b - N*x_N
	Eigen::VectorXd rhs = data.b - N * x_N;
	Eigen::VectorXd x_B = Eigen::VectorXd::Zero(data.m);
    (void)umfpack_di_solve(UMFPACK_A, B.outerIndexPtr(), B.innerIndexPtr(), B.valuePtr(), x_B.data(), rhs.data(), Numeric, null, null);

	simplex.x_values = Eigen::VectorXd::Zero(data.n);
	for (int i = 0; i < data.n - data.m; i++)
	{
		simplex.x_values[data.non_basic_indices[i]] = x_N[i];
	}
	for (int i = 0; i < data.m; i++)
	{
		simplex.x_values[data.basic_indices[i]] = x_B[i];
	}

	// cout << "\nx_values = " << simplex.x_values.transpose() << endl;
}

int main(int argc, char** argv)
{
	std::string mps_path = argv[1];
	int pre_process = std::stoi(argv[2]); // can be 1 or 0 to activate it or not

	mpsReader mps;
	mps.read(mps_path, pre_process);

	Eigen::SparseMatrix <double> A_sparse = mps.A.sparseView();
	int m = mps.n_rows_eq + mps.n_rows_inq;
	int n = mps.n_cols + mps.n_rows_inq; 
	Data data(A_sparse, mps.b, mps.c, mps.ub, mps.lb, m, n);
	data.print_data(mps.Name);

	auto start = std::chrono::steady_clock::now();

	// initialize basic matrix B
	Eigen::SparseMatrix <double> B (data.m, data.m);
	for (int i = 0; i < data.m; i++)
	{
		B.col(i) = data.A.col(data.basic_indices[i]);
	}
	// cout << "B = \n" << MatrixXd(B) << "\n";

	// LU factorization of the initial basic matrix B
	double *null = (double *) NULL ;
	void *Symbolic, *Numeric ;

	(void) umfpack_di_symbolic (data.m, data.m, B.outerIndexPtr(), B.innerIndexPtr(), B.valuePtr(), &Symbolic, null, null);
	(void) umfpack_di_numeric (B.outerIndexPtr(), B.innerIndexPtr(), B.valuePtr(), Symbolic, &Numeric, null, null);

	Simplex simplex(data, B, Symbolic, Numeric, null, 1);

	generate_initial_basic_solution(data, simplex, B, Symbolic, Numeric, null); // phase 1: determine initial solution

	cout << endl << "Solving..." << endl;
	int iter = 0;
	while (true)
	{
		if (simplex.phase == 1 && apply_penalties(mps, simplex.data, simplex.x_values))
		{
			simplex.phase = 2;

			// cout << endl << "================================================" << endl;
			// cout << "Phase 2 started!" << endl;
			// cout << "================================================" << endl;

			simplex.data.restore_original_data(mps.c, mps.ub, mps.lb);
		}

		// cout << endl << "================================================" << endl;
		// cout << "Iter: " << iter+1 << endl;
		// cout << "================================================" << endl << endl;
		VectorXd y = simplex.BTRAN();
		bool found_entering_variable = simplex.calculate_entering_variable(y);
		if (!found_entering_variable)
		{
			cout << "Solution is optimal!" << endl;
			cout << "Objetive value = " << (-mps.c).transpose() * simplex.x_values << endl;

			auto end = std::chrono::steady_clock::now();
			chrono::duration<double> elapsed_time = end - start;
			cout << "Time (seconds) = " << elapsed_time.count() << endl << endl;
			return 0;
		}
	
		VectorXd d = simplex.FTRAN();
		simplex.calculate_leaving_variable(d);
	
		simplex.update_basis(d);

		iter++;
	}

	return 0;
}



