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

#include "mpsReader.h"
#include "data.h"
#include "simplex.h"

using Eigen::MatrixXd;
using namespace std;

double pInf = numeric_limits<double>::infinity();
double nInf = -numeric_limits<double>::infinity();
double EPSILON_1 = 1e-5;

// gerador de matrizes inversiveis B aleatorias (it_max e o grau de aleatoriedade)
Eigen::MatrixXd gen_random_non_singular_mat(int n, int it_max)
{

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n,n);

    std::vector<int> idx_vec(n);
    std::iota(idx_vec.begin(), idx_vec.end(), 1);

    for (int k = 0; k < it_max; k++)
    {
        int r1 = idx_vec[rand() % n] - 1;
        std::swap(idx_vec.back(), idx_vec[r1]);
        int r2 = idx_vec[rand() % (n - 1)] - 1;

        I.row(r1) += ((double) rand() / RAND_MAX) * (rand() % 2 ? -1 : 1) * I.row(r2);
    }

    return I;
}

// cria matriz E aleatoria 
std::pair<int, Eigen::VectorXd> gen_random_eta_mat(int n)
{
	int p = rand() % n;
	Eigen::VectorXd d = Eigen::VectorXd::Random(n);

	while (std::abs(d[p]) < 0.000001)
	{
		d = Eigen::VectorXd::Random(n);
	}

	return std::make_pair(p, d);
}

void test_eigen(int n)
{
	Eigen::MatrixXd B_dense = gen_random_non_singular_mat(n, 20); // generating a random non-singular matrix

	Eigen::SparseMatrix<double> B = B_dense.sparseView(); // compressing the matrix, converting it to a sparse matrix

	// Criando decomposicao LU para a matriz esparsa B usando UMFPACK
	double *null = (double *) NULL ;
	void *Symbolic, *Numeric ;

	(void) umfpack_di_symbolic (n, n, B.outerIndexPtr(), B.innerIndexPtr(), B.valuePtr(), &Symbolic, null, null);
	(void) umfpack_di_numeric (B.outerIndexPtr(), B.innerIndexPtr(), B.valuePtr(), Symbolic, &Numeric, null, null);

	// Resolvendo o sistema Bd = a varias vezes, reutilizando a mesma decomposicao LU de B para varios vetores a diferentes
	Eigen::VectorXd d(n);

	for (int i = 0; i < 100; i++)
	{
		Eigen::VectorXd a = Eigen::VectorXd::Random(n);

		(void) umfpack_di_solve(UMFPACK_A, B.outerIndexPtr(), B.innerIndexPtr(), B.valuePtr(), d.data(), a.data(), Numeric, null, null);

		// verificando se de fato Bd = a
		if ((B * d - a).norm() > 0.0000001)
		{
			std::cout << "Deu errado" << std::endl;
			exit(0);
		}
	}

	umfpack_di_free_symbolic (&Symbolic);
	umfpack_di_free_numeric (&Numeric);

	// gerando matriz E aleatoria (apenas a coluna e sua localizacao sao necessarias para representa-la)
	auto [eta_idx, eta_col] = gen_random_eta_mat(n);

	// matriz E representada na tora, apenas para fins ilustrativos
	Eigen::MatrixXd E = Eigen::MatrixXd::Identity(n,n); 
	E.col(eta_idx) = eta_col;

	std::cout << "Antiga matriz B:\n" << B.toDense() << std::endl << std::endl;
	std::cout << "Nova matriz B:\n" << B * E << std::endl;
}


int main(int argc, char** argv)
{
	std::string mps_path = argv[1];
	int pre_process = std::stoi(argv[2]); // can be 1 or 0 to activate it or not

	mpsReader mps;
	mps.read(mps_path, pre_process);

	Eigen::SparseMatrix <double> A_sparse = mps.A.sparseView();
	Data data(A_sparse, mps.b, mps.c, mps.ub, mps.lb, mps.n_rows_eq + mps.n_rows_inq, mps.n_cols + mps.n_rows_inq + mps.n_rows_eq);
	data.print_data(mps.Name);

	// initialize basic matrix B
	Eigen::SparseMatrix <double> B (data.m, data.m);
	for (int i = 0; i < data.m; i++)
	{
		B.col(i) = data.A.col(data.basic_indices[i]);
	}
	cout << "B = \n" << MatrixXd(B) << "\n";

	// LU factorization of the initial basic matrix B
	double *null = (double *) NULL ;
	void *Symbolic, *Numeric ;

	(void) umfpack_di_symbolic (data.m, data.m, B.outerIndexPtr(), B.innerIndexPtr(), B.valuePtr(), &Symbolic, null, null);
	(void) umfpack_di_numeric (B.outerIndexPtr(), B.innerIndexPtr(), B.valuePtr(), Symbolic, &Numeric, null, null);

	Simplex simplex(data, B, Symbolic, Numeric, null);

	// =======================================================================
	// determine initial basic solution
	// for now, start with all non-basic variables fixed at their lower bounds
	Eigen::VectorXd x_N(data.n - data.m);
    Eigen::MatrixXd N = Eigen::MatrixXd::Zero(data.m, data.n - data.m);

    for (int i = 0; i < data.n - data.m; i++)
    {
        int j = data.non_basic_indices[i];
        N.col(i) = data.A.col(j);
        if (data.ub[j] == pInf && data.lb[j] == nInf)
            x_N[i] = 0;
        else if (data.lb[j] == -pInf)
            x_N[i] = data.ub[j];
        else
            x_N[i] = data.lb[j];
    }

    // solving B * x_B = b - N*x_N
	Eigen::VectorXd b = data.b - N * x_N;
	Eigen::VectorXd x_B = Eigen::VectorXd::Zero(data.m);

    (void)umfpack_di_solve(UMFPACK_A, B.outerIndexPtr(), B.innerIndexPtr(), B.valuePtr(), x_B.data(), b.data(), Numeric, null, null);


	simplex.x_values = Eigen::VectorXd::Zero(data.n);
	for (int i = 0; i < data.n - data.m; i++)
	{
		simplex.x_values[data.non_basic_indices[i]] = x_N[i];
	}
	for (int i = 0; i < data.m; i++)
	{
		simplex.x_values[data.basic_indices[i]] = x_B[i];
	}

	cout << "\nx_values = " << simplex.x_values.transpose() << "\n";
	// =======================================================================

	while (true)
	{
		VectorXd y = simplex.BTRAN();
		bool found_entering_variable = simplex.calculate_entering_variable(y);
		if (!found_entering_variable)
		{
			cout << "Solution is optimal!" << endl;
			// to-do: add function to print the solution
			return 0;
		}
	
		VectorXd d = simplex.FTRAN();
		simplex.calculate_leaving_variable(d);
	
		simplex.update_basis(d);
	}

	return 0;
}

// === EIGEN TEST ===
// srand(time(NULL));
// test_eigen(3); // test the eigen library with matrix 3x3



