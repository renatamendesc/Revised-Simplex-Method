#ifndef SIMPLEX_H
#define SIMPLEX_H

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/src/Core/Matrix.h"

#include "data.h"

class Simplex
{
public:
    Simplex(Data &data);

    void BTRAN (); // calculate B*y = c
    void FTRAN (); // calculate B*d = a

    Data data;

};