#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <iostream>
#include <valarray>
#include "doctest/doctest.h"
#include "Matrix.hpp"

TEST_CASE("Matrix initialisation and dot product")
{
    Matrix matrix(2, 3);
    matrix.set(0, 0, 1);
    matrix.set(0, 1, 2);
    matrix.set(0, 2, 3);
    matrix.set(1, 0, 4);
    matrix.set(1, 1, 5);
    matrix.set(1, 2, 6);

    std::valarray<double> X = {1, 2, 3};

    std::valarray<double> Y = matrix.dot(X);

    CHECK(Y[0] == 1 + 4 + 9);
    CHECK(Y[1] == 4 + 10 + 18);
}

TEST_CASE("Matrix transpose")
{
    Matrix matrix(2, 3);
    matrix.set(0, 0, 1);
    matrix.set(0, 1, 2);
    matrix.set(0, 2, 3);
    matrix.set(1, 0, 4);
    matrix.set(1, 1, 5);
    matrix.set(1, 2, 6);

    Matrix matrix_T = matrix.transpose();

    CHECK(matrix_T.get(0, 0) == 1);
    CHECK(matrix_T.get(0, 1) == 4);
    CHECK(matrix_T.get(1, 0) == 2);
    CHECK(matrix_T.get(1, 1) == 5);
    CHECK(matrix_T.get(2, 0) == 3);
    CHECK(matrix_T.get(2, 1) == 6);
}