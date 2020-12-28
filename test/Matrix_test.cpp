#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <iostream>
#include <valarray>
#include "doctest/doctest.h"
#include "Matrix.hpp"

TEST_CASE("Matrix initialisation and dot product")
{
    Matrix matrix(2, 3);
    matrix(0, 0) = 1;
    matrix(0, 1) = 2;
    matrix(0, 2) = 3;
    matrix(1, 0) = 4;
    matrix(1, 1) = 5;
    matrix(1, 2) = 6;

    std::valarray<double> X = {1, 2, 3};

    std::valarray<double> Y = matrix.dot(X);

    CHECK(Y[0] == 1 + 4 + 9);
    CHECK(Y[1] == 4 + 10 + 18);
}

TEST_CASE("Testing sum")
{
    Matrix matrix(2, 3);
    matrix(0, 0) = 1;
    matrix(0, 1) = 2;
    matrix(0, 2) = 3;
    matrix(1, 0) = 4;
    matrix(1, 1) = 5;
    matrix(1, 2) = 6;

    matrix = matrix + matrix + matrix + matrix;

    CHECK(matrix(0, 0) == 1 * 4);
    CHECK(matrix(0, 1) == 2 * 4);
    CHECK(matrix(0, 2) == 3 * 4);
    CHECK(matrix(1, 0) == 4 * 4);
    CHECK(matrix(1, 1) == 5 * 4);
    CHECK(matrix(1, 2) == 6 * 4);
}

TEST_CASE("Testing product with scalar")
{
    Matrix matrix(2, 3);
    matrix(0, 0) = 1;
    matrix(0, 1) = 2;
    matrix(0, 2) = 3;
    matrix(1, 0) = 4;
    matrix(1, 1) = 5;
    matrix(1, 2) = 6;

    matrix = matrix * 4;

    CHECK(matrix(0, 0) == 1 * 4);
    CHECK(matrix(0, 1) == 2 * 4);
    CHECK(matrix(0, 2) == 3 * 4);
    CHECK(matrix(1, 0) == 4 * 4);
    CHECK(matrix(1, 1) == 5 * 4);
    CHECK(matrix(1, 2) == 6 * 4);
}

TEST_CASE("Matrix transpose")
{
    Matrix matrix(2, 3);
    matrix(0, 0) = 1;
    matrix(0, 1) = 2;
    matrix(0, 2) = 3;
    matrix(1, 0) = 4;
    matrix(1, 1) = 5;
    matrix(1, 2) = 6;

    Matrix matrix_T = matrix.transpose();

    CHECK(matrix_T(0, 0) == 1);
    CHECK(matrix_T(0, 1) == 4);
    CHECK(matrix_T(1, 0) == 2);
    CHECK(matrix_T(1, 1) == 5);
    CHECK(matrix_T(2, 0) == 3);
    CHECK(matrix_T(2, 1) == 6);
}
