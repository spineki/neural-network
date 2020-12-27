#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <iostream>
#include "doctest/doctest.h"
#include "Matrix.hpp"

TEST_CASE("Matrix initialisation")
{
    Matrix matrix(5, 5);
    matrix.set(0, 0, 666);
    CHECK(matrix.get(0, 0) == 666);
}
