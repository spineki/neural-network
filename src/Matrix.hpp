#pragma once
#include <string>
#include <valarray>

class Matrix
{

private:
    double *values;
    int nb_rows;
    int nb_columns;

public:
    Matrix();

    Matrix(int nb_rows, int nb_columns);

    // METHODS
    void randomInit();

    double get(int i, int j);

    void set(int i, int j, double value);

    std::string to_string();

    // Compute
    std::valarray<double> dot(const std::valarray<double> &X);

    // Destructor
    ~Matrix();
};

Matrix dot(const std::valarray<double> &C, const std::valarray<double> &L);