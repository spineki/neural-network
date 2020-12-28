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

    // C
    Matrix(Matrix const &other);
    Matrix operator=(const Matrix &other);
    friend Matrix &operator+(const Matrix &m1, const Matrix &m2);
    friend Matrix &operator+=(Matrix &m1, const Matrix &m2);
    friend Matrix operator*(const Matrix &m, const double k);
    friend Matrix operator-(const Matrix &m1, const Matrix &m2);
    // METHODS
    void randomInit();
    void fillWith(double value);

    int const getNbColumns();
    int const getNbRows();

    double const
    get(const int i, const int j);

    void set(const int i, const int j, const double value);

    Matrix const transpose();

    std::string const to_string();

    // Compute
    std::valarray<double> dot(const std::valarray<double> &X);

    // Destructor
    ~Matrix();

    auto test();
};

Matrix dot(const std::valarray<double> &C, const std::valarray<double> &L);