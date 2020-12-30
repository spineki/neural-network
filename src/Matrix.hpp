#pragma once
#include <string>
#include <valarray>
#include <vector>

class Matrix
{

private:
    std::vector<double> data;
    int nb_rows;
    int nb_columns;

public:
    Matrix(int nb_rows, int nb_columns);

    // C
    // Matrix(Matrix const &other);
    // Matrix operator=(const Matrix &other);
    friend Matrix operator+(const Matrix &m1, const Matrix &m2);
    friend Matrix operator+=(Matrix &m1, const Matrix &m2);
    friend Matrix operator*(const Matrix &m, const double k);
    friend Matrix operator-(const Matrix &m1, const Matrix &m2);
    // METHODS
    void randomInit();
    void testInit();
    void fillWith(double value);

    int const getNbColumns();
    int const getNbRows();

    double &operator()(const int row, const int col);
    double operator()(const int row, const int col) const;

    Matrix const transpose();

    std::string const to_string();

    // Compute
    std::valarray<double> dot(const std::valarray<double> &X);

    // Destructor
    ~Matrix();

    auto test();
};

Matrix dot(const std::valarray<double> &C, const std::valarray<double> &L);