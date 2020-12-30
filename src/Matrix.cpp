#include <string>
#include <valarray>
#include <random>
#include <iostream>
#include <cassert>
#include <stdexcept>

#include "Matrix.hpp"

std::random_device rd;
std::mt19937 rng(rd());

//Helpers

void fill_array_with_random(std::valarray<double> &array, double a = 0, double b = 1)
{
    std::uniform_real_distribution<double> uniform_distrib(a, b);

    for (std::size_t i = 0; i < array.size(); i++)
    {
        array[i] = uniform_distrib(rng);
    }
}

void fill_array_with_random(std::vector<double> &array, double a = 0, double b = 1)
{
    std::uniform_real_distribution<double> uniform_distrib(a, b);

    for (std::size_t i = 0; i < array.size(); i++)
    {
        array[i] = uniform_distrib(rng);
    }
}

void fill_array_with_random(double *array, std::size_t size, double a = 0, double b = 1)
{

    std::uniform_real_distribution<double> uniform_distrib(a, b);

    for (std::size_t i = 0; i < size; i++)
    {
        array[i] = uniform_distrib(rng);
    }
}

// CONSTRUCTORS
Matrix::Matrix(int nb_rows, int nb_columns)
{
    this->nb_rows = nb_rows;
    this->nb_columns = nb_columns;
    this->data = std::vector<double>(nb_columns * nb_rows);

    // std::cout << "💧 Init Matrix : " << this->nb_rows << " X " << this->nb_columns << " = " << this->nb_columns * this->nb_rows << " : " << this << std::endl;
}

// OVERLOADS
Matrix operator+=(Matrix &m1, const Matrix &m2)
{
    assert(m1.nb_columns == m2.nb_columns);
    assert(m1.nb_rows == m2.nb_rows);
    for (int i = 0; i < m1.nb_rows; i++)
    {
        for (int j = 0; j < m1.nb_columns; j++)
        {
            m1(i, j) = m1(i, j) + m2(i, j);
        }
    }

    return m1;
}

Matrix operator+(const Matrix &m1, const Matrix &m2)
{

    // std::cout << "sum" << std::endl;
    assert(m1.nb_columns == m2.nb_columns);
    assert(m1.nb_rows == m2.nb_rows);

    Matrix sum(m1.nb_rows, m1.nb_columns);

    sum.nb_rows = m2.nb_rows;
    sum.nb_columns = m2.nb_columns;
    for (int i = 0; i < m1.nb_rows; i++)
    {
        for (int j = 0; j < m1.nb_columns; j++)
        {
            int position = i * m1.nb_columns + j;
            sum(i, j) = m1(i, j) + m2(i, j);
        }
    }

    return sum;
}

Matrix operator*(const Matrix &m, const double k)
{

    Matrix product(m.nb_rows, m.nb_columns);

    product.nb_rows = m.nb_rows;
    product.nb_columns = m.nb_columns;
    for (int i = 0; i < m.nb_rows; i++)
    {
        for (int j = 0; j < m.nb_columns; j++)
        {
            product(i, j) = m(i, j) * k;
        }
    }

    return product;
}

Matrix operator-(const Matrix &m1, const Matrix &m2)
{
    // std::cout << &m1 << " " << &m2 << std::endl;

    assert(m1.nb_columns == m2.nb_columns);
    assert(m1.nb_rows == m2.nb_rows);

    Matrix sub(m1.nb_rows, m1.nb_columns);

    sub.nb_rows = m2.nb_rows;
    sub.nb_columns = m2.nb_columns;
    for (int i = 0; i < m1.nb_rows; i++)
    {
        for (int j = 0; j < m1.nb_columns; j++)
        {
            sub(i, j) = m1(i, j) - m2(i, j);
        }
    }

    return sub;
}

inline double &Matrix::operator()(const int row, const int col)
{
    if (row >= this->nb_rows || col >= this->nb_columns)
    {
        throw std::out_of_range("Matrix subscript out of bounds");
    }
    return this->data[this->nb_columns * row + col];
}

inline double Matrix::operator()(const int row, const int col) const
{
    if (row >= this->nb_rows || col >= this->nb_columns)
    {
        throw std::out_of_range("Matrix subscript out of bounds");
    }
    return this->data[this->nb_columns * row + col];
}

// METHODS

void Matrix::randomInit()
{
    // std::cout << "Random init" << std::endl;
    // TODO change this

    fill_array_with_random(this->data, -1, 1);
}

void Matrix::testInit()
{
    for (int i = 0; i < this->data.size(); i++)
    {
        this->data[i] = i;
    }
}

void Matrix::fillWith(double value)
{
    for (int i = 0; i < this->nb_rows; i++)
    {
        for (int j = 0; j < this->nb_columns; j++)
        {
            this->operator()(i, j) = value;
        }
    }
}

int const Matrix::getNbColumns()
{
    return this->nb_columns;
}

int const Matrix::getNbRows()
{
    return this->nb_rows;
}

std::string const Matrix::to_string()
{
    std::string display = "";

    for (int i = 0; i < this->nb_rows; i++)
    {
        for (int j = 0; j < this->nb_columns; j++)
        {
            display += std::to_string(this->operator()(i, j)) + ' ';
        }
        display += '\n';
    }
    return display;
}

Matrix const Matrix::transpose()
{
    // opposite ordre for columns and rows
    Matrix matrix_T(this->nb_columns, this->nb_rows);

    for (int i = 0; i < this->nb_rows; i++)
    {
        for (int j = 0; j < this->nb_columns; j++)
        {
            matrix_T.operator()(j, i) = this->operator()(i, j);
        }
    }

    return matrix_T;
}

// Destructor
Matrix::~Matrix()
{
    //std::cout << "💀  Destructeur terminé" << std::endl;
}

// Compute
std::valarray<double> Matrix::dot(const std::valarray<double> &X)
{
    assert(this->nb_columns == X.size());

    std::valarray<double> Y(this->nb_rows);

    for (int i = 0; i < this->nb_rows; i++)
    {
        for (int j = 0; j < this->nb_columns; j++)
        {
            Y[i] += this->operator()(i, j) * X[j];
        }
    }
    return Y;
}

// Related functions

/**
 * @brief Colonne fois ligne
 * 
 * @arg C: vecteur correspondant à la colonne
 * @arg L: vecteur correspondant à la ligne
 * 
 * @return M: Matrice de C x L
 * */

Matrix dot(const std::valarray<double> &C, const std::valarray<double> &L)
{
    Matrix M(C.size(), L.size());

    for (std::size_t i = 0; i < C.size(); i++)
    {
        for (std::size_t j = 0; j < L.size(); j++)
        {
            M(i, j) = C[i] * L[j];
        }
    }
    return M;
}