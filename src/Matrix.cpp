#include <string>
#include <valarray>
#include <random>
#include <iostream>
#include <cassert>

#include "Matrix.hpp"

std::random_device rd;
std::mt19937 rng(rd());

void fill_array_with_random(std::valarray<double> &array, double a = 0, double b = 1)
{
    std::uniform_real_distribution<double> uniform_distrib(a, b);

    for (int i = 0; i < array.size(); i++)
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

Matrix::Matrix()
{
    this->nb_rows = 0;
    this->nb_columns = 0;
    this->values = new double[0];

    std::cout << "❄️ Init Matrix : " << this->nb_rows << " X " << this->nb_columns << " = " << this->nb_columns * this->nb_rows << " : " << this << std::endl;
}

Matrix::Matrix(int nb_rows, int nb_columns)
{
    this->nb_rows = nb_rows;
    this->nb_columns = nb_columns;
    this->values = new double[nb_rows * nb_columns];

    std::cout << "💧 Init Matrix : " << this->nb_rows << " X " << this->nb_columns << " = " << this->nb_columns * this->nb_rows << " : " << this << std::endl;
}

// Copy constructor
Matrix::Matrix(Matrix const &model)
{
    std::cout << "🛠️  copy construction: " << &model << " -> " << this << std::endl;
    this->nb_rows = model.nb_rows;
    this->nb_columns = model.nb_columns;
    this->values = new double[nb_rows * nb_columns];
    for (int i = 0; i < nb_rows; i++)
    {
        for (int j = 0; j < nb_columns; j++)
        {
            int position = i * nb_columns + j;
            this->values[position] = model.values[position];
        }
    }

    std::cout << "✔️  copy done: " << &model << " -> " << this << " |   " << ((this->values == model.values) ? "💥" : "✔️") << "  " << this->values << std::endl;
}

Matrix Matrix::operator=(const Matrix &rhs)
{

    std::cout << "🛠️  =: " << &rhs << " -> " << this << std::endl;

    std::cout << nb_rows << std::endl;
    std::cout << rhs.nb_rows << std::endl;

    // Only do assignment if RHS is a different object from this.
    if (this != &rhs)
    {
        delete[] this->values;
        this->values = new double[nb_rows * nb_columns];

        nb_rows = rhs.nb_rows;
        nb_columns = rhs.nb_columns;
        values = new double[nb_rows * nb_columns];
        for (int i = 0; i < nb_rows; i++)
        {
            for (int j = 0; j < nb_columns; j++)
            {
                int position = i * nb_columns + j;
                this->values[position] = rhs.values[position];
            }
        }
    }

    // std::cout << "✔️  = done: " << &model << " -> " << this << " |   " << ((this->values == model.values) ? "💥" : "✔️") << "  " << this->values << std::endl;

    return *this;
}

Matrix &operator+=(Matrix &m1, const Matrix &m2)
{
    assert(m1.nb_columns == m2.nb_columns);
    assert(m1.nb_rows == m2.nb_rows);
    for (int i = 0; i < m1.nb_rows; i++)
    {
        for (int j = 0; j < m1.nb_columns; j++)
        {
            int position = i * m1.nb_columns + j;
            m1.values[position] = m1.values[position] + m2.values[position];
        }
    }

    return m1;
}

Matrix &operator+(const Matrix &m1, const Matrix &m2)
{

    std::cout << "sum" << std::endl;
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
            sum.values[position] = m1.values[position] + m2.values[position];
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
            int position = i * m.nb_columns + j;
            product.values[position] = m.values[position] * k;
        }
    }

    return product;
}

Matrix operator-(const Matrix &m1, const Matrix &m2)
{
    std::cout << &m1 << " " << &m2 << std::endl;

    assert(m1.nb_columns == m2.nb_columns);
    assert(m1.nb_rows == m2.nb_rows);

    Matrix sub(m1.nb_rows, m1.nb_columns);

    sub.nb_rows = m2.nb_rows;
    sub.nb_columns = m2.nb_columns;
    for (int i = 0; i < m1.nb_rows; i++)
    {
        for (int j = 0; j < m1.nb_columns; j++)
        {
            int position = i * m1.nb_columns + j;
            sub.values[position] = m1.values[position] - m2.values[position];
        }
    }

    return sub;
}

// METHODS

void Matrix::randomInit()
{
    // std::cout << "Random init" << std::endl;
    fill_array_with_random(this->values, this->nb_rows * this->nb_columns, 0, 1);
}

void Matrix::fillWith(double value)
{
    for (int i = 0; i < this->nb_rows; i++)
    {
        for (int j = 0; j < this->nb_columns; j++)
        {
            this->set(i, j, value);
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

double const Matrix::get(int i, int j)
{
    return this->values[j + i * nb_columns];
}

void Matrix::set(const int i, const int j, const double value)
{
    this->values[j + i * nb_columns] = value;
}

std::string const Matrix::to_string()
{
    std::string display = "";

    for (int i = 0; i < this->nb_rows; i++)
    {
        for (int j = 0; j < this->nb_columns; j++)
        {
            display += std::to_string(this->get(i, j)) + ' ';
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
            matrix_T.set(j, i, this->get(i, j));
        }
    }

    return matrix_T;
}

// Destructor
Matrix::~Matrix()
{
    std::cout << "🔥  Destructeur appellé: " << this << " ressources : " << this->values << std::endl;

    delete[] this->values;
    std::cout << "💀  Destructeur terminé" << std::endl;
}

auto Matrix::test()
{
    return this->values;
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
            Y[i] += this->get(i, j) * X[j];
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
    std::cout << "<" << std::endl;
    Matrix M(C.size(), L.size());
    std::cout << M.test() << std::endl;

    for (std::size_t i = 0; i < C.size(); i++)
    {
        for (std::size_t j = 0; j < L.size(); j++)
        {
            M.set(i, j, C[i] * L[j]);
        }
    }
    return M;
}