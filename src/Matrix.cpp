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

// METHODS

void Matrix::randomInit()
{
    // std::cout << "Random init" << std::endl;
    fill_array_with_random(this->values, this->nb_rows * this->nb_columns, 0, 1);
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
    Matrix M(C.size(), L.size());

    for (int i = 0; i < (int)C.size(); i++)
    {
        for (int j = 0; j < (int)L.size(); j++)
        {
            M.set(i, j, C[i] * L[j]);
        }
    }
    return M;
}