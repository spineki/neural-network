#include <string>
#include <valarray>
#include <random>

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

    for (int i = 0; i < size; i++)
    {
        array[i] = uniform_distrib(rng);
    }
}

Matrix::Matrix(int nb_rows, int nb_columns)
{
    this->nb_rows = nb_rows;
    this->nb_columns = nb_columns;
    this->values = new double[nb_rows * nb_columns];
}

// METHODS

void Matrix::random_init()
{

    fill_array_with_random(this->values, nb_rows * nb_columns, 0, 1);
}

double Matrix::get(int i, int j)
{
    return this->values[j + i * nb_columns];
}

void Matrix::set(int i, int j, double value)
{
    this->values[j + i * nb_columns] = value;
}

std::string Matrix::to_string()
{
    std::string display = ">>>\n";

    for (int i = 0; i < this->nb_rows; i++)
    {
        for (int j = 0; j < this->nb_columns; j++)
        {
            display += std::to_string(this->get(i, j)) + " ";
        }
        display += "\n";
    }
    return display;
}

// Compute

std::valarray<double> Matrix::dot(const std::valarray<double> &X)
{
    std::valarray<double> Y(X);

    for (int i = 0; i < this->nb_rows; i++)
    {
        for (int j = 0; j < this->nb_columns; j++)
        {
            Y[i] += this->get(i, j) * X[j];
        }
    }
    return Y;
}

// Destructor
Matrix::~Matrix()
{
    delete[] this->values;
}
