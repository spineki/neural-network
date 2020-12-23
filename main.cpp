#include <iostream>
#include <algorithm>
#include <random>
#include <functional>
#include <vector>
#include <string>

/**
* Fill an array with random values in [a, b[ 
**/

template <typename T>
void fill_array_with_random(T *array, size_t size, double a = 0, double b = 1)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<T> uniform_distrib(a, b);

    std::generate(array, array + size, std::bind(uniform_distrib, rng));
}

template <typename T>
class Matrix
{

private:
    T *values;
    int nb_rows;
    int nb_columns;

public:
    Matrix(int nb_rows, int nb_columns)
    {
        this->nb_rows = nb_rows;
        this->nb_columns = nb_columns;
        this->values = new T[nb_rows * nb_columns];
    }

    void random_init()
    {
        fill_array_with_random<double>(this->values, nb_rows * nb_columns, 0, 1);
    }

    T get(int i, int j)
    {
        return this->values[j + i * nb_rows];
    }

    std::string to_string()
    {

        std::string display = "";

        for (int i = 0; i < this->nb_rows; i++)
        {
            for (int j = 0; j < this->nb_columns; j++)
            {
                display += std::to_string(this->values[i]) + " ";
            }
            display += "\n";
        }
        return display;
    }

    ~Matrix()
    {
        }
};

class Network
{

private:
    // number of layers in the Network
    int nb_layer;

    // Pointer to array of number of neural by layer
    int *layer_sizes;

    // biases for every layer except the first one
    double *biases;

    // weights for every layer except the first one
    std::vector<Matrix<double>> weights;

    double learning_rate;

public:
    Network(int layer_sizes[], int nb_layer, double learning_rate)
    {

        this->learning_rate = learning_rate;
        this->nb_layer = nb_layer;

        // copy of number of neurals by layers
        this->layer_sizes = new int[nb_layer];
        std::copy(layer_sizes, layer_sizes + nb_layer, this->layer_sizes);

        // initializing (nb_layer -1) numbers for biaises
        this->biases = new double[nb_layer - 1];
        fill_array_with_random<double>(this->biases, nb_layer - 1, 0.0, 1.0);

        // initializing (nb_layer -1) number of arrays for weights
        for (int i = 0; i < nb_layer - 1; i++)
        {
            int nb_neuron_input = layer_sizes[i];
            int nb_neuron_output = layer_sizes[i + 1];

            this->weights.push_back(Matrix<double>(nb_neuron_output, nb_neuron_input));
            this->weights[i].random_init();
            std::cout << this->weights[i].to_string() << std::endl;
        }
    }

    // methods

    ~Network()
    {
        delete[] this->layer_sizes;
        delete[] this->biases;
    }
};

int main()
{

    int layer_sizes[3] = {2, 3, 1};
    Network network(layer_sizes, 3, 0.1);

    return 0;
}