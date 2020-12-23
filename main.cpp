#include <iostream>
#include <algorithm>
#include <random>
#include <functional>
#include <vector>
#include <string>
#include <cmath>
#include <valarray>

/**
* Fill an array with random values in [a, b[ 
**/

template <typename T>
void fill_array_with_random(std::valarray<T> &array, double a = 0, double b = 1)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<T> uniform_distrib(a, b);

    for (int i = 0; i < array.size(); i++)
    {
        array[i] = uniform_distrib(rng);
    }
}

template <typename T>
void fill_array_with_random(T *array, std::size_t size, double a = 0, double b = 1)
{

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<T> uniform_distrib(a, b);

    for (int i = 0; i < size; i++)
    {
        array[i] = uniform_distrib(rng);
    }
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

template <typename T>
void print(std::valarray<T> &array)
{
    for (T element : array)
    {
        std::cout << std::to_string(element) << "" << std::endl;
    }
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

    // METHODS
    void random_init()
    {

        fill_array_with_random<double>(this->values, nb_rows * nb_columns, 0, 1);
    }

    T get(int i, int j)
    {
        return this->values[j + i * nb_columns];
    }

    std::string to_string()
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
    std::valarray<double> dot(const std::valarray<double> &X)
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
    std::valarray<double> biases;

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
        // this->biases =std::valarray<double>(nb_layer-1).apply();

        this->biases.resize(nb_layer - 1);
        fill_array_with_random<double>(this->biases, 0.0, 1.0);

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

    /**
     * @brief passe the input throught the network and return an ouput vector
     * @param A: vector of inputs
     * @param: size, size of the inputs
     **/
    std::valarray<double> feedForward(const std::valarray<double> &inputs)
    {
        /*
        """Return the output of the network if "a" is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
        */

        std::valarray<double> outputs(inputs);

        for (int layer = 0; layer < this->nb_layer - 1; layer++)
        {
            double bias = this->biases[layer];
            Matrix<double> &layer_weights = this->weights.at(layer);

            outputs = (layer_weights.dot(outputs) + bias).apply(sigmoid);

            // temp = np.dot(w, a) + b

            //outputs[i] = this->weights[i].dot(inputs, size) + this->biases[i];
        }

        return outputs;
    }

    ~Network()
    {
        delete[] this->layer_sizes;
    }
};

int main()
{

    int layer_sizes[2] = {2, 1};

    Network network(layer_sizes, 2, 0.1);

    std::valarray<double> inputs = {1, 2};

    auto output = network.feedForward(inputs);

    print(output);

    return 0;
}