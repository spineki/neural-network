#include <valarray>
#include <utility>
#include <iostream>
#include <random>

#include "Network.hpp"
#include "Matrix.hpp"
#include <mnist_reader.hpp>
#include <cassert>

typedef std::valarray<double> vect;

void fill_valarray_random(vect &array, double a = 0, double b = 1)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    std::uniform_real_distribution<double> uniform_distrib(a, b);

    for (std::size_t i = 0; i < array.size(); i++)
    {
        array[i] = uniform_distrib(rng);
    }
}

std::string vect_to_string(const vect &x)
{
    std::string display = "";
    for (auto elem : x)
    {
        display += std::to_string(elem) + " ";
    }
    return display;
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

// Constructor
// tested
Network::Network(int *layer_sizes, std::size_t nb_layer)
{
    std::cout << "📚  Network initialisation..." << std::endl;
    this->nb_layer = nb_layer;

    std::cout << "Copying the number of neural by layer..." << std::endl;
    this->layer_sizes = new int[nb_layer];
    std::copy(layer_sizes, layer_sizes + nb_layer, this->layer_sizes);

    std::cout << "Creating layers for biases: " << nb_layer - 1 << std::endl;
    this->biases.resize(nb_layer - 1);

    // initializing (nb_layer -1) number of arrays for weights
    for (int i = 0; i < nb_layer - 1; i++)
    {
        int nb_neuron_input = layer_sizes[i];
        int nb_neuron_output = layer_sizes[i + 1];

        std::cout << "Initializing layer " << i << ": " << nb_neuron_input << " -> " << nb_neuron_output << std::endl;

        // randow weights
        std::cout << "  random weights..." << std::endl;
        this->weights.push_back(Matrix(nb_neuron_output, nb_neuron_input));

        this->weights[i].randomInit();

        //random biases
        std::cout << "  random biaises..." << std::endl;
        this->biases[i] = vect(nb_neuron_output);

        // to do, set correct filling again

        // fill_valarray_random(this->biases[i], -1.0, 1.0);
        for (int j = 0; j < this->biases[i].size(); j++)
        {
            this->biases[i][j] = j + 1;
        }
    }
}

void Network::loadTest()
{
    // test biases
    for (int i = 0; i < this->nb_layer - 1; i++)
    {
        for (int j = 0; j < this->biases[i].size(); j++)
        {
            this->biases[i][j] = j + 1;
            this->weights[i].testInit();
        }
    }
}
//methods

// tested
std::valarray<double> Network::feedForward(const std::valarray<double> &inputs)
{

    vect outputs = inputs;

    for (std::size_t i = 0; i < this->biases.size(); i++)
    {
        outputs = (this->weights[i].dot(outputs) + this->biases[i]); //.apply(sigmoid);
    }

    return outputs;
}

// tested
std::string const Network::to_string()
{
    std::string display = "==========\nnb_layers " + std::to_string(this->nb_layer) + "\n";

    for (int i = 0; i < this->nb_layer - 1; i++)
    {
        display += "Layer " + std::to_string(i) + " : " + std::to_string(this->layer_sizes[i]) + " neurons\n";
        display += "Poids\n";
        display += this->weights[i].to_string();
        display += "Biais\n";
        display += vect_to_string(this->biases[i]) + "\n";
    }
    display += "Layer " + std::to_string(this->nb_layer - 1) + " : " + std::to_string(this->layer_sizes[this->nb_layer - 1]) + " neurons\n";

    display += "\n==========";
    return display;
}

// //std::pair<std::valarray<std::valarray<double>>, std::valarray<Matrix>> backPropagation(const std::valarray<double> &X, const std::valarray<double> &Y);
// std::pair<std::vector<std::valarray<double>>, std::valarray<Matrix>> backPropagation(const std::valarray<double> &X, const std::valarray<double> &Y);
// void update_mini_batch(std::valarray<std::pair<std::valarray<double>, std::valarray<double>>> &mini_batch, double eta);
// void stochasticGradientDescent(std::valarray<std::pair<std::valarray<double>, std::valarray<double>>> &training_datas, int epochs, int mini_batch_size, double eta, std::valarray<std::pair<std::valarray<double>, std::valarray<double>>> test_data);
// int evaluate(const std::valarray<std::pair<std::valarray<double>, std::valarray<double>>> &test_datas);
