#include <valarray>
#include <utility>
#include <iostream>
#include <random>

#include "Network.hpp"
#include "Matrix.hpp"
#include <mnist_reader.hpp>
#include <cassert>

typedef std::valarray<double> vect;

// HELPERS
void print(const vect &array)
{
    for (double element : array)
    {
        printf("%f ", element);
    }
    printf("\n");
}

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

double sigmoid_prime(double x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}

vect cost_derivative(const vect &output_activations, const vect &y)
{
    return (output_activations - y);
}

std::size_t argmax(const vect &X)
{
    int arg = 0;
    int max = X[0];
    for (std::size_t i = 0; i < X.size(); i++)
    {
        if (X[i] > max)
        {
            max = X[i];
            arg = i;
        }
    }

    return arg;
}

// CONSTRUCTORS
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
// METHODS

/**
 * Tested, no worries
 **/
std::valarray<double> Network::feedForward(const std::valarray<double> &inputs)
{

    vect outputs = inputs;

    for (std::size_t i = 0; i < this->biases.size(); i++)
    {
        outputs = (this->weights[i].dot(outputs) + this->biases[i]).apply(sigmoid);
    }

    return outputs;
}

std::pair<std::vector<vect>, std::vector<Matrix>> Network::backPropagation(const vect &X, const vect &Y)
{

    // std::cout << "  ⚙️  backPropagation of " << this->weights.size() << " wieghts" << std::endl;

    std::vector<vect> nabla_b = this->biases;
    std::vector<Matrix> nabla_w = this->weights;

    assert(nabla_w.size() == nabla_b.size());

    vect activation = X;

    std::vector<vect> activations(this->nb_layer);
    activations.push_back(X);

    std::vector<vect> zs(this->nb_layer - 1);

    std::cout << "Feedforward:" << std::endl;

    for (int i = 0; i < this->nb_layer - 1; i++)
    {
        // std::cout << "layer " << i << "/" << this->nb_layer - 1 << std::endl;
        // copies de matrices inutiles, TODO
        Matrix w = this->weights[i];
        vect b = this->biases[i];

        vect z = w.dot(activation) + b;

        zs.push_back(z);
        activation = z.apply(sigmoid);
        activations.push_back(activation);
    }

    // testé jusqu'ici: TODO aller plus loin

    std::cout << "BackWard pass" << std::endl;
    vect delta = cost_derivative(activations[activations.size() - 1], Y) * (zs[zs.size() - 1]).apply(sigmoid_prime);

    nabla_b[nabla_b.size() - 1] = delta;
    nabla_w[nabla_w.size() - 1] = dot(delta, activations[activations.size() - 2]);

    for (int l = 2; l < this->nb_layer; l++)
    {
        // std::cout << "b layer " << l << "/" << this->nb_layer << std::endl;
        vect z = zs[zs.size() - l];

        vect sp = z.apply(sigmoid_prime);

        Matrix weight_T = this->weights[this->weights.size() - l + 1].transpose();

        delta = weight_T.dot(delta) * sp;
        nabla_b[nabla_b.size() - l] = delta;
        nabla_w[nabla_w.size() - l] = dot(delta, activations[activations.size() - l - 1]);
    }

    std::cout << "🎈 fin backpropagation" << std::endl;

    // //std::cout << nabla_w[0].to_string() << std::endl;

    return {nabla_b, nabla_w};
}

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
std::size_t Network::evaluate(const std::valarray<std::pair<std::valarray<double>, std::valarray<double>>> &test_datas)
{
    std::size_t sum = 0;
    for (auto &pair : test_datas)
    {
        vect output = this->feedForward(pair.first);

        // if same repartition between computed and expected
        if (argmax(output) == argmax(pair.second))
        {
            // std::cout << ">>>" << std::endl;
            // print(pair.second);
            // print(output);

            auto label = argmax(output);
            sum += 1;
        }
    }

    return sum;
}
