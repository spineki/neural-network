#include <valarray>
#include <utility>
#include <iostream>
#include <random>

#include "Network.hpp"
#include "Matrix.hpp"

typedef std::valarray<double> vect;

void test()
{
    std::cout << "starf" << std::endl;
}

template <typename T>
void fill_array_with_random(std::valarray<T> &array, double a = 0, double b = 1)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    std::uniform_real_distribution<T> uniform_distrib(a, b);

    for (std::size_t i = 0; i < array.size(); i++)
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

double sigmoid_prime(double x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}

int argmax(const vect &X)
{
    int arg = 0;
    int max = X[0];
    for (int i = 0; i < (int)X.size(); i++)
    {
        if (X[i] > max)
        {
            max = X[i];
            arg = i;
        }
    }

    return arg;
}

vect cost_derivative(vect output_activations, vect y)
{
    return (output_activations - y);
}

Network::Network(int *layer_sizes, int nb_layer, double learning_rate)
{
    std::cout << "⚙️  Network initialisation..." << std::endl;
    this->learning_rate = learning_rate;
    this->nb_layer = nb_layer;

    // copy of number of neurals by layers
    std::cout << "Copying the number of neural by layer..." << std::endl;
    this->layer_sizes = new int[nb_layer];
    std::copy(layer_sizes, layer_sizes + nb_layer, this->layer_sizes);

    // initializing (nb_layer -1) numbers for biaises
    // this->biases =std::valarray<double>(nb_layer-1).apply();
    std::cout << "Creating layers for biases..." << std::endl;
    this->biases.resize(nb_layer - 1);

    // initializing (nb_layer -1) number of arrays for weights
    for (int i = 0; i < nb_layer - 1; i++)
    {
        std::cout << "Initializing layer " << i << std::endl;
        int nb_neuron_input = layer_sizes[i];
        int nb_neuron_output = layer_sizes[i + 1];

        // randow weights
        std::cout << "  random weights..." << std::endl;
        this->weights.push_back(Matrix(nb_neuron_output, nb_neuron_input));
        this->weights[i].randomInit();
        std::cout << this->weights[i].to_string() << std::endl;

        //random biases
        std::cout << "  random biaises..." << std::endl;
        this->biases[i] = vect(nb_neuron_output);
        fill_array_with_random<double>(this->biases[i], 0.0, 1.0);
    }
}

// methods

/**
     * @brief passe the input throught the network and return an ouput vector
     * @param A: vector of inputs
     * @param: size, size of the inputs
     **/
std::valarray<double> Network::feedForward(const std::valarray<double> &inputs)
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
        vect bias = this->biases[layer];
        Matrix &layer_weights = this->weights.at(layer);

        outputs = (layer_weights.dot(outputs) + bias).apply(sigmoid);
    }

    return outputs;
}

std::pair<std::valarray<vect>, std::valarray<Matrix>> Network::backPropagation(const vect &X, const vect &Y)
{
    std::valarray<vect> nabla_b(this->biases.size());
    std::valarray<Matrix> nabla_w(this->weights.size());

    vect activation = X;
    std::valarray<vect> activations(this->nb_layer);
    activations[0] = X;
    std::valarray<vect> zs(this->nb_layer - 1);

    for (int i = 0; i < this->nb_layer - 1; i++)
    {
        auto w = this->weights[i];
        auto b = this->biases[i];

        vect z = w.dot(activation) + b;
        zs[i] = z;
        activation = z.apply(sigmoid);
        activations[i + 1] = activation;
    }

    vect delta = cost_derivative(activations[activations.size() - 1], Y) * (zs[zs.size() - 1]).apply(sigmoid_prime);

    nabla_b[nabla_b.size() - 1] = delta;
    nabla_w[nabla_w.size() - 1] = dot(delta, activations[activations.size() - 2]);

    for (int l = 2; l < this->nb_layer; l++)
    {
        vect z = zs[zs.size() - l];
        vect sp = z.apply(sigmoid_prime);

        delta = (this->weights[this->weights.size() - l + 1]).dot(delta) * sp;
        nabla_b[nabla_b.size() - l] = delta;
        nabla_w[nabla_w.size() - l] = dot(delta, activations[activations.size() - l - 1]);
    }

    return {nabla_b, nabla_w};
}

/**
     * @brief mets à jour les poids et les biais par déscente de gradient et rétropropagation sur un batch
     *
* */
void Network::update_mini_batch(std::valarray<std::pair<vect, vect>> &mini_batch, double eta)
{
    auto nabla_b = vect(this->biases.size());
    // auto nabla_w =

    for (auto training_data : mini_batch)
    {
        auto [delta_nabla_b, delta_nabla_w] = this->backPropagation(training_data.first, training_data.second);
    }
}

void Network::stochasticGradientDescent(std::valarray<std::pair<vect, vect>> &training_datas, int epochs, int mini_batch_size, double eta) //, const vect &test_data = {}
{

    std::random_device rd;
    std::mt19937 rng(rd());

    double nb_data = training_datas.size();

    // auto indices = vect(nb_data);
    // for (int i = 0; i < nb_data; i++)
    // {
    //     indices[i] = i;
    // }

    for (int epoch = 0; epoch < epochs; epoch++)
    {

        std::shuffle(std::begin(training_datas), std::end(training_datas), rng);

        int nb_batch = std::ceil(nb_data / mini_batch_size);

        for (int batch_num = 0; batch_num < nb_batch; batch_num += mini_batch_size)
        {
            std::valarray<std::pair<vect, vect>> mini_batch = training_datas[std::slice(batch_num, mini_batch_size, 1)];

            this->update_mini_batch(mini_batch, eta);
        }
    }
}

int Network::evaluate(const std::valarray<std::pair<vect, vect>> &test_datas)
{

    int sum = 0;
    for (auto &pair : test_datas)
    {
        vect output = this->feedForward(pair.first);

        // if same repartition between computed and expected
        if (argmax(output) == argmax(pair.second))
        {
            sum += 1;
        }
    }

    return sum;
}

Network::~Network()
{
    delete[] this->layer_sizes;
}
