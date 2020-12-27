#pragma once

#include <valarray>
#include "Matrix.hpp"

class Network
{

private:
    // number of layers in the Network
    int nb_layer;

    // Pointer to array of number of neural by layer
    int *layer_sizes;

    // biases for every layer except the first one
    std::valarray<std::valarray<double>> biases;

    // weights for every layer except the first one
    std::vector<Matrix> weights;

    double learning_rate;

public:
    Network(int *layer_sizes, int nb_layer, double learning_rate);
    std::valarray<double> feedForward(const std::valarray<double> &inputs);
    std::pair<std::valarray<std::valarray<double>>, std::valarray<Matrix>> backPropagation(const std::valarray<double> &X, const std::valarray<double> &Y);
    void update_mini_batch(std::valarray<std::pair<std::valarray<double>, std::valarray<double>>> &mini_batch, double eta);
    void stochasticGradientDescent(std::valarray<std::pair<std::valarray<double>, std::valarray<double>>> &training_datas, int epochs, int mini_batch_size, double eta);
    int evaluate(const std::valarray<std::pair<std::valarray<double>, std::valarray<double>>> &test_datas);
    ~Network();
};