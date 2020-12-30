#pragma once

#include <valarray>
#include <vector>
#include "Matrix.hpp"

class Network
{

private:
    // number of layers in the Network
    int nb_layer;

    // Pointer to array of number of neural by layer
    int *layer_sizes;

    // biases for every layer except the first one
    std::vector<std::valarray<double>> biases;

    // weights for every layer except the first one
    std::vector<Matrix> weights;

    double learning_rate;

public:
    Network(int *layer_sizes, std::size_t nb_layer);

    std::string const to_string();

    void loadTest();

    std::valarray<double>
    feedForward(const std::valarray<double> &inputs);
    //std::pair<std::valarray<std::valarray<double>>, std::valarray<Matrix>> backPropagation(const std::valarray<double> &X, const std::valarray<double> &Y);
    // std::pair<std::vector<std::valarray<double>>, std::valarray<Matrix>> backPropagation(const std::valarray<double> &X, const std::valarray<double> &Y);
    // void update_mini_batch(std::valarray<std::pair<std::valarray<double>, std::valarray<double>>> &mini_batch, double eta);
    // void stochasticGradientDescent(std::valarray<std::pair<std::valarray<double>, std::valarray<double>>> &training_datas, int epochs, int mini_batch_size, double eta, std::valarray<std::pair<std::valarray<double>, std::valarray<double>>> test_data);
    // int evaluate(const std::valarray<std::pair<std::valarray<double>, std::valarray<double>>> &test_datas);
};
