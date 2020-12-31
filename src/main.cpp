#include <iostream>
#include <algorithm>
#include <random>
#include <functional>
#include <vector>
#include <string>
#include <cmath>
#include <valarray>
#include <iterator>
#include <utility>

#include "Matrix.hpp"
#include "mnist_reader.hpp"
#include "Network.hpp"

/**
* Fill an array with random values in [a, b[ 
**/

void print_a(const std::valarray<double> &array)
{
    for (auto element : array)
    {
        std::cout << std::to_string(element) << " ";
    }
    std::cout << std::endl;
}

int main()
{

    std::string folder = "../data/";

    auto test_data = loadMnistTest(folder, 28);
    auto training_data = loadMnistTrain(folder, 28);

    printImage(training_data[0].first, 28);

    printImage(test_data[0].first, 28);

    int layer_sizes[3] = {784, 30, 10};

    std::cout << "Creating a network" << std::endl;
    Network network(layer_sizes, 3);

    auto a = test_data;
    a.resize(0);

    network.stochasticGradientDescent(training_data, 10, 10, 3.0, test_data);

    for (int i = 0; i < 10; i++)
    {
        printImage(training_data[i].first, 28);
        print_a(network.feedForward(training_data[i].first));
        print_a(training_data[i].second);
    }

    return 0;
}