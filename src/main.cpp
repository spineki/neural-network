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

template <typename T>
void print(std::valarray<T> &array)
{
    for (T element : array)
    {
        std::cout << std::to_string(element) << "" << std::endl;
    }
}

int main()
{

    std::string folder = "../data/";

    auto training_data = loadMnistTest(folder, 28);
    auto test_data = loadMnistTrain(folder, 28);

    printImage(training_data[10].first, 28);

    printImage(test_data[10].first, 28);

    int layer_sizes[3] = {784, 30, 10};

    std::cout << "Creating a network" << std::endl;
    Network network(layer_sizes, 3, 0.1);

    network.stochasticGradientDescent(training_data, 30, 10, 3.0, test_data);

    return 0;
}