#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <iostream>
#include <valarray>
#include "doctest/doctest.h"
#include "Network.hpp"

TEST_CASE("Network initialization ")
{
    int layer_sizes[3] = {3, 2, 3};

    Network network(layer_sizes, 3);

    std::cout << network.to_string() << std::endl;

    //network.stochasticGradientDescent(training_data, 1000, 10, 25, test_data);
    auto ouputs = network.feedForward({5, 6, 7});

    for (auto elem : ouputs)
    {
        std::cout << elem << " ";
    }

    CHECK(14 == 1 + 4 + 9);
}

// Layer 0 : 3 neurons
// Poids
// 0.000000 1.000000 2.000000
// 3.000000 4.000000 5.000000
// Biais
// 1.000000 2.000000
// Layer 1 : 2 neurons
// Poids
// 0.000000 1.000000
// 2.000000 3.000000
// 4.000000 5.000000
// Biais
// 1.000000 2.000000 3.000000
// Layer 2 : 3 neurons