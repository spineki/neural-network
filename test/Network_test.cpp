#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <iostream>
#include <valarray>
#include <iomanip>

#include "doctest/doctest.h"

#include "Network.hpp"
#include "mnist_reader.hpp"

std::string folder = "../data/";
auto test_data = loadMnistTest(folder, 28);

void print_v(const std::valarray<double> &array)
{
    for (double element : array)
    {
        printf("%f ", element);
    }
    printf("\n");
}

TEST_CASE("Network initialization and feedforward")
{
    int layer_sizes[3] = {3, 2, 3};

    Network network(layer_sizes, 3);

    network.loadTest();

    //network.stochasticGradientDescent(training_data, 1000, 10, 25, test_data);
    auto outputs = network.feedForward({5, 6, 7});

    // std::valarray<double> target = {sigmoid(77), sigmoid(272), sigmoid(467)}; // sigmo of 77,  272, 467

    // verified outputs
    CHECK(outputs[0] == 0.88079707797788231449231943770428188145160675048828125);
    CHECK(outputs[1] == 0.99908894880421905693168582729413174092769622802734375);
    CHECK(outputs[2] == 0.9999938558253791409669020140427164733409881591796875);
}

TEST_CASE("Network evaluating")
{
    int layer_sizes[3] = {784, 30, 10};
    Network network(layer_sizes, 3);

    network.loadTest();

    size_t good_case = network.evaluate(test_data);

    CHECK(good_case == 8);
}

TEST_CASE("NETWORK backpropagation")
{
    int layer_sizes[3] = {784, 30, 10};
    Network network(layer_sizes, 3);

    for (auto &data : test_data)
    {
        auto [nabla_b, nabla_w] = network.backPropagation(data.first, data.second);
        for (auto &nabla : nabla_b)
        {
            print_v(nabla);
        }

        for (auto &nabla : nabla_w)
        {
            std::cout << nabla.to_string() << std::endl;
        }
    }
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