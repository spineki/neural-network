#pragma once

#include <valarray>

//std::valarray<std::pair<std::valarray<double>, std::valarray<double>>> loadMnist(std::string full_path, int image_size = 28, int number_of_images = 10'000);

std::valarray<std::pair<std::valarray<double>, std::valarray<double>>> loadMnistTrain(std::string folder_path, int image_size = 28);
std::valarray<std::pair<std::valarray<double>, std::valarray<double>>> loadMnistTest(std::string folder_path, int image_size = 28);

// void printImage(std::valarray<double> picture, int size);
void printImage(std::valarray<double> &picture, int size);
