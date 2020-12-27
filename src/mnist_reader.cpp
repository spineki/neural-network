#include <iostream>
#include <fstream>
#include <string>
#include <fstream>
#include <utility>
#include <valarray>
#include <error.h>
#include <sstream>
#include <filesystem>

void printImage(std::valarray<double> &picture, int size)
{

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            std::cout << (picture[i * size + j] > 125 ? ' ' : '#');
        }
        printf("\n");
    }
}

typedef std::valarray<double> vect;
std::valarray<std::pair<vect, vect>> loadMnistFile(std::string file_path, int image_size = 28, int number_of_images = 10)
{
    std::cout << std::filesystem::current_path() << std::endl;

    std::cout << "readind mnist located at " << file_path << std::endl;

    std::valarray<std::pair<vect, vect>> dataset(number_of_images);

    int added_pictures = 0;

    std::ifstream file(file_path);

    std::string line;

    //
    if (file.is_open())
    {
        // going throught the whole file
        while (std::getline(file, line))
        {
            // just taking as many pictures as needed
            if (added_pictures >= number_of_images)
            {
                return dataset;
            }

            std::stringstream ss(line);
            int val;
            int i = 0;
            int j = 0;

            int label;
            ss >> label;
            ss.ignore();
            std::valarray<double> label_array(10);

            label_array[label] = 1.0;

            // taking space for the picture data
            std::valarray<double> picture(image_size * image_size);
            while (ss >> val)
            {
                picture[j + i * image_size] = val;
                j++;
                if (j >= image_size)
                {
                    j = 0;
                    i++;
                }
                // If the next token is a comma, ignore it and move on
                if (ss.peek() == ',')
                    ss.ignore();
            }

            dataset[added_pictures] = std::make_pair(picture, label_array);

            //printf("\n\n%d\n", label);
            //printImage(picture, image_size);

            added_pictures += 1;
        }
        file.close();
    }
    else
    {
        throw std::runtime_error("Cannot open file `" + file_path + "`!");
    }

    return dataset;
}

std::valarray<std::pair<vect, vect>> loadMnistTrain(std::string folder_path, int image_size = 28)
{
    return loadMnistFile(folder_path + "mnist_train.csv", image_size, 6'000); //60'000
}

std::valarray<std::pair<vect, vect>> loadMnistTest(std::string folder_path, int image_size = 28)
{
    return loadMnistFile(folder_path + "mnist_test.csv", image_size, 1'000); //10'0000
}
