#include <iostream>
#include <fstream>
#include "./NeuralNetwork.hpp"

uint32_t reverse32bit(uint32_t &input)
{
  uint32_t b1 = input & 0x000000ff;
  uint32_t b2 = input & 0x0000ff00;
  uint32_t b3 = input & 0x00ff0000;
  uint32_t b4 = input & 0xff000000;
  input = (b1 << 24) + (b2 << 8) + (b3 >> 8) + (b4 >> 24);
  return input;
}

int main()
{
  NeuralNetwork::Network<784, 16, 16, 10, float> network;

  std::ifstream images, labels;
  images.open("train-images.idx3-ubyte", std::ifstream::binary | std::ifstream::in);
  labels.open("train-labels.idx1-ubyte", std::ifstream::binary | std::ifstream::in);

  uint32_t image_magic_number, number_of_images, number_of_rows, number_of_columns;
  uint32_t label_magic_number, number_of_labels;
  images.read((char*) &image_magic_number, 4);
  reverse32bit(image_magic_number);
  images.read((char*) &number_of_images, 4);
  reverse32bit(number_of_images);
  images.read((char*) &number_of_rows, 4);
  reverse32bit(number_of_rows);
  images.read((char*) &number_of_columns, 4);
  reverse32bit(number_of_columns);

  labels.read((char*) &label_magic_number, 4);
  reverse32bit(label_magic_number);
  labels.read((char*) &number_of_labels, 4);
  reverse32bit(number_of_labels);


  std::cout << "Starting Training...\n";
  for(std::size_t o = 0; o < 5999; o++)
  {
    std::vector<std::array<float, 784>> inputs;
    std::vector<std::array<float, 10>> outputs;

    for(std::size_t i = 0; i < 9; i++)
    {
      std::array<float, 784> image;
      for(std::size_t j = 0; j < number_of_rows*number_of_columns; j++)
      { uint8_t byte; images.read((char*)&byte, 1); image[j] = float(byte)/256.0; }
      inputs.push_back(image);

      std::array<float, 10> label = {0,0,0,0,0,0,0,0,0,0};
      uint8_t byte; labels.read((char*)&byte, 1); label[byte] = 1.0;
      outputs.push_back(label);
    }

    std::cout
      << "\n(" << o << "/5998) Loss: "
      << network.calculateCost(inputs, outputs) << '\n';
    network.train(inputs, outputs, 1.0/128.0);

    /* Show Test Example */
    std::array<float, 784> test_image;
    for(std::size_t j = 0; j < number_of_rows*number_of_columns; j++)
    { uint8_t byte; images.read((char*)&byte, 1); test_image[j] = float(byte)/255.0; }

    uint8_t expected_num;
    labels.read((char*)&expected_num, 1);

    for(std::size_t y = 0; y < number_of_columns; y++)
    {
      for(std::size_t x = 0; x < number_of_rows; x++)
      {
        if(test_image[x + number_of_rows*y]*4 > 1)
        { std::cout << "##"; }
        else {std::cout << ".."; }
      }
      std::cout << '\n';
    }
    std::cout << "Correct Output:  " << (int)expected_num << '\n';
    std::cout << "Network Guess:   ";
    std::array<float, 10> network_output;
    network_output = network.getOutput(test_image);
    uint8_t num = 0; float high = 0;
    for(std::size_t i = 0; i < 10; i++)
    {
      if(network_output[i] > high)
      {
        high = network_output[i];
        num = i;
      }
    }
    std::cout << (int)num << " (Confidence " << high << "%) \n\n";
  }
}
