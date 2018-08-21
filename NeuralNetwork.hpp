#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

// size_t
#include <cstdint>

// Rand
#include <cstdlib>
#include <ctime>

// Lists
#include <vector>
#include <array>

// Definitions
#define abs(x) ((x > 0) ? x : -x)
#define sigmoid(x) ((x/(1.0 + abs(x)) + 1.0) * 0.5)

namespace NeuralNetwork
{
  /********************/
  /*** NEURON CLASS ***/
  /********************/

  template<std::size_t input_size, class T>
  struct Neuron
  {
    Neuron()
    {
      for(std::size_t i = 0; i < input_size; i++)
      { mult[i] = ((T(rand()%0x10000)/T(0x8000))-T(1)); }
      value = ((T(rand()%0x10000)/T(0x8000))-T(1));
      bias = ((T(rand()%0x10000)/T(0x8000))-T(1));
    }

    template<std::size_t input_input_size>
    T setValue(const std::array<Neuron<input_input_size, T>, input_size> inputs)
    {
      value = 0;
      for(std::size_t i = 0; i < input_size; i++)
      { value += inputs[i].value*mult[i]; }
      value += bias;

      value = sigmoid(value);

      return value;
    }

    T setValueFloat(const std::array<T, input_size> inputs)
    {
      value = 0;
      for(std::size_t i = 0; i < input_size; i++)
      { value += inputs[i]*mult[i]; }
      value += bias;

      value = sigmoid(value);
      return value;
    }

    T value;
    T bias;
    T mult[input_size];
  };




  /*********************/
  /*** NETWORK CLASS ***/
  /*********************/

  template
  <
    std::size_t input_size,
    std::size_t layer1_size,
    std::size_t layer2_size,
    std::size_t output_size,
    class T = float
  >
  class Network
  {
  public: // Methods
    Network()
    { srand(time(NULL)); }

    std::array<T, output_size> getOutput(const std::array<T, input_size> input)
    {
      for(std::size_t i = 0; i < layer1_size; i++)
      { layer1[i].setValueFloat(input); }

      for(std::size_t i = 0; i < layer2_size; i++)
      { layer2[i].setValue(layer1); }

      std::array<T, output_size> out;
      for(std::size_t i = 0; i < output_size; i++)
      { out[i] = output[i].setValue(layer2); }

      return out;
    }

    T calculateCost
    (
      const std::vector<std::array<T, input_size>> input_layer,
      const std::vector<std::array<T, output_size>> output_layer
    )
    {
      if(input_layer.size() > output_layer.size()){ return 0; }
      std::size_t test_size =
      (input_layer.size() > output_layer.size())
      ? output_layer.size() :  input_layer.size();

      T cost = 0;
      for(std::size_t test = 0; test < test_size; test++)
      {
        for(std::size_t i = 0; i < layer1_size; i++)
        { layer1[i].setValueFloat(input_layer[test]); }

        for(std::size_t i = 0; i < layer2_size; i++)
        { layer2[i].setValue(layer1); }

        for(std::size_t i = 0; i < output_size; i++)
        { output[i].setValue(layer2); }

        for(std::size_t i = 0; i < output_size; i++)
        {
          T diff = output[i].value - output_layer[test][i];
          cost += diff*diff;
        }
      }

      return cost;
    }

    T train
    (
      const std::vector<std::array<T, input_size>> input_layer,
      const std::vector<std::array<T, output_size>> output_layer,
      const T step_size = 0.001
    )
    {
      for(std::size_t i = 0; i < layer1_size; i++)
      {
        for(std::size_t j = 0; j < input_size; j++)
        { optimizeWeight(input_layer, output_layer, layer1[i].mult[j], step_size); }
        optimizeWeight(input_layer, output_layer, layer1[i].bias, step_size);
      }

      for(std::size_t i = 0; i < layer2_size; i++)
      {
        for(std::size_t j = 0; j < layer1_size; j++)
        { optimizeWeight(input_layer, output_layer, layer2[i].mult[j], step_size); }
        optimizeWeight(input_layer, output_layer, layer2[i].bias, step_size);
      }

      for(std::size_t i = 0; i < output_size; i++)
      {
        for(std::size_t j = 0; j < layer2_size; j++)
        { optimizeWeight(input_layer, output_layer, output[i].mult[j], step_size); }
        optimizeWeight(input_layer, output_layer, output[i].bias, step_size);
      }

      return calculateCost(input_layer, output_layer);
    }

  public: // Variables
    std::array<Neuron<input_size, T>,  layer1_size> layer1;
    std::array<Neuron<layer1_size, T>, layer2_size> layer2;
    std::array<Neuron<layer2_size, T>, output_size> output;

  private: // Methods
    void optimizeWeight
    (
      const std::vector<std::array<T, input_size>> input_layer,
      const std::vector<std::array<T, output_size>> output_layer,
      T& weight, const T step_size
    )
    {
      const T startCost = calculateCost(input_layer, output_layer);
      const T startWeight = weight;

      weight += step_size;
      const T highCost = calculateCost(input_layer, output_layer);
      const T highWeight = weight;

      weight -= step_size;
      weight -= step_size;
      const T lowCost = calculateCost(input_layer, output_layer);

      if(startCost < highCost)
      {
        if(startCost < lowCost)
        { weight = startWeight; }
      } else
      {
        if(highCost < lowCost)
        { weight = highWeight; }
      }
    }
  };
}

#endif
