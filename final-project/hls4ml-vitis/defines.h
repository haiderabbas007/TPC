#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<8,3>, 1*1> input_t;
typedef nnet::array<ap_fixed<8,3>, 1*1> layer12_t;
typedef ap_fixed<8,3> model_default_t;
typedef nnet::array<ap_fixed<10,4>, 2*1> layer2_t;
typedef ap_fixed<8,2> conv1_weight_t;
typedef ap_fixed<8,2> conv1_bias_t;
typedef nnet::array<ap_fixed<8,3>, 2*1> layer3_t;
typedef ap_fixed<18,8> conv1_relu_table_t;
typedef nnet::array<ap_fixed<8,3>, 2*1> layer4_t;
typedef nnet::array<ap_fixed<8,3>, 2*1> layer13_t;
typedef nnet::array<ap_fixed<10,4>, 4*1> layer5_t;
typedef ap_fixed<8,2> conv2_weight_t;
typedef ap_fixed<8,2> conv2_bias_t;
typedef nnet::array<ap_fixed<8,3>, 4*1> layer6_t;
typedef ap_fixed<18,8> conv2_relu_table_t;
typedef nnet::array<ap_fixed<8,3>, 4*1> layer7_t;
typedef nnet::array<ap_fixed<8,3>, 4*1> layer8_t;
typedef nnet::array<ap_fixed<10,4>, 4*1> layer9_t;
typedef ap_fixed<8,2> dense1_weight_t;
typedef ap_fixed<8,2> dense1_bias_t;
typedef ap_uint<1> layer9_index;
typedef nnet::array<ap_fixed<8,3>, 4*1> layer10_t;
typedef ap_fixed<18,8> dense1_relu_table_t;
typedef nnet::array<ap_fixed<10,4>, 1*1> result_t;
typedef ap_fixed<8,2> output_weight_t;
typedef ap_fixed<8,2> output_bias_t;
typedef ap_uint<1> layer11_index;

// hls-fpga-machine-learning insert emulator-defines


#endif
