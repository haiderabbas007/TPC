#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    hls::stream<input_t> &input_layer,
    hls::stream<result_t> &layer11_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=input_layer,layer11_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<conv1_weight_t, 60>(w2, "w2.txt");
        nnet::load_weights_from_txt<conv1_bias_t, 4>(b2, "b2.txt");
        nnet::load_weights_from_txt<conv2_weight_t, 672>(w5, "w5.txt");
        nnet::load_weights_from_txt<conv2_bias_t, 8>(b5, "b5.txt");
        nnet::load_weights_from_txt<dense1_weight_t, 64>(w9, "w9.txt");
        nnet::load_weights_from_txt<dense1_bias_t, 8>(b9, "b9.txt");
        nnet::load_weights_from_txt<output_weight_t, 8>(w11, "w11.txt");
        nnet::load_weights_from_txt<output_bias_t, 1>(b11, "b11.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer12_t> layer12_out("layer12_out");
    #pragma HLS STREAM variable=layer12_out depth=2288

    hls::stream<layer2_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=2000

    hls::stream<layer3_t> layer3_out("layer3_out");
    #pragma HLS STREAM variable=layer3_out depth=2000

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=500

    hls::stream<layer13_t> layer13_out("layer13_out");
    #pragma HLS STREAM variable=layer13_out depth=672

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=500

    hls::stream<layer6_t> layer6_out("layer6_out");
    #pragma HLS STREAM variable=layer6_out depth=500

    hls::stream<layer7_t> layer7_out("layer7_out");
    #pragma HLS STREAM variable=layer7_out depth=60

    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=1

    hls::stream<layer9_t> layer9_out("layer9_out");
    #pragma HLS STREAM variable=layer9_out depth=1

    hls::stream<layer10_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=1

    nnet::zeropad2d_cl<input_t, layer12_t, config12>(input_layer, layer12_out); // zp2d_conv1

    nnet::conv_2d_cl<layer12_t, layer2_t, config2>(layer12_out, layer2_out, w2, b2); // conv1

    nnet::thresholded_relu<layer2_t, model_default_t, layer3_t, thresholdedrelu_config3>(layer2_out, 0.0, layer3_out); // relu1

    nnet::pooling2d_cl<layer3_t, layer4_t, config4>(layer3_out, layer4_out); // pool1

    nnet::zeropad2d_cl<layer4_t, layer13_t, config13>(layer4_out, layer13_out); // zp2d_conv2

    nnet::conv_2d_cl<layer13_t, layer5_t, config5>(layer13_out, layer5_out, w5, b5); // conv2

    nnet::thresholded_relu<layer5_t, model_default_t, layer6_t, thresholdedrelu_config6>(layer5_out, 0.0, layer6_out); // relu2

    nnet::pooling2d_cl<layer6_t, layer7_t, config7>(layer6_out, layer7_out); // pool2

    nnet::global_pooling2d_cl<layer7_t, layer8_t, config8>(layer7_out, layer8_out); // global_avg_pool

    nnet::dense<layer8_t, layer9_t, config9>(layer8_out, layer9_out, w9, b9); // dense1

    nnet::thresholded_relu<layer9_t, model_default_t, layer10_t, thresholdedrelu_config10>(layer9_out, 0.0, layer10_out); // relu_dense1

    nnet::dense<layer10_t, result_t, config11>(layer10_out, layer11_out, w11, b11); // output

}

