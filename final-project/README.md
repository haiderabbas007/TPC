# PHYS 476 Final Project

**Project:** FPGA-Based CNN Trigger Prototype for Microcurler Detection in TPC Detectors  
**Student:** Muhammad Haider Abbas  
**Course:** PHYS 476 Electronics for Physicists  
**Date:** May 15, 2026

This repository contains the final project materials, including data-processing scripts, CNN training/model-preparation files, hls4ml/Vitis HLS outputs, Nexys Video FPGA integration files, Ethernet-extension work, Zynq/PYNQ validation files, and final results/evidence.

## Folder structure

* data-processing/
* dataset-files/
* cnn-training/
* hls4ml-vitis/
* fpga-integrate/
* ethernet/
* zynq-files/
* results

## Main result

The final Zynq/PYNQ validation processed 300 held-out test sub-hitmaps. The FPGA implementation matched the quantized-input Keras model with:

* Accuracy: 225/300 = 75.0%
* Average DMA/FPGA latency: approximately 1.605 ms per event

