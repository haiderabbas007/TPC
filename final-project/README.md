# PHYS 476 Final Project

**Project:** FPGA-Based CNN Trigger Prototype for Microcurler Detection in TPC Detectors  
**Student:** Muhammad Haider Abbas  
**Course:** PHYS 476 Electronics for Physicists  
**Date:** May 15, 2026

This repository contains the final project materials, including data-processing scripts, CNN training/model-preparation files, hls4ml/Vitis HLS outputs, Nexys Video FPGA integration files, Ethernet-extension work, Zynq/PYNQ validation files, and final results/evidence.

## Folder structure

* 1\_data\_processing\_dataset\_preparation/
* 2\_dataset\_files/
* 3\_cnn\_training\_model\_preparation/
* 4\_hls4ml\_vitis\_hls\_generated\_files/
* 5\_nexys\_video\_fpga\_board\_integration/
* 6\_ethernet\_extension\_files/
* 7\_zynq\_pynq\_validation\_files/
* 8\_results\_evidence\_files/

## Main result

The final Zynq/PYNQ validation processed 300 held-out test sub-hitmaps. The FPGA implementation matched the quantized-input Keras model with:

* Accuracy: 225/300 = 75.0%
* Average DMA/FPGA latency: approximately 1.605 ms per event

