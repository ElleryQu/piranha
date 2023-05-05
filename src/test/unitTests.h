
#pragma once

#define EXP_TIMES 1

#include <gtest/gtest.h>

#include <iostream>
#include <math.h>
#include <random>
#include <vector>
#include <random>

#include <cublas_v2.h>

#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/core_io.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_splitk_parallel.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/tensor_view_io.h>

#include "../globals.h"

#include "../nn/LNConfig.h"
#include "../nn/LNLayer.h"
#include "../nn/CNNConfig.h"
#include "../nn/CNNLayer.h"
#include "../nn/FCConfig.h"
#include "../nn/FCLayer.h"
#include "../nn/MaxpoolConfig.h"
#include "../nn/MaxpoolLayer.h"
#include "../nn/AveragepoolConfig.h"
#include "../nn/AveragepoolLayer.h"
#include "../nn/NeuralNetConfig.h"
#include "../nn/NeuralNetwork.h"
#include "../nn/ReLUConfig.h"
#include "../nn/ReLULayer.h"
#include "../nn/ResLayerConfig.h"
#include "../nn/ResLayer.h"

#include "../util/Profiler.h"
#include "../util/util.cuh"
#include "../gpu/bitwise.cuh"
#include "../gpu/convolution.cuh"
#include "../gpu/DeviceData.h"
#include "../gpu/matrix.cuh"
#include "../gpu/gemm.cuh"
#include "../gpu/conv.cuh"

#define writeProfile(pf, prot, func, input_size, question_scale,\
    online_comp_time, online_comm_rounds, online_comm_tx_size, \
    online_comm_rx_size, online_comm_time) \
    {\
        stringstream _os;\
        _os << prot << '\t' << func << '\t' << input_size << '\t' \
        << question_scale << '\t' << online_comp_time << '\t' << online_comm_rounds \
        << '\t' << online_comm_tx_size << '\t' << online_comm_rx_size << '\t'\
        << online_comm_time << std::endl;\
        pf << _os.str();\
    }

extern int partyNum;

extern size_t INPUT_SIZE, LAST_LAYER_SIZE, WITH_NORMALIZATION;
extern void getBatch(std::ifstream &, std::istream_iterator<double> &, std::vector<double> &);

#ifndef PF_PATH
#define PF_PATH
extern std::string profiling_path;
extern std::ofstream pf;
#endif

extern std::default_random_engine generator;
extern void random_vector(std::vector<double> &v, int size);

int runTests(int argc, char **argv);

// template<typename T1, typename T2>
// void writeProfile(
//     std::ofstream& pf, 
//     std::string prot, 
//     std::string func,
//     T1 input_size,
//     string question_scale,
//     T2 online_comp_time,
//     T1 online_comm_rounds,
//     T2 online_comm_tx_size,
//     T2 online_comm_rx_size,
//     T2 online_comm_time
// );