
#pragma once

#include <limits>
#include <thrust/device_vector.h>

#include "DeviceData.h"
#include "../util/util.cuh"

namespace kernel {

/// @brief 通用im2row算子。把图片扩展成矩阵。输入DinHW. 输出Conv_windowFFDin.
template<typename T>
__global__ void im2row(T *im, T *output,
        int imageWidth, int imageHeight,
        int filterSize, int Din, int stride, int padding) {
    
    // 这里是对整个线程池的刻画。
    // 线程所在地。y轴上表示第几个卷积窗口。
    int CONV_WINDOW_IDX = blockIdx.y*blockDim.y+threadIdx.y;
    int IM_INDEX = blockIdx.x*blockDim.x+threadIdx.x;

    // 从宽度来看，给定卷积核尺寸和图片尺寸，卷积核在宽度上需要漫游多少次？
    int widthKernels = ((imageWidth - filterSize + (2*padding))/stride)+1;
    int heightKernels = ((imageHeight - filterSize + (2*padding))/stride)+1;

    // 如果当前线程所在位置在x轴上大于等于输入的图片数量，或是在y轴上大于等于卷积采样下的总大小。
    // 那就说明这个线程没活。歇着吧您。
    if (IM_INDEX >= Din ||
        CONV_WINDOW_IDX >= widthKernels * heightKernels) {
        return;
    }

    // 输出索引：（卷积窗口位置*输入图象数+图像索引）*过滤器面积。
    // 从排列上看，处于相同卷积窗口位置的不同图像被排列在一起、
    int outputIdx = (CONV_WINDOW_IDX * filterSize * filterSize * Din) + (IM_INDEX * filterSize * filterSize);

    // find top left corner of current convolution in image coordinates
    // 查找当前卷积窗口左上角在图片上的位置。（被填充的部分被记为负数）
    int baseRow = ((CONV_WINDOW_IDX / widthKernels) * stride) - padding;
    int baseCol = ((CONV_WINDOW_IDX % widthKernels) * stride) - padding;

    // 展开矩阵，将结果填充到output中。行优先。
    for(int r = 0; r < filterSize; r++) {
        for(int c = 0; c < filterSize; c++) {

            int y = baseRow + r;
            int x = baseCol + c;

            if (y < 0 || y >= imageHeight ||
                x < 0 || x >= imageWidth) { 

                output[outputIdx++] = 0; // pad with zeros

            } else {

                int imY = (IM_INDEX * imageHeight) + y; // row offset based on image number
                int imX = x; // always same column

                output[outputIdx++] = im[imY * imageWidth + imX];
            }
        }
    }
}

/// @brief maxpool专用的im2row。输入NHWC. 输出NWHC
/// @param Din 通道数。
template<typename T>
__global__ void maxpool_im2row(T *im, T *output,
        int imageWidth, int imageHeight,
        int filterSize, int Din, int batchSize, int stride, int padding) {

    // each thread:
    //   IMG_CHANNEL_IDX - which image+channel are we looking at
    //   CONV_WINDOW_IDX - which location in the image are we looking at
    // 图片通道索引。线程池y轴。
    int IMG_CHANNEL_IDX = blockIdx.y*blockDim.y+threadIdx.y;
    // 卷积窗口索引。线程池x轴。
    int CONV_WINDOW_IDX = blockIdx.x*blockDim.x+threadIdx.x;
    // 这里可能是因为输入图像格式为NHWC，CUDA线程按列优先排布，因此在y轴上是图片数*通道。

    int widthKernels = ((imageWidth - filterSize + (2*padding))/stride)+1;
    int heightKernels = ((imageHeight - filterSize + (2*padding))/stride)+1;

    if (IMG_CHANNEL_IDX >= batchSize * Din ||
        CONV_WINDOW_IDX >= widthKernels * heightKernels) {
        return;
    }

    int poolSize = filterSize * filterSize;

    // 这里提示一点：线程池排布和图片排布不是一回事、这里要求线程池的纵轴包含NC。
    int batch = IMG_CHANNEL_IDX / Din;
    int channel = IMG_CHANNEL_IDX % Din;

    // find top left corner of current convolution in image coordinates
    // 为什么这里选择除卷积窗口高度？这里规定卷积窗口索引是列优先排布的。卷积窗口索引在x轴。
    // x轴列优先，y轴行优先。为什么？
    int baseCol = ((CONV_WINDOW_IDX / heightKernels) * stride) - padding;
    int baseRow = ((CONV_WINDOW_IDX % heightKernels) * stride) - padding;

    // find the right row in the output
    // （图片标号*图片面积*通道数+池化窗口标号*通道数+通道标号）*池化核大小。
    int outputIdx = (batch * (widthKernels * heightKernels * Din * poolSize)) + (CONV_WINDOW_IDX * (Din * poolSize)) + (channel * poolSize);

    // 展开矩阵。列优先。
    for(int c = 0; c < filterSize; c++) {
        for(int r = 0; r < filterSize; r++) {

            int y = baseRow + r;
            int x = baseCol + c;

            if (y < 0 || y >= imageHeight ||
                x < 0 || x >= imageWidth) { 

                // output[outputIdx++] = 0;
                // 在maxpool里，不保留pad空的区域。为什么？

            } else {

                // x (col), y (row), batch, channel
                int src_idx = (batch * imageHeight * imageWidth * Din) + (x * imageHeight * Din) + (y * Din) + channel;

                output[outputIdx++] = im[src_idx];
            }
        }
    }
}

template<typename T>
__global__ void maxpool_row2im(T *rows, T *output,
        int imageHeight, int imageWidth,
        int filterSize, int Din, int batchSize, int stride) {

    // each col is a value in a flattened convolution window (row)
    //     int yThreads = filterSize * filterSize; 
    // each row is a conv view
    //     int xThreads = Din * widthKernels * heightKernels;
    
    // each thread adds one window to :
    //   CONV_WINDOW_IDX - which window we're extracting
    //   VAL_IDX - which value in the window we want
    
    int CONV_WINDOW_IDX = blockIdx.x*blockDim.x+threadIdx.x;
    int VAL_IDX = blockIdx.y*blockDim.y+threadIdx.y;

    int widthKernels = ((imageWidth - filterSize)/stride)+1;
    int heightKernels = ((imageHeight - filterSize)/stride)+1;

    if (VAL_IDX >= filterSize * filterSize ||
        CONV_WINDOW_IDX >= batchSize * Din * widthKernels * heightKernels) {
        return;
    }

    int batch = CONV_WINDOW_IDX / (widthKernels * heightKernels * Din);

    int windowIdx = (CONV_WINDOW_IDX % (widthKernels * heightKernels * Din)) / Din;
    int windowCol = windowIdx / heightKernels;
    int windowRow = windowIdx % heightKernels;

    int channel = (CONV_WINDOW_IDX % (widthKernels * heightKernels * Din)) % Din;

    int baseCol = (windowCol * stride);
    int baseRow = (windowRow * stride);

    int valCol = baseCol + (VAL_IDX / filterSize);
    int valRow = baseRow + (VAL_IDX % filterSize);

    int outputIdx = (batch * imageHeight * imageWidth * Din) + (valCol * imageHeight * Din) + (valRow * Din) + channel;

    atomicAdd(&(output[outputIdx]), rows[CONV_WINDOW_IDX * (filterSize * filterSize) + VAL_IDX]);
}

template<typename T>
__global__ void expandFilter(T *filter, T *out,
        int rows, int cols, int stride) {
    
    int ROW_IDX = blockIdx.y*blockDim.y+threadIdx.y;
    int COL_IDX = blockIdx.x*blockDim.x+threadIdx.x;

    int expandedWidth = cols + (cols - 1)*stride;
    int expandedHeight = rows + (rows - 1)*stride;

    if (ROW_IDX >= expandedHeight || COL_IDX >= expandedWidth) {
        return;
    }

    if (ROW_IDX % (stride+1) == 0 && COL_IDX % (stride+1) == 0) {
        out[ROW_IDX * expandedWidth + COL_IDX] = filter[ROW_IDX/(stride+1) * cols + COL_IDX/(stride+1)];
    } else {
        out[ROW_IDX * expandedWidth + COL_IDX] = 0;
    }
}

template<typename T, typename U>
__global__ void stride_pad(T *in, T *out, int rows, int cols, int stride, int pad, U pad_value) {

    int ROW_IDX = blockIdx.y*blockDim.y+threadIdx.y;
    int COL_IDX = blockIdx.x*blockDim.x+threadIdx.x;

    int input_idx = (ROW_IDX * cols * stride) + (COL_IDX * stride);
    int output_idx = (ROW_IDX * cols * (stride + pad)) + (COL_IDX * (stride + pad));

    for (int i = 0; i < stride; i++) {
        out[output_idx++] = in[input_idx++]; 
    }

    for (int i = 0; i < pad; i++) {
        out[output_idx++] = static_cast<T>(pad_value); 
    }
}


template<typename T>
__global__ void averagepool_expand_delta(T *in, T *out, int inputSize, int Din, int poolSize) {
    
    int DELTA_IDX = blockIdx.x*blockDim.x+threadIdx.x;

    if (DELTA_IDX >= inputSize) {
        return;
    }

    int batch = DELTA_IDX / Din;
    int channel = DELTA_IDX % Din;
    for (int i = (batch * Din * poolSize) + channel; i < (batch + 1) * Din * poolSize; i += Din) {
        out[i] = in[DELTA_IDX];
    }
}

}

namespace gpu {

template<typename T>
void im2row(const DeviceData<T> *im, DeviceData<T> *output,
        int imageWidth, int imageHeight,
        int filterSize, int Din, int stride, int padding) {

    // each row is a flattened window of a single conv window over each input im
    // 行：对每一个输入图像的某个卷积窗口展开。
    int xThreads = Din; 

    // each column is all the conv vews for a single input im
    // 列：单个图像的所有卷积窗口。
    int widthKernels = ((imageWidth - filterSize + (2*padding))/stride)+1;
    int heightKernels = ((imageHeight - filterSize + (2*padding))/stride)+1;
    int yThreads = widthKernels * heightKernels;

    // each thread flattens/pads one convolution window 
    output->resize(filterSize * filterSize * xThreads * yThreads);
    
    dim3 threadsPerBlock(xThreads, yThreads);
    dim3 blocksPerGrid(1, 1);

    // 一个块放不下就都放几个块。
    if (xThreads > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.x = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.x = ceil(double(xThreads)/double(threadsPerBlock.x));
    }
    
    if (yThreads > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.y = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.y = ceil(double(yThreads)/double(threadsPerBlock.y));
    }

    kernel::im2row<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&im->begin()[0]),
        thrust::raw_pointer_cast(&output->begin()[0]),
        imageWidth, imageHeight, filterSize, Din, stride, padding
    );

    cudaThreadSynchronize();
}

template<typename T, typename U>
void maxpool_im2row(const DeviceData<T> *im, DeviceData<T> *output,
        int imageHeight, int imageWidth,
        int filterSize, int Din, int batchSize, int stride, int padding, U pad_val) {

    // each thread row calculates all windows over a particular channel for a particular image
    // 行：特定图像特定通道的所有窗口。
    int yThreads = batchSize * Din;

    // each thread column calculates a window in the same position for all images/channels
    // 列：对于所有图片和通道，单个位置的窗口。
    int widthKernels = ((imageWidth - filterSize + (2*padding))/stride)+1;
    int heightKernels = ((imageHeight - filterSize + (2*padding))/stride)+1;
    int xThreads = widthKernels * heightKernels;

    // each individual thread flattens/pads one pool window
    // 每个线程处理单个图片单个通道单个窗口的展开。
    DeviceData<T> unpaddedOutput(filterSize * filterSize * xThreads * yThreads);
    
    dim3 threadsPerBlock(xThreads, yThreads);
    dim3 blocksPerGrid(1, 1);

    if (xThreads > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.x = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.x = ceil(double(xThreads)/double(threadsPerBlock.x));
    }
    
    if (yThreads > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.y = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.y = ceil(double(yThreads)/double(threadsPerBlock.y));
    }

    kernel::maxpool_im2row<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&im->begin()[0]),
        thrust::raw_pointer_cast(&unpaddedOutput.begin()[0]),
        imageHeight, imageWidth, filterSize, Din, batchSize, stride, padding
    );

    // pad output pools to next power of 2
    // 填充到2次幂。
    int paddedSize = pow(ceil(log2(filterSize * filterSize)), 2);
    output->resize(paddedSize * xThreads * yThreads);

    if (paddedSize == filterSize * filterSize) {
        thrust::copy(unpaddedOutput.begin(), unpaddedOutput.end(), output->begin());
    } else {
        kernel::stride_pad<<<blocksPerGrid,threadsPerBlock>>>(
            thrust::raw_pointer_cast(&unpaddedOutput.begin()[0]),
            thrust::raw_pointer_cast(&output->begin()[0]),
            batchSize * Din, widthKernels * heightKernels,
            filterSize * filterSize, paddedSize - (filterSize * filterSize),
            pad_val
        );
    }

    cudaThreadSynchronize();
}

template<typename T, typename U>
void stride_pad(const DeviceData<T> *input, DeviceData<T> *output,
        int stride, int padding, U pad_val) {

    int yThreads = input->size() / stride;
    int xThreads = 1;

    dim3 threadsPerBlock(xThreads, yThreads);
    dim3 blocksPerGrid(1, 1);

    if (xThreads > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.x = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.x = ceil(double(xThreads)/double(threadsPerBlock.x));
    }
    
    if (yThreads > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.y = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.y = ceil(double(yThreads)/double(threadsPerBlock.y));
    }

    kernel::stride_pad<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&input->begin()[0]),
        thrust::raw_pointer_cast(&output->begin()[0]),
        input->size() / stride,
        1,
        stride,
        padding,
        pad_val
    );

    cudaThreadSynchronize();
}

template<typename T>
void averagepool_im2row(const DeviceData<T> *im, DeviceData<T> *output,
        int imageWidth, int imageHeight,
        int filterSize, int Din, int batchSize, int stride, int padding) {
   
    // each thread row calculates all windows over a particular channel for a particular image
    int yThreads = batchSize * Din;

    // each thread column calculates a window in the same position for all images/channels
    int widthKernels = ((imageWidth - filterSize + (2*padding))/stride)+1;
    int heightKernels = ((imageHeight - filterSize + (2*padding))/stride)+1;
    int xThreads = widthKernels * heightKernels;

    dim3 threadsPerBlock(xThreads, yThreads);
    dim3 blocksPerGrid(1, 1);

    if (xThreads > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.x = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.x = ceil(double(xThreads)/double(threadsPerBlock.x));
    }
    
    if (yThreads > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.y = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.y = ceil(double(yThreads)/double(threadsPerBlock.y));
    }

    output->resize(batchSize * heightKernels * widthKernels * Din * filterSize * filterSize);

    kernel::maxpool_im2row<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&im->begin()[0]),
        thrust::raw_pointer_cast(&output->begin()[0]),
        imageHeight, imageWidth, filterSize, Din, batchSize, stride, padding
    );

    cudaThreadSynchronize();
}

template<typename T, typename I>
void maxpool_row2im(DeviceData<T, I> *rows, DeviceData<T, I> *output,
        int imageHeight, int imageWidth,
        int filterSize, int Din, int batchSize, int stride) {

    // each col is an element of a flattened pool window
    int yThreads = filterSize * filterSize; 

    // each row is a conv view over the input images
    int widthKernels = ((imageWidth - filterSize)/stride)+1;
    int heightKernels = ((imageHeight - filterSize)/stride)+1;
    int xThreads = batchSize * widthKernels * heightKernels * Din;

    // each thread adds one convolution window value into the output
    dim3 threadsPerBlock(xThreads, yThreads);
    dim3 blocksPerGrid(1, 1);

    if (xThreads > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.x = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.x = ceil(double(xThreads)/double(threadsPerBlock.x));
    }
    
    if (yThreads > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.y = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.y = ceil(double(yThreads)/double(threadsPerBlock.y));
    }

    kernel::maxpool_row2im<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&rows->begin()[0]),
        thrust::raw_pointer_cast(&output->begin()[0]),
        imageHeight, imageWidth, filterSize, Din, batchSize, stride
    );

    cudaThreadSynchronize();
}

template<typename T, typename I>
void averagepool_expand_delta(const DeviceData<T, I> *input_delta, DeviceData<T, I> *expanded_delta,
        int Din, int poolSize) {

    int xThreads = input_delta->size();
    int yThreads = 1;

    // each thread expands one delta value into the output
    dim3 threadsPerBlock(xThreads, yThreads);
    dim3 blocksPerGrid(1, 1);

    if (xThreads > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.x = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.x = ceil(double(xThreads)/double(threadsPerBlock.x));
    }
    
    if (yThreads > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.y = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.y = ceil(double(yThreads)/double(threadsPerBlock.y));
    }

    kernel::averagepool_expand_delta<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&input_delta->begin()[0]),
        thrust::raw_pointer_cast(&expanded_delta->begin()[0]),
        input_delta->size(), Din, poolSize
    );

    cudaThreadSynchronize();
}

template<typename T, typename I>
void expandFilter(DeviceData<T, I> &filter, DeviceData<T, I> &out,
        int rows, int cols, int stride) {

    int xThreads = cols + (cols - 1)*stride; 
    int yThreads = rows + (rows - 1)*stride;

    //output.resize(yThreads * xThreads);
    
    dim3 threadsPerBlock(xThreads, yThreads);
    dim3 blocksPerGrid(1, 1);

    if (xThreads > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.x = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.x = ceil(double(xThreads)/double(threadsPerBlock.x));
    }
    
    if (yThreads > MAX_THREADS_PER_BLOCK) {
        threadsPerBlock.y = MAX_THREADS_PER_BLOCK;
        blocksPerGrid.y = ceil(double(yThreads)/double(threadsPerBlock.y));
    }

    kernel::expandFilter<<<blocksPerGrid,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&filter.begin()[0]),
        thrust::raw_pointer_cast(&out.begin()[0]),
        rows, cols, stride
    );

    cudaThreadSynchronize();
}

}

