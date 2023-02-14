#include "unitTests.h"

template<typename T>
struct GForceTest : public testing::Test {
    using ParamType = T;
};

bool use_offline = true;

TYPED_TEST_CASE(GForceTest, GFO<uint32_t>);

TYPED_TEST(GForceTest, Mult) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share a ({12, 24, 3, 5, -2, -3}, false); 
    Share b ({1, 0, 11, 3, -1, 11}, false);

    DeviceData<T> result(a.size());

    if (use_offline){
        b.offline_known = true;
    }
    a *= b;
    reconstruct(a, result);

    std::vector<double> expected = {12, 0, 33, 15, 2, -33};
    assertDeviceData(result, expected, false);
}

TYPED_TEST(GForceTest, MatMul) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share a = {1, 1, 1, 1, 1, 1};  // 2 x 3
    Share b = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}; // 3 x 4
    Share c(8); // 2 x 4

    DeviceData<T> result(8);

    if (use_offline){
        b.offline_known = true;
    }
    matmul(a, b, c, 2, 4, 3, false, false, false, (T)FLOAT_PRECISION);
    reconstruct(c, result);

    std::vector<double> expected = {1, 1, 1, 1, 0, 0, 1, 1};

    assertDeviceData(result, expected);
}

TYPED_TEST(GForceTest, MatMul2) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share a = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};  // 4 x 4
    Share b = {-2.786461, -1.280988, -2.209210, 0.049379, 0.241369, 1.617007, -0.572261, 0.705014, -1.176370, -0.814461, 0.992866, 0.856274};
    Share c(12); // 4 x 3

    DeviceData<T> result(12);

    if (use_offline){
        b.offline_known = true;
    }
    matmul(a, b, c, 4, 3, 4, true, true, true, (T)FLOAT_PRECISION);
    reconstruct(c, result);

    std::vector<double> expected = {-2.786461, -1.280988, -2.209210, 0.049379, 0.241369, 1.617007, -0.572261, 0.705014, -1.176370, -0.814461, 0.992866, 0.856274};
    assertDeviceData(result, expected);
}

TYPED_TEST(GForceTest, DRELU) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share input = {
        -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1,
        -1, 1
    };

    //Change Share to TPC<uint8_t>
    Share result(input.size());
    dReLU(input, result);

    std::vector<double> expected = {
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 1
    };

    //Change to <uint8_t>
    DeviceData<uint32_t> super_result(result.size());
    reconstruct(result, super_result);
    assertDeviceData(super_result, expected, false);
}

TYPED_TEST(GForceTest, DRELU2) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share input = {
        0.326922, 0.156987, 0.461417, -0.221444, 39.846086, 0.000000, 0.000000, 0.000000,
        0.326922, 0.156987, 0.461417, -0.221444, 39.846086, 0.000000, 0.000000, 0.000000,
        0.326922, 0.156987, 0.461417, -0.221444, 39.846086, 0.000000, 0.000000, 0.000000
    };

    //Change Share to TPC<uint8_t>
    Share result(input.size());
    dReLU(input, result);

    std::vector<double> expected = {
        1, 1, 1, 0, 1, 1, 1, 1,
        1, 1, 1, 0, 1, 1, 1, 1,
        1, 1, 1, 0, 1, 1, 1, 1
    };

    //Change to <uint8_t>
    DeviceData<uint32_t> super_result(result.size());
    reconstruct(result, super_result);

    printDeviceData(super_result, "actual", false);
    assertDeviceData(super_result, expected, false);
}

TYPED_TEST(GForceTest, DRELU3) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    Share input = {
        0.326922, 0.156987, 0.461417, -0.221444, 39.846086, 0.000000, 0.000000, 0.000000
    };

    T negative = (T)(-10 * (1 << FLOAT_PRECISION));
    DeviceData<T> add = {
        0, 0, 0, 0, 0, negative, negative, negative
    };
    for(int share = 0; share < Share::numShares(); share++) {
        *input.getShare(share) += add;
    }

    Share result(input.size());
    dReLU(input, result);

    std::vector<double> expected = {
        1, 1, 1, 0, 1, 0, 0, 0
    };

    assertShare(result, expected, false);
}

TYPED_TEST(GForceTest, Truncate) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;
    // true
    printf("is GFO share?\t%d\n", typeid(Share)==typeid(GFO<T>));
    // false
    printf("is TPC share?\t%d\n", typeid(Share)==typeid(TPC<T>));

    Share a = {1 << 3, 2 << 3, 3 << 3, -3 << 3};
    dividePublic(a, (T)1 << 3);

    DeviceData<T> result(a.size());
    std::vector<double> expected = {1, 2, 3, -3};
    reconstruct(a, result);

    assertDeviceData(result, expected);
}

TYPED_TEST(GForceTest, Convolution) {

    using Share = typename TestFixture::ParamType;
    using T = typename Share::share_type;

    if (partyNum >= Share::numParties) return;

    size_t batchSize = 2;
    size_t imageWidth = 3, imageHeight = 3;
    size_t filterSize = 3;
    size_t Din = 1, Dout = 1;
    size_t stride = 1, padding = 1;

    // N=2, H=3, W=3, C=1
    Share im = {
        1, 2, 1, 2, 3, 2, 1, 2, 1,
        1, 2, 1, 2, 3, 2, 1, 2, 1
    };

    // N(Dout)=1, H=3, W=3, C(Din)=1
    Share filters = {
        1, 0, 1, 0, 1, 0, 1, 0, 1
    };

    size_t wKernels = (imageWidth - filterSize + (2*padding)) / stride + 1;
    size_t hKernels = (imageHeight - filterSize + (2*padding)) / stride + 1;
    Share out(batchSize * wKernels * hKernels * Dout);

    filters.offline_known = true;
    convolution(im, filters, out,
        cutlass::conv::Operator::kFprop,
        batchSize, imageHeight, imageWidth, filterSize,
        Din, Dout, stride, padding, FLOAT_PRECISION);

    std::vector<double> expected = {
        4, 6, 4, 6, 7, 6, 4, 6, 4,
        4, 6, 4, 6, 7, 6, 4, 6, 4,
    };
    DeviceData<T> result(out.size());
    reconstruct(out, result);
    assertDeviceData(result, expected);
}

// TYPED_TEST(GForceTest, LocalConvolution) {

//     using Share = typename TestFixture::ParamType;
//     using T = typename Share::share_type;

//     if (partyNum >= Share::numParties) return;

//     size_t imageWidth = 2, imageHeight = 2;
//     size_t filterSize = 3;
//     size_t Din = 2, Dout = 2;
//     size_t stride = 1, padding = 1;

//     // 2x2, Din=2
//     Share im = {
//         1, 2,
//         1, 2,

//         3, 0,
//         1, 1
//     };

//     // 3x3, Din=2, Dout=2
//     Share filters = {
//         1, 2, 1,
//         0, 1, 0,
//         1, 2, 2,

//         0, 1, 0,
//         0, 0, 1,
//         2, 1, 1,


//         1, 1, 1,
//         0, 0, 0,
//         1, 1, 1,

//         1, 0, 1,
//         1, 0, 1,
//         1, 0, 1
//     };

//     // imageW - filterSize + (2*padding) / stride + 1
//     size_t wKernels = 2;
//     // imageH - filterSize + (2*padding) / stride + 1
//     size_t hKernels = 2;
//     DeviceData<T> out(wKernels * hKernels * 2); // Dout = 2

//     filters.offline_known = true;
//     localConvolution(im, filters, out, imageWidth, imageHeight, filterSize, Din, Dout, stride, padding);

//     Share rss_out(out.size());
//     *rss_out.getShare(0) += out;
//     dividePublic(rss_out, 1 << FLOAT_PRECISION);

//     DeviceData<T> result(out.size());
//     reconstruct(rss_out, result);
//     std::vector<double> expected = {
//         9, 10,
//         9, 7,

//         4, 7,    
//         4, 7
//     };

//     assertDeviceData(result, expected);
// }