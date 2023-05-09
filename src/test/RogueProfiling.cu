// /**
//  * Rogue benchmark.
// */

// #include "unitTests.h" 

// extern Profiler comm_profiler;
// extern Profiler func_profiler;
// extern Profiler test_profiler;

// #define TEST_USLEEP_TIME 0
// #define OFFLINE_KNOWN true

// template<typename T>
// struct RogueTest: public testing::Test {
//     using ParamType = T;
// };

// using Types = testing::Types<TPC<uint64_t>, GFO<uint64_t>, ROG<uint64_t> >;

// TYPED_TEST_CASE(RogueTest, Types);

// // TYPED_TEST(RogueTest, Mult_Profiling_BENCHMARK) {

// //     using Share = typename TestFixture::ParamType;
// //     using T = typename Share::share_type;

// //     if (partyNum >= 2) return;

// //     func_profiler.clear();
// //     comm_profiler.clear();
// //     test_profiler.clear();

// //     std::vector<double> rnd_vals;

// //     std::vector<int> Np = {10, 10000, 1 << 17};
// //     for (int j = 0; j < Np.size(); j++){
// //         std::vector<int> N(EXP_TIMES + 1, Np[j]);
// //         N[0] = 1;
// //         for (int i = 0; i < N.size(); i++) {

// //             int n = N[i];

// //             random_vector(rnd_vals, n);
// //             Share a(n);
// //             a.setPublic(rnd_vals);

// //             random_vector(rnd_vals, n);
// //             Share b(n);
// //             b.setPublic(rnd_vals);
// //             b.offline_known = OFFLINE_KNOWN;

// //             test_profiler.start();
// //             a *= b;
// //             test_profiler.accumulate("mult_bm");

// //             if (i == 0) { // sacrifice run to spin up GPU
// //                 func_profiler.clear();
// //                 comm_profiler.clear();
// //                 test_profiler.clear();
// //                 continue;
// //             }
// //         }
// //         printf("mult_bm (N=%d) - %f sec.\n", Np[j], test_profiler.get_elapsed("mult_bm") / 1000.0 / EXP_TIMES);
// //         writeProfile(
// //             pf, Share::getProt(), "mult_bm", Np[j], 1, 
// //             test_profiler.get_elapsed("mult_bm") / 1000.0 / EXP_TIMES,
// //             comm_profiler.get_rounds() / EXP_TIMES,
// //             comm_profiler.get_comm_tx_bytes() / 1024. / 1024. / EXP_TIMES,
// //             comm_profiler.get_comm_rx_bytes() / 1024. / 1024. / EXP_TIMES,
// //             comm_profiler.get_elapsed_all() / 1000.0 / EXP_TIMES
// //         );

// //         func_profiler.clear();
// //         comm_profiler.clear();
// //         test_profiler.clear();
// //     }
// // }

// TYPED_TEST(RogueTest, MatMul_Profiling_BENCHMARK) {


//     using Share = typename TestFixture::ParamType;
//     using T = typename Share::share_type;

//     if (partyNum >= 2) return;

//     func_profiler.clear();
//     comm_profiler.clear();
//     test_profiler.clear();

//     std::vector<double> rnd_vals;

//     std::vector<int> Np = {1, 10, 100, 500, 1000};
//     std::stringstream os;
//     os << "output/profiling/matmul_profiling_party" << partyNum << ".tsv";
//     std::string path = os.str();
//     std::ofstream mlpf = std::ofstream(path, ios::app);
    
//     for (int j = 0; j < Np.size(); j++){
//         std::vector<int> N(EXP_TIMES + 1, Np[j]);
//         N[0] = 1;
//         double curr_elap = 0;
//         mlpf << Share::getProt() << " " << Np[j] << "\t";
//         for (int i = 0; i < N.size(); i++) {

//             int n = N[i];

//             random_vector(rnd_vals, n * n);
//             Share a(n*n);
//             a.setPublic(rnd_vals);

//             random_vector(rnd_vals, n * n);
//             Share b(n*n);
//             b.setPublic(rnd_vals);
//             b.offline_known = OFFLINE_KNOWN;

//             Share c(n*n);

//             test_profiler.start();

//             matmul(a, b, c, n, n, n, false, false, false, (uint64_t)FLOAT_PRECISION);

//             test_profiler.accumulate("matmul_bm");

//             mlpf << test_profiler.get_elapsed("matmul_bm") - curr_elap << "\t";
//             curr_elap = test_profiler.get_elapsed("matmul_bm");

//             if (i == 0) { // sacrifice run to spin up GPU
//                 func_profiler.clear();
//                 comm_profiler.clear();
//                 test_profiler.clear();

//                 continue;
//             }
//         }
//         mlpf << std::endl;

//         printf("matmul_bm (N=%d) - %f sec.\n", Np[j], test_profiler.get_elapsed("matmul_bm") / 1000.0 / EXP_TIMES);
//         writeProfile(
//             pf, Share::getProt(), "matmul_bm", 1, Np[j], 
//             test_profiler.get_elapsed("matmul_bm") / 1000.0 / EXP_TIMES,
//             comm_profiler.get_rounds() / EXP_TIMES,
//             comm_profiler.get_comm_tx_bytes() / 1024. / 1024. / EXP_TIMES,
//             comm_profiler.get_comm_rx_bytes() / 1024. / 1024. / EXP_TIMES,
//             comm_profiler.get_elapsed_all() / 1000.0 / EXP_TIMES
//         );

//         func_profiler.clear();
//         comm_profiler.clear();
//         test_profiler.clear();
//     }
// }

// // TYPED_TEST(RogueTest, Conv_Profiling) {

// //     using Share = typename TestFixture::ParamType;
// //     using T = typename Share::share_type;

// //     if (partyNum >= 2) return;
// //     comm_profiler.clear();

// //     std::vector<double> rnd_vals;

// //     std::vector<std::tuple<int, int, int, int, int> > dims = {
// //         std::make_tuple(28, 1, 16, 5, 24),
// //         std::make_tuple(28, 1, 16, 5, 24),
// //         std::make_tuple(12, 20, 50, 3, 10),
// //         std::make_tuple(32, 3, 50, 7, 24),
// //         std::make_tuple(64, 3, 32, 5, 60),
// //         std::make_tuple(224, 3, 64, 5, 220)
// //     };
// //     for (int i = 0; i < dims.size(); i++) {

// //         auto dim = dims[i];

// //         int im_size = std::get<0>(dim);
// //         int din = std::get<1>(dim);
// //         int dout = std::get<2>(dim);
// //         int f_size = std::get<3>(dim);
// //         int out_size = std::get<4>(dim);

// //         int N = 1;

// //         int a_size = N * din * im_size * im_size;
// //         random_vector(rnd_vals, a_size);
// //         Share a(a_size);
// //         a.setPublic(rnd_vals);

// //         int b_size = din * dout * f_size * f_size;
// //         random_vector(rnd_vals, b_size);
// //         Share b(b_size);
// //         b.setPublic(rnd_vals);
// //         b.offline_known = OFFLINE_KNOWN;

// //         Share c(N * dout * out_size * out_size);

// //         test_profiler test_profiler;
// //         test_profiler.start();

// //         convolution(a, b, c, cutlass::conv::Operator::kFprop, N, im_size, im_size, f_size, din, dout, 1, 0, FLOAT_PRECISION);

// //         test_profiler.accumulate("conv");

// //         if (i == 0) continue; // sacrifice run to spin up GPU
// //         ostringstream os;
// //         os << din << ' ' << im_size << ' ' << dout << ' ' << f_size << ' ' << "CHimNfHf";
// //         printf("conv (N=1, Iw/h=%d, Din=%d, Dout=%d, f=%d) - %f sec.\n", im_size, din, dout, f_size, test_profiler.get_elapsed("conv") / 1000.0);
// //         writeProfile(
// //             pf, Share::getProt(), "conv", N, os.str(), 
// //             test_profiler.get_elapsed("conv") / 1000.0,
// //             comm_profiler.get_rounds() / EXP_TIMES,
// //             comm_profiler.get_comm_tx_bytes()/1024./1024.,
// //             comm_profiler.get_comm_rx_bytes()/1024./1024.,
// //             comm_profiler.get_elapsed("conv") / 1000.0
// //             );
// //         usleep(TEST_USLEEP_TIME);
// //     }
// // }

// // TYPED_TEST(RogueTest, ReLU_Profiling_BENCHMARK) {

// //     using Share = typename TestFixture::ParamType;
// //     using T = typename Share::share_type;

// //     if (partyNum >= 2) return;
    
// //     func_profiler.clear();
// //     comm_profiler.clear();
// //     test_profiler.clear();

// //     std::vector<double> rnd_vals;

// //     std::vector<int> Np = {10, 1000, 10000};
// //     for (int j = 0; j < Np.size(); j++){
// //         std::vector<int> N(EXP_TIMES + 1, Np[j]);
// //         N[0] = 1;
// //         for (int i = 0; i < N.size(); i++) {

// //             int n = N[i];

// //             random_vector(rnd_vals, n);
// //             Share a(n);
// //             a.setPublic(rnd_vals);
// //             // std::cout << a.offline_known << std::endl;
// //             // a.offline_known=false;

// //             Share c(n);
// //             Share dc(n);

// //             test_profiler.start();

// //             ReLU(a, c, dc);

// //             test_profiler.accumulate("relu_bm");

// //             if (i == 0) { // sacrifice run to spin up GPU
// //                 func_profiler.clear();
// //                 comm_profiler.clear();
// //                 test_profiler.clear();

// //                 continue;
// //             }
// //         }
// //         printf("relu_bm (N=%d) - %f sec.\n", Np[j], test_profiler.get_elapsed("relu_bm") / 1000.0 / EXP_TIMES);
// //         writeProfile(
// //             pf, Share::getProt(), "relu_bm", Np[j], std::to_string(1), 
// //             test_profiler.get_elapsed("relu_bm") / 1000.0 / EXP_TIMES,
// //             comm_profiler.get_rounds() / EXP_TIMES,
// //             comm_profiler.get_comm_tx_bytes() / 1024. / 1024. / EXP_TIMES,
// //             comm_profiler.get_comm_rx_bytes() / 1024. / 1024. / EXP_TIMES,
// //             comm_profiler.get_elapsed_all() / 1000.0 / EXP_TIMES
// //         );

// //         func_profiler.clear();
// //         comm_profiler.clear();
// //         test_profiler.clear();
// //     }
// // }

// TYPED_TEST(RogueTest, MatMul_Profiling_BENCHMARK_vs_meteor) {


//     using Share = typename TestFixture::ParamType;
//     using T = typename Share::share_type;
//     using triple = typename std::tuple<int, int, int>;

//     if (partyNum >= 2) return;

//     func_profiler.clear();
//     comm_profiler.clear();
//     test_profiler.clear();

//     std::vector<double> rnd_vals;

//     std::vector<triple> Np = {triple(784, 128, 10), triple(128, 500, 100)};
//     std::stringstream os;
//     os << "output/profiling/matmul_profiling_party" << partyNum << ".tsv";
//     std::string path = os.str();
//     std::ofstream mlpf = std::ofstream(path, ios::app);
    
//     for (int j = 0; j < Np.size(); j++){
//         double curr_elap = 0;
//         int M = std::get<0>(Np[j]), K = std::get<1>(Np[j]), N = std::get<2>(Np[j]);
//         mlpf << Share::getProt() << " " << M << "\t";
//         for (int i = 0; i < EXP_TIMES + 1; i++) {

//             random_vector(rnd_vals, M * K);
//             Share a(M * K);
//             a.setPublic(rnd_vals);

//             random_vector(rnd_vals, K * N);
//             Share b(K * N);
//             b.setPublic(rnd_vals);
//             b.offline_known = OFFLINE_KNOWN;

//             Share c(M * N);

//             test_profiler.start();

//             matmul(a, b, c, M, N, K, false, false, false, (uint64_t)FLOAT_PRECISION);

//             test_profiler.accumulate("matmul_bm");

//             if (i == 0) { // sacrifice run to spin up GPU
//                 func_profiler.clear();
//                 comm_profiler.clear();
//                 test_profiler.clear();

//                 continue;
//             } else {
//                 mlpf << test_profiler.get_elapsed("matmul_bm") - curr_elap << "\t";
//                 curr_elap = test_profiler.get_elapsed("matmul_bm");
//             }
//         }
//         mlpf << std::endl;

//         printf("matmul_bm (N=%d) - %f sec.\n", std::get<0>(Np[j]), test_profiler.get_elapsed("matmul_bm") / 1000.0 / EXP_TIMES);
//         writeProfile(
//             pf, Share::getProt(), "matmul_bm", 1, std::get<0>(Np[j]), 
//             test_profiler.get_elapsed("matmul_bm") / 1000.0 / EXP_TIMES,
//             comm_profiler.get_rounds() / EXP_TIMES,
//             comm_profiler.get_comm_tx_bytes() / 1024. / 1024. / EXP_TIMES,
//             comm_profiler.get_comm_rx_bytes() / 1024. / 1024. / EXP_TIMES,
//             comm_profiler.get_elapsed_all() / 1000.0 / EXP_TIMES
//         );

//         func_profiler.clear();
//         comm_profiler.clear();
//         test_profiler.clear();
//     }
// }