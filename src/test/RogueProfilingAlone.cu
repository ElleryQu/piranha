// /**
//  * Rogue benchmark.
// */

// #include "unitTests.h" 

// extern Profiler comm_profiler;
// extern Profiler func_profiler;
// extern Profiler test_profiler;

// #define TEST_USLEEP_TIME 0
// #define OFFLINE_KNOWN true
// #define EXP_TIMES 1

// struct RogueTest: public testing::Test{};

// TEST(RogueTest, ReLU_benchmark_TPC) {
//     if (partyNum >= 2) return;
    
//     func_profiler.clear();
//     comm_profiler.clear();
//     test_profiler.clear();

//     std::vector<double> rnd_vals;

//     std::vector<int> Np = {1, 10, 100, 1000, 10000};

//     std::stringstream os;
//     os << "output/profiling/relu_profiling_party" << partyNum << ".tsv";
//     std::string path = os.str();
//     std::ofstream mlpf = std::ofstream(path, ios::app);

//     for (int j = 0; j < Np.size(); j++){
//         std::vector<int> N(EXP_TIMES + 1, Np[j]);
//         N[0] = 1;
//         double curr_elap = 0;
//         mlpf << "TPC" << " " << Np[j] << "\t";
//         for (int i = 0; i < N.size(); i++) {

//             int n = N[i];

//             random_vector(rnd_vals, n);
//             TPC<uint64_t> a(n);
//             a.setPublic(rnd_vals);
//             // std::cout << a.offline_known << std::endl;
//             // a.offline_known=false;

//             TPC<uint64_t> c(n);
//             TPC<uint8_t> dc(n);

//             test_profiler.start();

//             ReLU(a, c, dc);

//             test_profiler.accumulate("relu_bm");

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
//         printf("relu_bm (N=%d) - %f sec.\n", Np[j], test_profiler.get_elapsed("relu_bm") / 1000.0 / EXP_TIMES);
//         writeProfile(
//             pf, "TPC", "relu_bm", Np[j], std::to_string(1), 
//             test_profiler.get_elapsed("relu_bm") / 1000.0 / EXP_TIMES,
//             comm_profiler.get_rounds() / EXP_TIMES,
//             comm_profiler.get_comm_tx_bytes() / 1024. / 1024. / EXP_TIMES,
//             comm_profiler.get_comm_rx_bytes() / 1024. / 1024. / EXP_TIMES,
//             comm_profiler.get_elapsed_all() / 1000.0 / EXP_TIMES
//         );
//         mlpf << std::endl;

//         func_profiler.clear();
//         comm_profiler.clear();
//         test_profiler.clear();
//     }
// }

// TEST(RogueTest, ReLU_benchmark_GFO) {
//     if (partyNum >= 2) return;
    
//     func_profiler.clear();
//     comm_profiler.clear();
//     test_profiler.clear();

//     std::vector<double> rnd_vals;

//     std::vector<int> Np = {1, 10, 100, 1000, 10000};

//     std::stringstream os;
//     os << "output/profiling/relu_profiling_party" << partyNum << ".tsv";
//     std::string path = os.str();
//     std::ofstream mlpf = std::ofstream(path, ios::app);

//     for (int j = 0; j < Np.size(); j++){
//         std::vector<int> N(EXP_TIMES + 1, Np[j]);
//         N[0] = 1;
//         double curr_elap = 0;
//         mlpf << "GFO" << " " << Np[j] << "\t";
//         for (int i = 0; i < N.size(); i++) {

//             int n = N[i];

//             random_vector(rnd_vals, n);
//             GFO<uint64_t> a(n);
//             a.setPublic(rnd_vals);
//             // std::cout << a.offline_known << std::endl;
//             // a.offline_known=false;

//             GFO<uint64_t> c(n);
//             GFO<uint64_t> dc(n);

//             test_profiler.start();

//             ReLU<uint64_t, uint32_t, BufferIterator<uint64_t>, BufferIterator<uint64_t>, BufferIterator<uint64_t>>(a, c, dc);

//             test_profiler.accumulate("relu_bm");

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
//         printf("relu_bm (N=%d) - %f sec.\n", Np[j], test_profiler.get_elapsed("relu_bm") / 1000.0 / EXP_TIMES);
//         writeProfile(
//             pf, "GFO", "relu_bm", Np[j], std::to_string(1), 
//             test_profiler.get_elapsed("relu_bm") / 1000.0 / EXP_TIMES,
//             comm_profiler.get_rounds() / EXP_TIMES,
//             comm_profiler.get_comm_tx_bytes() / 1024. / 1024. / EXP_TIMES,
//             comm_profiler.get_comm_rx_bytes() / 1024. / 1024. / EXP_TIMES,
//             comm_profiler.get_elapsed_all() / 1000.0 / EXP_TIMES
//         );
//         mlpf << std::endl;

//         func_profiler.clear();
//         comm_profiler.clear();
//         test_profiler.clear();
//     }
// }

// TEST(RogueTest, ReLU_benchmark_ROG) {
//     if (partyNum >= 2) return;
    
//     func_profiler.clear();
//     comm_profiler.clear();
//     test_profiler.clear();

//     std::vector<double> rnd_vals;

//     std::vector<int> Np = {1, 10, 100, 1000, 10000};

//     std::stringstream os;
//     os << "output/profiling/relu_profiling_party" << partyNum << ".tsv";
//     std::string path = os.str();
//     std::ofstream mlpf = std::ofstream(path, ios::app);

//     for (int j = 0; j < Np.size(); j++){
//         std::vector<int> N(EXP_TIMES + 1, Np[j]);
//         N[0] = 1;
//         double curr_elap = 0;
//         mlpf << "ROG" << " " << Np[j] << "\t";
//         for (int i = 0; i < N.size(); i++) {

//             int n = N[i];

//             random_vector(rnd_vals, n);
//             ROG<uint64_t> a(n);
//             a.setPublic(rnd_vals);
//             // std::cout << a.offline_known << std::endl;
//             // a.offline_known=false;

//             ROG<uint64_t> c(n);
//             ROG<uint32_t> dc(n);

//             test_profiler.start();

//             ReLU(a, c, dc);

//             test_profiler.accumulate("relu_bm");

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
//         printf("relu_bm (N=%d) - %f sec.\n", Np[j], test_profiler.get_elapsed("relu_bm") / 1000.0 / EXP_TIMES);
//         writeProfile(
//             pf, "ROG", "relu_bm", Np[j], std::to_string(1), 
//             test_profiler.get_elapsed("relu_bm") / 1000.0 / EXP_TIMES,
//             comm_profiler.get_rounds() / EXP_TIMES,
//             comm_profiler.get_comm_tx_bytes() / 1024. / 1024. / EXP_TIMES,
//             comm_profiler.get_comm_rx_bytes() / 1024. / 1024. / EXP_TIMES,
//             comm_profiler.get_elapsed_all() / 1000.0 / EXP_TIMES
//         );
//         mlpf << std::endl;

//         func_profiler.clear();
//         comm_profiler.clear();
//         test_profiler.clear();
//     }
// }

// // vs meteor & GForce.
// TEST(RogueTest, ReLU_BENCHMARK_gm_GFO) {

//     if (partyNum >= 2) return;
    
//     func_profiler.clear();
//     comm_profiler.clear();
//     test_profiler.clear();

//     std::vector<double> rnd_vals;

//     std::vector<int> Np = {1 << 17, 128*128, 576*20};
//     for (int j = 0; j < Np.size(); j++){
//         std::vector<int> N(EXP_TIMES + 1, Np[j]);
//         N[0] = 1;
//         for (int i = 0; i < N.size(); i++) {

//             int n = N[i];

//             random_vector(rnd_vals, n);
//             GFO<uint64_t> a(n);
//             a.setPublic(rnd_vals);
//             // std::cout << a.offline_known << std::endl;
//             // a.offline_known=false;

//             GFO<uint64_t> c(n);
//             GFO<uint64_t> dc(n);

//             test_profiler.start();

//             ReLU<uint64_t, uint32_t, BufferIterator<uint64_t>, BufferIterator<uint64_t>, BufferIterator<uint64_t>>(a, c, dc);

//             test_profiler.accumulate("relu_bm");

//             if (i == 0) { // sacrifice run to spin up GPU
//                 func_profiler.clear();
//                 comm_profiler.clear();
//                 test_profiler.clear();

//                 continue;
//             }
//         }
//         printf("relu_bm (N=%d) - %f sec.\n", Np[j], test_profiler.get_elapsed("relu_bm") / 1000.0 / EXP_TIMES);
//         writeProfile(
//             pf, "GFO", "relu_bm", Np[j], std::to_string(1), 
//             test_profiler.get_elapsed("relu_bm") / 1000.0 / EXP_TIMES,
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

// TEST(RogueTest, ReLU_BENCHMARK_gm_ROG) {

//     if (partyNum >= 2) return;
    
//     func_profiler.clear();
//     comm_profiler.clear();
//     test_profiler.clear();

//     std::vector<double> rnd_vals;

//     std::vector<int> Np = {1 << 17, 128*128, 576*20};
//     for (int j = 0; j < Np.size(); j++){
//         std::vector<int> N(EXP_TIMES + 1, Np[j]);
//         N[0] = 1;
//         for (int i = 0; i < N.size(); i++) {

//             int n = N[i];

//             random_vector(rnd_vals, n);
//             ROG<uint64_t> a(n);
//             a.setPublic(rnd_vals);
//             // std::cout << a.offline_known << std::endl;
//             // a.offline_known=false;

//             ROG<uint64_t> c(n);
//             ROG<uint32_t> dc(n);

//             test_profiler.start();

//             ReLU(a, c, dc);

//             test_profiler.accumulate("relu_bm");

//             if (i == 0) { // sacrifice run to spin up GPU
//                 func_profiler.clear();
//                 comm_profiler.clear();
//                 test_profiler.clear();

//                 continue;
//             }
//         }
//         printf("relu_bm (N=%d) - %f sec.\n", Np[j], test_profiler.get_elapsed("relu_bm") / 1000.0 / EXP_TIMES);
//         writeProfile(
//             pf, "ROG", "relu_bm", Np[j], std::to_string(1), 
//             test_profiler.get_elapsed("relu_bm") / 1000.0 / EXP_TIMES,
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