#pragma once

#include "Precompute.h"

template<typename T, typename Share>
void Precompute::getRandomNumber(Share &r) {

	comm_profiler.pause();
	func_profiler.pause();
	test_profiler.pause();

	T* rr = new T[r.size()];
	// align AES step.
	aes_objects[1-partyNum]->getRandom(rr, r.size());
	aes_objects[partyNum]->getRandom(rr, r.size());
	thrust::copy(rr, rr + r.size(), r.getShare(0)->begin());
	r *= 1;

	delete [] rr;
	synchronize(1, 2);

	comm_profiler.start();
	func_profiler.start();
	test_profiler.start();
}

template<typename T>
void Precompute::getRandomNumber(DeviceData<T> &r) {

	comm_profiler.pause();
	func_profiler.pause();
	test_profiler.pause();

	T* rr = new T[r.size()];
	// align AES step.
	aes_objects[1-partyNum]->getRandom(rr, r.size());
	aes_objects[partyNum]->getRandom(rr, r.size());
	thrust::copy(rr, rr + r.size(), r.begin());

	delete [] rr;
	synchronize(1, 2);

	comm_profiler.start();
	func_profiler.start();
	test_profiler.start();
}

template<typename T, typename Share>
void Precompute::getCoin(Share &r) {

	comm_profiler.pause();
	func_profiler.pause();
	test_profiler.pause();

	T* rr = new T[r.size()];
	// align AES step.
	aes_objects[1-partyNum]->getRandom(rr, r.size());
	aes_objects[partyNum]->getRandom(rr, r.size());
	thrust::copy(rr, rr + r.size(), r.getShare(0)->begin());
	r &= 1;

	delete [] rr;
	synchronize(1, 2);

	comm_profiler.start();
	func_profiler.start();
	test_profiler.start();
}

template<typename T>
void Precompute::getCoin(DeviceData<T> &r) {

	comm_profiler.pause();
	func_profiler.pause();
	test_profiler.pause();

	T* rr = new T[r.size()];
	// align AES step.
	aes_objects[1-partyNum]->getRandom(rr, r.size());
	aes_objects[partyNum]->getRandom(rr, r.size());
	thrust::copy(rr, rr + r.size(), r.begin());
	r &= 1;

	delete [] rr;
	synchronize(1, 2);

	comm_profiler.start();
	func_profiler.start();
	test_profiler.start();
}

template<typename T, typename Share>
void Precompute::getBeaverTriples(Share &x, Share &y, Share &z) {

	comm_profiler.pause();
	func_profiler.pause();
	test_profiler.pause();

	if (!ENABLE_OFFLINE_RANDOMNESS) {
		x.fill(1);
		y.fill(1);
		z.fill(1);
	} 
	else {
		T* rx = new T[x.size()];
		T* ry = new T[y.size()];
		T* rz = new T[z.size()];
		aes_objects[partyNum]->getRandom(rx, x.size());
		aes_objects[partyNum]->getRandom(ry, y.size());
		aes_objects[partyNum]->getRandom(rz, z.size());
		thrust::copy(rx, rx + x.size(), x.getShare(0)->begin());
		thrust::copy(ry, ry + y.size(), y.getShare(0)->begin());
		thrust::copy(rz, rz + z.size(), z.getShare(0)->begin());
		x *= 1;
		y *= 1;
		z *= 1;

		// thrust::copy(x.getShare(0)->begin(), x.getShare(0)->end(), test);
		// std::cout << "----------- my x --------------" << std::endl;
		// for (T t: test) {
		// 	std::cout << t << " ";
		// }
		// std::cout << std::endl;
		// thrust::copy(y.getShare(0)->begin(), y.getShare(0)->end(), test);
		// std::cout << "----------- my y --------------" << std::endl;
		// for (T t: test) {
		// 	std::cout << t << " ";
		// }
		// std::cout << std::endl;
		// thrust::copy(z.getShare(0)->begin(), z.getShare(0)->end(), test);
		// std::cout << "----------- my z --------------" << std::endl;
		// for (T t: test) {
		// 	std::cout << t << " ";
		// }
		// std::cout << std::endl;

		aes_objects[1-partyNum]->getRandom(rx, x.size());
		aes_objects[1-partyNum]->getRandom(ry, y.size());
		aes_objects[1-partyNum]->getRandom(rz, z.size());

		// TODO: HE and communication.
		// 0: Server.
		if (partyNum == 0){
			Share vx(x.size()), vy(y.size()), vz(z.size());
			thrust::copy(rx, rx + x.size(), vx.getShare(0)->begin());
			thrust::copy(ry, ry + y.size(), vy.getShare(0)->begin());
			thrust::copy(rz, rz + z.size(), vz.getShare(0)->begin());
			vx *= 1;
			vy *= 1;
			vz *= 1;

			// thrust::copy(vx.getShare(0)->begin(), vx.getShare(0)->end(), test);
			// std::cout << "----------- another x --------------" << std::endl;
			// for (T t: test) {
			// 	std::cout << t << " ";
			// }
			// std::cout << std::endl;
			// thrust::copy(vy.getShare(0)->begin(), vy.getShare(0)->end(), test);
			// std::cout << "----------- another y --------------" << std::endl;
			// for (T t: test) {
			// 	std::cout << t << " ";
			// }
			// std::cout << std::endl;
			// thrust::copy(vz.getShare(0)->begin(), vz.getShare(0)->end(), test);
			// std::cout << "----------- another z --------------" << std::endl;
			// for (T t: test) {
			// 	std::cout << t << " ";
			// }
			// std::cout << std::endl;

			vx += x;
			vy += y;
			vx *= *vy.getShare(0);
			z.zero();
			z += vx;
			z -= vz;

			// thrust::copy(z.getShare(0)->begin(), z.getShare(0)->end(), test);
			// std::cout << "----------- my new z --------------" << std::endl;
			// for (T t: test) {
			// 	std::cout << t << " ";
			// }
			// std::cout << std::endl;

		}
		synchronize(1, 2);

		delete [] rx;
		delete [] ry;
		delete [] rz;
	}

	comm_profiler.start();
	func_profiler.start();
	test_profiler.start();
}

// TODO: There is a error in TPC's CarryOut protocol. Fix it.
template<typename T, typename Share>
void Precompute::getBooleanBeaverTriples(Share &x, Share &y, Share &z) {

	comm_profiler.pause();
	func_profiler.pause();
	test_profiler.pause();

	// if (!ENABLE_OFFLINE_RANDOMNESS) {
		x.fill(1);
		y.fill(1);
		z.fill(1);
	// } 
	// else {
	// 	T rx[x.size()], ry[y.size()], rz[z.size()];
	// 	aes_objects[partyNum]->getRandom(rx, x.size());
	// 	aes_objects[partyNum]->getRandom(ry, y.size());
	// 	aes_objects[partyNum]->getRandom(rz, z.size());
	// 	thrust::copy(rx, rx + x.size(), x.getShare(0)->begin());
	// 	thrust::copy(ry, ry + y.size(), y.getShare(0)->begin());
	// 	thrust::copy(rz, rz + z.size(), z.getShare(0)->begin());
	// 	x &= uint64_t(1);
	// 	y &= uint64_t(1);
	// 	z &= uint64_t(1);

	// 	aes_objects[1-partyNum]->getRandom(rx, x.size());
	// 	aes_objects[1-partyNum]->getRandom(ry, y.size());
	// 	aes_objects[1-partyNum]->getRandom(rz, z.size());

	// 	// TODO: HE and communication.
	// 	// 0: Server.
	// 	if (partyNum == 0){
	// 		Share vx(x.size()), vy(y.size()), vz(z.size());
	// 		thrust::copy(rx, rx + x.size(), vx.getShare(0)->begin());
	// 		thrust::copy(ry, ry + y.size(), vy.getShare(0)->begin());
	// 		thrust::copy(rz, rz + z.size(), vz.getShare(0)->begin());
	// 		vx &= uint64_t(1);
	// 		vy &= uint64_t(1);
	// 		vz &= uint64_t(1);
	// 		vx ^= x;
	// 		vy ^= y;
	// 		vx &= *vy.getShare(0);
	// 		z.zero();
	// 		z ^= vx;
	// 		z ^= vz;
	// 	}
	// }
	synchronize(1, 2);

	comm_profiler.start();
	func_profiler.start();
	test_profiler.start();
}

template<typename T, typename Share>
void Precompute::getMatrixBeaverTriple(Share &x, Share &y, Share &z,
	int a_rows, int a_cols, int b_rows, int b_cols,
	bool transpose_a, bool transpose_b, bool transpose_c) 
{
	comm_profiler.pause();
	func_profiler.pause();
	test_profiler.pause();

	int rows = transpose_a ? a_cols : a_rows;

	int shared = transpose_a ? a_rows : a_cols;
	assert(shared == (transpose_b ? b_cols : b_rows));

	int cols = transpose_b ? b_rows : b_cols;

	if (!ENABLE_OFFLINE_RANDOMNESS) {
		x.fill(1);
		y.fill(1);
		z.fill(shared);
	}
	else {
		T* rx = new T[x.size()];
		T* ry = new T[y.size()];
		T* rz = new T[z.size()];
		// generate my randomness.
		aes_objects[partyNum]->getRandom(rx, x.size());
		aes_objects[partyNum]->getRandom(ry, y.size());
		aes_objects[partyNum]->getRandom(rz, z.size());
		thrust::copy(rx, rx + x.size(), x.getShare(0)->begin());
		thrust::copy(ry, ry + y.size(), y.getShare(0)->begin());
		thrust::copy(rz, rz + z.size(), z.getShare(0)->begin());
		// auto modular for GForce.
		x *= 1;
		y *= 1;
		z *= 1;

		// generate the other party's randomness.
		aes_objects[1-partyNum]->getRandom(rx, x.size());
		aes_objects[1-partyNum]->getRandom(ry, y.size());
		aes_objects[1-partyNum]->getRandom(rz, z.size());

		// TODO: HE and communication.
		// 0: Server.
		if (partyNum == 0){
			Share vx(x.size()), vy(y.size()), vz(z.size());
			thrust::copy(rx, rx + x.size(), vx.getShare(0)->begin());
			thrust::copy(ry, ry + y.size(), vy.getShare(0)->begin());
			thrust::copy(rz, rz + z.size(), vz.getShare(0)->begin());
			vx *= 1;
			vy *= 1;
			vz *= 1;
			gpu::gemm(rows, cols, shared, vx.getShare(0), transpose_a, y.getShare(0), transpose_b, z.getShare(0), transpose_c);
			z -= vz;
			gpu::gemm(rows, cols, shared, x.getShare(0), transpose_a, vy.getShare(0), transpose_b, vz.getShare(0), transpose_c);
			z += vz;
			gpu::gemm(rows, cols, shared, x.getShare(0), transpose_a, y.getShare(0), transpose_b, vz.getShare(0), transpose_c);
			z += vz;
			gpu::gemm(rows, cols, shared, vx.getShare(0), transpose_a, vy.getShare(0), transpose_b, vz.getShare(0), transpose_c);
			z += vz;
		}

		delete []rx;
		delete []ry;
		delete []rz;
	}
	synchronize(1, 2);

	comm_profiler.start();
	func_profiler.start();
	test_profiler.start();
}

template<typename T, typename Share>
void Precompute::getConvBeaverTriple_fprop(Share &x, Share &y, Share &z,
	int batchSize, int imageHeight, int imageWidth, int Din,
	int Dout, int filterHeight, int filterWidth,
	int paddingHeight, int paddingWidth,
	int stride, int dilation) {
	
	comm_profiler.pause();
	func_profiler.pause();
	test_profiler.pause();

	// int outputHeight = (imageHeight + 2 * padding - filterSize) / stride + 1; 
	// int outputWidth = (imageWidth + 2 * padding - filterSize) / stride + 1; 

	// assert(x.size() == imageWidth * imageHeight * Din * batchSize && "Incorrect x input for conv beaver triple");
	// assert(y.size() == filterSize * filterSize * Din * Dout && "Incorrect y input for conv beaver triple");
	// assert(z.size() == outputWidth * outputHeight * Dout * batchSize && "Incorrect z input for conv beaver triple");

	if (!ENABLE_OFFLINE_RANDOMNESS) {
		x.fill(0);
		y.fill(0);
		z.fill(0);
	}
	else {
		T* rx = new T[x.size()];
		T* ry = new T[y.size()];
		T* rz = new T[z.size()];

		// generate my randomness.
		aes_objects[partyNum]->getRandom(rx, x.size());
		aes_objects[partyNum]->getRandom(ry, y.size());
		aes_objects[partyNum]->getRandom(rz, z.size());
		thrust::copy(rx, rx + x.size(), x.getShare(0)->begin());
		thrust::copy(ry, ry + y.size(), y.getShare(0)->begin());
		thrust::copy(rz, rz + z.size(), z.getShare(0)->begin());
		// auto modular for GForce.
		x *= 1;
		y *= 1;
		z *= 1;

		// generate the other party's randomness.
		aes_objects[1-partyNum]->getRandom(rx, x.size());
		aes_objects[1-partyNum]->getRandom(ry, y.size());
		aes_objects[1-partyNum]->getRandom(rz, z.size());

		// TODO: HE and communication.
		// 0: Server.
		if (partyNum == 0){
			Share vx(x.size()), vy(y.size()), vz(z.size());
			thrust::copy(rx, rx + x.size(), vx.getShare(0)->begin());
			thrust::copy(ry, ry + y.size(), vy.getShare(0)->begin());
			thrust::copy(rz, rz + z.size(), vz.getShare(0)->begin());
			vx *= 1;
			vy *= 1;
			vz *= 1;
			gpu::conv_fprop(vx.getShare(0), y.getShare(0), z.getShare(0), 
				batchSize, imageHeight, imageWidth, Din,
				Dout, filterHeight, filterWidth,
				paddingHeight, paddingWidth,
				stride, dilation);
			z -= vz;
			gpu::conv_fprop(x.getShare(0), vy.getShare(0), vz.getShare(0), 
				batchSize, imageHeight, imageWidth, Din,
				Dout, filterHeight, filterWidth,
				paddingHeight, paddingWidth,
				stride, dilation);
			z += vz;
			gpu::conv_fprop(x.getShare(0), y.getShare(0), vz.getShare(0), 
				batchSize, imageHeight, imageWidth, Din,
				Dout, filterHeight, filterWidth,
				paddingHeight, paddingWidth,
				stride, dilation);
			z += vz;
			gpu::conv_fprop(vx.getShare(0), vy.getShare(0), vz.getShare(0), 
				batchSize, imageHeight, imageWidth, Din,
				Dout, filterHeight, filterWidth,
				paddingHeight, paddingWidth,
				stride, dilation);
			z += vz;
		}

		delete []rx;
		delete []ry;
		delete []rz;
	}
	synchronize(1, 2);

	comm_profiler.start();
	func_profiler.start();
	test_profiler.start();
}

template<typename T, typename Share>
void Precompute::getConvBeaverTriple_dgrad(Share &x, Share &y, Share &z,
	int batchSize, int outputHeight, int outputWidth, int Dout,
	int filterHeight, int filterWidth, int Din,
	int paddingHeight, int paddingWidth, int stride, int dilation,
	int imageHeight, int imageWidth) {
	
	comm_profiler.pause();
	func_profiler.pause();
	test_profiler.pause();

	// int outputHeight = (imageHeight + 2 * padding - filterSize) / stride + 1; 
	// int outputWidth = (imageWidth + 2 * padding - filterSize) / stride + 1; 

	// assert(x.size() == imageWidth * imageHeight * Din * batchSize && "Incorrect x input for conv beaver triple");
	// assert(y.size() == filterSize * filterSize * Din * Dout && "Incorrect y input for conv beaver triple");
	// assert(z.size() == outputWidth * outputHeight * Dout * batchSize && "Incorrect z input for conv beaver triple");

	if (!ENABLE_OFFLINE_RANDOMNESS) {
		x.fill(0);
		y.fill(0);
		z.fill(0);
	}
	else {
		T* rx = new T[x.size()];
		T* ry = new T[y.size()];
		T* rz = new T[z.size()];

		// generate my randomness.
		aes_objects[partyNum]->getRandom(rx, x.size());
		aes_objects[partyNum]->getRandom(ry, y.size());
		aes_objects[partyNum]->getRandom(rz, z.size());
		thrust::copy(rx, rx + x.size(), x.getShare(0)->begin());
		thrust::copy(ry, ry + y.size(), y.getShare(0)->begin());
		thrust::copy(rz, rz + z.size(), z.getShare(0)->begin());
		// auto modular for GForce.
		x *= 1;
		y *= 1;
		z *= 1;

		// generate the other party's randomness.
		aes_objects[1-partyNum]->getRandom(rx, x.size());
		aes_objects[1-partyNum]->getRandom(ry, y.size());
		aes_objects[1-partyNum]->getRandom(rz, z.size());

		// TODO: HE and communication.
		// 0: Server.
		if (partyNum == 0){
			Share vx(x.size()), vy(y.size()), vz(z.size());
			thrust::copy(rx, rx + x.size(), vx.getShare(0)->begin());
			thrust::copy(ry, ry + y.size(), vy.getShare(0)->begin());
			thrust::copy(rz, rz + z.size(), vz.getShare(0)->begin());
			vx *= 1;
			vy *= 1;
			vz *= 1;
			gpu::conv_dgrad(vx.getShare(0), y.getShare(0), z.getShare(0),
				batchSize, outputHeight, outputWidth, Dout,
				filterHeight, filterWidth, Din,
				paddingHeight, paddingWidth, stride, dilation,
				imageHeight, imageWidth);
			z -= vz;
			gpu::conv_dgrad(x.getShare(0), vy.getShare(0), vz.getShare(0),
				batchSize, outputHeight, outputWidth, Dout,
				filterHeight, filterWidth, Din,
				paddingHeight, paddingWidth, stride, dilation,
				imageHeight, imageWidth);
			z += vz;
			gpu::conv_dgrad(x.getShare(0), y.getShare(0), vz.getShare(0), 
				batchSize, outputHeight, outputWidth, Dout,
				filterHeight, filterWidth, Din,
				paddingHeight, paddingWidth, stride, dilation,
				imageHeight, imageWidth);
			z += vz;
			gpu::conv_dgrad(vx.getShare(0), vy.getShare(0), vz.getShare(0), 
				batchSize, outputHeight, outputWidth, Dout,
				filterHeight, filterWidth, Din,
				paddingHeight, paddingWidth, stride, dilation,
				imageHeight, imageWidth);
			z += vz;
		}

		delete []rx;
		delete []ry;
		delete []rz;
	}
	synchronize(1, 2);

	comm_profiler.start();
	func_profiler.start();
	test_profiler.start();
}

template<typename T, typename Share>
void Precompute::getConvBeaverTriple_wgrad(Share &x, Share &y, Share &z,
	int batchSize, int outputHeight, int outputWidth, int Dout,
	int imageHeight, int imageWidth, int Din,
	int filterHeight, int filterWidth,
	int paddingHeight, int paddingWidth, int stride, int dilation) {

	comm_profiler.pause();
	func_profiler.pause();
	test_profiler.pause();

	// int outputHeight = (imageHeight + 2 * padding - filterSize) / stride + 1; 
	// int outputWidth = (imageWidth + 2 * padding - filterSize) / stride + 1; 

	// assert(x.size() == imageWidth * imageHeight * Din * batchSize && "Incorrect x input for conv beaver triple");
	// assert(y.size() == filterSize * filterSize * Din * Dout && "Incorrect y input for conv beaver triple");
	// assert(z.size() == outputWidth * outputHeight * Dout * batchSize && "Incorrect z input for conv beaver triple");

	if (!ENABLE_OFFLINE_RANDOMNESS) {
		x.fill(0);
		y.fill(0);
		z.fill(0);
	}
	else {
		T* rx = new T[x.size()];
		T* ry = new T[y.size()];
		T* rz = new T[z.size()];

		// generate my randomness.
		aes_objects[partyNum]->getRandom(rx, x.size());
		aes_objects[partyNum]->getRandom(ry, y.size());
		aes_objects[partyNum]->getRandom(rz, z.size());
		thrust::copy(rx, rx + x.size(), x.getShare(0)->begin());
		thrust::copy(ry, ry + y.size(), y.getShare(0)->begin());
		thrust::copy(rz, rz + z.size(), z.getShare(0)->begin());
		// auto modular for GForce.
		x *= 1;
		y *= 1;
		z *= 1;

		// generate the other party's randomness.
		aes_objects[1-partyNum]->getRandom(rx, x.size());
		aes_objects[1-partyNum]->getRandom(ry, y.size());
		aes_objects[1-partyNum]->getRandom(rz, z.size());

		// TODO: HE and communication.
		// 0: Server.
		if (partyNum == 0){
			Share vx(x.size()), vy(y.size()), vz(z.size());
			thrust::copy(rx, rx + x.size(), vx.getShare(0)->begin());
			thrust::copy(ry, ry + y.size(), vy.getShare(0)->begin());
			thrust::copy(rz, rz + z.size(), vz.getShare(0)->begin());
			vx *= 1;
			vy *= 1;
			vz *= 1;
			gpu::conv_wgrad(vx.getShare(0), y.getShare(0), z.getShare(0),
				batchSize, outputHeight, outputWidth, Dout,
				imageHeight, imageWidth, Din,
				filterHeight, filterWidth,
				paddingHeight, paddingWidth, stride, dilation);
			z -= vz;
			gpu::conv_wgrad(x.getShare(0), vy.getShare(0), vz.getShare(0),
				batchSize, outputHeight, outputWidth, Dout,
				imageHeight, imageWidth, Din,
				filterHeight, filterWidth,
				paddingHeight, paddingWidth, stride, dilation);
			z += vz;
			gpu::conv_wgrad(x.getShare(0), y.getShare(0), vz.getShare(0),
				batchSize, outputHeight, outputWidth, Dout,
				imageHeight, imageWidth, Din,
				filterHeight, filterWidth,
				paddingHeight, paddingWidth, stride, dilation);
			z += vz;
			gpu::conv_wgrad(vx.getShare(0), vy.getShare(0), vz.getShare(0),
				batchSize, outputHeight, outputWidth, Dout,
				imageHeight, imageWidth, Din,
				filterHeight, filterWidth,
				paddingHeight, paddingWidth, stride, dilation);
			z += vz;
		}

		delete []rx;
		delete []ry;
		delete []rz;
	}
	synchronize(1, 2);

	comm_profiler.start();
	func_profiler.start();
	test_profiler.start();
}      

// 	Delphi's linear layer offline phase protocol.
// 	output:
//		Server: out1 = rs, out2 = 0.
//		Client: out1 = w*rc-rs, out2 = rc.
template<typename T, typename ShareBase, typename Share>
void Precompute::getCorrelatedRandomness(
	const ShareBase& w, Share& out1, Share& out2
) {
	comm_profiler.pause();
	func_profiler.pause();
	test_profiler.pause();

	T* myr = new T[w.size()];
	T* otherr = new T[w.size()];	
	aes_objects[partyNum]->getRandom(myr, w.size());
	aes_objects[1-partyNum]->getRandom(otherr, w.size());
	thrust::copy(otherr, otherr + w.size(), out1.getShare(0)->begin());
	thrust::copy(myr, myr + w.size(), out2.getShare(0)->begin());
	out1 *= 1;
	out2 *= 1;

	// Server. out2 = myr = rs, out1 = otherr = rc.
	if (partyNum == 0) {
		out1 *= *w.getShare(0);
		out1 -= out2;
		out1.getShare(0)->transmit(1);
		out1.zero();
		out1 += out2;
		out1.getShare(0)->join();
	}
	// Client. out2 = myr = rc, out1 = otherr = rs.
	else if (partyNum == 1) {
		out1.getShare(0)->receive(0);
		out1.getShare(0)->join();
	}

	delete [] myr;
	delete [] otherr;
	synchronize(1, 2);

	comm_profiler.start();
	func_profiler.start();
	test_profiler.start();
}

// 	Delphi's linear layer offline phase protocol for MatMul.
// 	output:
//		Server: out1 = Rs, out2 = 0.
//		Client: out1 = Rc*W-Rs, out2 = Rc.
template<typename T, typename Share>
void Precompute::getCorrelatedRandomness_matmul(
	const Share& w, Share& out1, Share& out2,
	int a_rows, int a_cols, int b_rows, int b_cols,
	bool transpose_a, bool transpose_b, bool transpose_c
) {
	comm_profiler.pause();
	func_profiler.pause();
	test_profiler.pause();

	int rows = transpose_a ? a_cols : a_rows;

	int shared = transpose_a ? a_rows : a_cols;
	assert(shared == (transpose_b ? b_cols : b_rows));

	int cols = transpose_b ? b_rows : b_cols;

	// random1: Rs. random2: Rc.
	T* random1 = new T[out1.size()];
	T* random2 = new T[out2.size()];

	aes_objects[0]->getRandom(random1, out1.size());
	aes_objects[1]->getRandom(random2, out2.size());
	thrust::copy(random1, random1 + out1.size(), out1.getShare(0)->begin());
	thrust::copy(random2, random2 + out2.size(), out2.getShare(0)->begin());
	out1 *= 1;
	out2 *= 1;

	// Server. out1 = Rs, out2 = 0.
	if (partyNum == 0) {
		DeviceData<T> temp(out1.size());
		gpu::gemm(rows, cols, shared, out1.getShare(0), transpose_a, w.getShare(0), transpose_b, &temp, transpose_c);
		temp -= *out2.getShare(0);
		temp.transmit(1);
		temp.join();
	}
	// Client. out2 = myr = rc, out1 = otherr = rs.
	else if (partyNum == 1) {
		out1.getShare(0)->receive(0);
		out1.getShare(0)->join();
	}

	delete [] random1;
	delete [] random2;
	synchronize(1, 2);

	comm_profiler.start();
	func_profiler.start();
	test_profiler.start();
}

// 	Delphi's linear layer offline phase protocol for Fprop.
// 	output:
//		Server: out1 = Rs, out2 = 0.
//		Client: out1 = Rc*W-Rs, out2 = Rc.
template<typename T, typename Share>
void Precompute::getCorrelatedRandomness_fprop(
	const Share& w, Share& out1, Share& out2,
	int batchSize, int imageHeight, int imageWidth, int Din,
	int Dout, int filterHeight, int filterWidth,
	int paddingHeight, int paddingWidth,
	int stride, int dilation
) {
	comm_profiler.pause();
	func_profiler.pause();
	test_profiler.pause();

	// random1: Rs. random2: Rc.
	T* random1 = new T[out1.size()];
	T* random2 = new T[out2.size()];

	aes_objects[0]->getRandom(random1, out1.size());
	aes_objects[1]->getRandom(random2, out2.size());
	thrust::copy(random1, random1 + out1.size(), out1.getShare(0)->begin());
	thrust::copy(random2, random2 + out2.size(), out2.getShare(0)->begin());
	out1 *= 1;
	out2 *= 1;

	// Server. out1 = Rs, out2 = 0.
	if (partyNum == 0) {
		DeviceData<T> temp(out1.size());
		gpu::conv_fprop(out1.getShare(0), w.getShare(0), &temp, 
			batchSize, imageHeight, imageWidth, Din,
			Dout, filterHeight, filterWidth,
			paddingHeight, paddingWidth,
			stride, dilation);
		temp -= *out2.getShare(0);
		temp.transmit(1);
		temp.join();
	}
	// Client. out2 = myr = rc, out1 = otherr = rs.
	else if (partyNum == 1) {
		out1.getShare(0)->receive(0);
		out1.getShare(0)->join();
	}

	delete [] random1;
	delete [] random2;
	synchronize(1, 2);

	comm_profiler.start();
	func_profiler.start();
	test_profiler.start();
}

// 	Our linear layer offline phase protocol.
//	input:
//		Server: in = w.
//		Client: in = [x]^C.
// 	output:
//		Server: out = s_Z = [x]^C*w - r^C.
//		Client: out = [z]^C = r^C.
template<typename T, typename ShareBase1, typename ShareBase2, typename Share>
void Precompute::getCorrelatedPairs(
	const ShareBase1& in, ShareBase2& out
) {
	comm_profiler.pause();
	func_profiler.pause();
	test_profiler.pause();

	// Server. out = s_Z.
	if (partyNum == 0) {
		// [x]^C.
		Share xc(in.size());
		xc.getShare(0)->receive(1);

		// r^C.
		T otherr[in.size()];
		aes_objects[1-partyNum]->getRandom(otherr, in.size());
		thrust::copy(otherr, otherr + in.size(), out.getShare(0)->begin());

		xc.getShare(0)->join();
		
		xc *= *in.getShare(0);
		out *= static_cast<T>(-1);
		out += xc; 
	}
	// Client. out = [z]^C.
	else if (partyNum == 1) {
		Share xc(in.size());
		xc.zero();
		xc += in;

		xc.getShare(0)->transmit(0);
		xc.getShare(0)->join();

		T myr[in.size()];	
		aes_objects[partyNum]->getRandom(myr, in.size());
		thrust::copy(myr, myr + in.size(), out.getShare(0)->begin());
	}
	synchronize(1, 2);

	comm_profiler.start();
	func_profiler.start();
	test_profiler.start();
}

// 	Our linear layer offline phase protocol.
//	input:
//		Server: in = W.
//		Client: in = [X]^C.
// 	output:
//		Server: out = s_Z = [X]^C*W - R^C.
//		Client: out = [Z]^C = R^C.
template<typename T, typename Share>
void Precompute::getCorrelatedPairs_matmul(
	const Share& in, Share& out,
	int a_rows, int a_cols, int b_rows, int b_cols,
	bool transpose_a, bool transpose_b, bool transpose_c
) {
	comm_profiler.pause();
	func_profiler.pause();
	test_profiler.pause();

	int rows = transpose_a ? a_cols : a_rows;

	int shared = transpose_a ? a_rows : a_cols;
	assert(shared == (transpose_b ? b_cols : b_rows));

	int cols = transpose_b ? b_rows : b_cols;

	// // Server. out = s_Z.
	if (partyNum == 0) {
		// [X]^C.
		Share Xc(a_rows * a_cols), temp(out.size());
		Xc.getShare(0)->receive(1);

		// R^C.
		T* otherr = new T[out.size()];
		aes_objects[1-partyNum]->getRandom(otherr, out.size());
		thrust::copy(otherr, otherr + out.size(), out.getShare(0)->begin());

		Xc.getShare(0)->join();

		gpu::gemm(rows, cols, shared, Xc.getShare(0), transpose_a, in.getShare(0), transpose_b, temp.getShare(0), transpose_c);
		out *= static_cast<T>(-1);
		out += temp;

		delete [] otherr;
	}
	// Client. out = [z]^C.
	else if (partyNum == 1) {
		Share Xc(in.size());
		Xc.zero();
		Xc += in;

		Xc.getShare(0)->transmit(0);
		Xc.getShare(0)->join();

		T* myr = new T[out.size()];	
		aes_objects[partyNum]->getRandom(myr, out.size());
		thrust::copy(myr, myr + out.size(), out.getShare(0)->begin());
		delete [] myr;
	}
	synchronize(1, 2);

	comm_profiler.start();
	func_profiler.start();
	test_profiler.start();
}

// 	Our linear layer offline phase protocol for Fprop.
//	input:
//		Server: in = W.
//		Client: in = [X]^C.
// 	output:
//		Server: out = s_Z = [X]^C*W - R^C.
//		Client: out = [Z]^C = R^C.
template<typename T, typename Share>
void Precompute::getCorrelatedPairs_fprop(
	const Share& in, Share& out,
	int batchSize, int imageHeight, int imageWidth, int Din,
	int Dout, int filterHeight, int filterWidth,
	int paddingHeight, int paddingWidth,
	int stride, int dilation
) {
	comm_profiler.pause();
	func_profiler.pause();
	test_profiler.pause();

	// Server. out = s_Z.
	if (partyNum == 0) {
		// [X]^C.
		Share Xc(batchSize*imageHeight*imageWidth*Din), temp(out.size());
		Xc.getShare(0)->receive(1);

		// R^C.
		T* otherr = new T[out.size()];	
		aes_objects[1-partyNum]->getRandom(otherr, out.size());
		thrust::copy(otherr, otherr + out.size(), out.getShare(0)->begin());

		Xc.getShare(0)->join();

		gpu::conv_fprop(Xc.getShare(0), in.getShare(0), temp.getShare(0), 
			batchSize, imageHeight, imageWidth, Din,
			Dout, filterHeight, filterWidth,
			paddingHeight, paddingWidth,
			stride, dilation);
		out *= static_cast<T>(-1);
		out += temp;

		delete [] otherr;
	}
	// Client. out = [z]^C.
	else if (partyNum == 1) {
		Share Xc(in.size());
		Xc.zero();
		Xc += in;

		Xc.getShare(0)->transmit(0);
		Xc.getShare(0)->join();

		T* myr = new T[out.size()];	
		aes_objects[partyNum]->getRandom(myr, out.size());
		thrust::copy(myr, myr + out.size(), out.getShare(0)->begin());
		
		delete [] myr;
	}
	synchronize(1, 2);

	comm_profiler.start();
	func_profiler.start();
	test_profiler.start();
}

// Our reshareC protocol's offline phase.
// input:
//		Client: [x]^C.
// output:
//		Server: [d_x]^S.
//		Client: [d_x]^C.
template<typename T, typename Share, typename Share2>
void Precompute::reshareC_off(
	const Share& in, Share2& out
) {
	comm_profiler.pause();
	func_profiler.pause();
	test_profiler.pause();

	// Server. out = s_Z.
	if (partyNum == 0) {
		T* myr = new T[out.size()];	

		aes_objects[partyNum]->getRandom(myr, out.size());
		thrust::copy(myr, myr + out.size(), out.getShare(1)->begin());

		delete [] myr;
	}
	else if (partyNum == 1) {
		T* otherr = new T[out.size()];	

		aes_objects[1-partyNum]->getRandom(otherr, out.size());

		out.zero();
		*out.getShare(1) -= *in.getShare(0);

		delete [] otherr;
	}
	synchronize(1, 2);

	comm_profiler.start();
	func_profiler.start();
	test_profiler.start();
}

// Our FusionMux protocol's offline phase.
// input:
//		Server: [d_x]^S, [b]_2^S.
//		Client: [d_x]^C.
// output:
//		Server: [d_bd_x]^S, [d_b]^S, [d_b]_2^S, 0.
//		Client: [d_bd_x]^C, [d_b]^C, [d_b]_2^C, [z]^C.
template<typename T, typename Share, typename Share2, typename Share3, typename Share4>
void Precompute::FusionMux_off(
	const Share& x, const Share2& tb,
	Share2& dbdx, Share2& db, Share3& bang, Share4& z
) {
	comm_profiler.pause();
	func_profiler.pause();
	test_profiler.pause();

	size_t size = z.size();
	// Server.
	if (partyNum == 0) {

		bang.zero();
		*bang.getShare(1) += *tb.getShare(0);

		// bit2A: db.
		T* anotherr = new T[size];	
		aes_objects[1-partyNum]->getRandom(anotherr, size);
		thrust::copy(anotherr, anotherr + size, db.getShare(0)->begin());
		*db.getShare(0) &= 1;
		db ^= *bang.getShare(1);
		aes_objects[1-partyNum]->getRandom(anotherr, size);
		thrust::copy(anotherr, anotherr + size, z.getShare(0)->begin());

		// Mux.
		dbdx.getShare(0)->receive(1);
		dbdx.getShare(0)->join();
		dbdx += *x.getShare(1);
		dbdx *= *db.getShare(0);
		aes_objects[1-partyNum]->getRandom(anotherr, size);
		thrust::copy(anotherr, anotherr + size, bang.getShare(0)->begin());
		dbdx -= *bang.getShare(0);

		// bit2A.
		db -= z;

		aes_objects[1-partyNum]->getRandom(anotherr, size);
		z.zero();

		delete [] anotherr;
	}
	// Client.
	else if (partyNum == 1) {
		T* myr = new T[size];	

		aes_objects[partyNum]->getRandom(myr, size);
		thrust::copy(myr, myr + size, bang.getShare(1)->begin());
		*bang.getShare(1) &= 1;

		// bit2A.
		aes_objects[partyNum]->getRandom(myr, size);
		thrust::copy(myr, myr + size, db.getShare(0)->begin());

		// Mux.
		z.zero();
		*z.getShare(0) += *x.getShare(1);
		z.getShare(0)->transmit(0);
		aes_objects[partyNum]->getRandom(myr, size);
		thrust::copy(myr, myr + size, dbdx.getShare(0)->begin());
		z.getShare(0)->join();

		aes_objects[partyNum]->getRandom(myr, size);
		thrust::copy(myr, myr + size, z.getShare(0)->begin());

		delete [] myr;
	}
	synchronize(1, 2);

	comm_profiler.start();
	func_profiler.start();
	test_profiler.start();
}