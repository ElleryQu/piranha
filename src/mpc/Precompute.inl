#pragma once

#include "Precompute.h"

template<typename T, typename Share>
void Precompute::getRandomNumber(Share &r) {
	T rr[r.size()];
	// align AES step.
	aes_objects[1-partyNum]->getRandom(rr, r.size());
	aes_objects[partyNum]->getRandom(rr, r.size());
	thrust::copy(rr, rr + r.size(), r.getShare(0)->begin());
	r *= 1;
}

template<typename T>
void Precompute::getRandomNumber(DeviceData<T> &r) {
	T rr[r.size()];
	// align AES step.
	aes_objects[1-partyNum]->getRandom(rr, r.size());
	aes_objects[partyNum]->getRandom(rr, r.size());
	thrust::copy(rr, rr + r.size(), r.begin());
}

template<typename T, typename Share>
void Precompute::getCoin(Share &r) {
	T rr[r.size()];
	// align AES step.
	aes_objects[1-partyNum]->getRandom(rr, r.size());
	aes_objects[partyNum]->getRandom(rr, r.size());
	thrust::copy(rr, rr + r.size(), r.getShare(0)->begin());
	r &= 1;
}

template<typename T>
void Precompute::getCoin(DeviceData<T> &r) {
	T rr[r.size()];
	// align AES step.
	aes_objects[1-partyNum]->getRandom(rr, r.size());
	aes_objects[partyNum]->getRandom(rr, r.size());
	thrust::copy(rr, rr + r.size(), r.begin());
	r &= 1;
}

template<typename T, typename Share>
void Precompute::getBeaverTriples(Share &x, Share &y, Share &z) {

	// T test[z.size()];

	if (!ENABLE_OFFLINE_RANDOMNESS) {
		x.fill(1);
		y.fill(1);
		z.fill(1);
	} 
	else {
		T rx[x.size()], ry[y.size()], rz[z.size()];
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
	}
}

// TODO: There is a error in TPC's CarryOut protocol. Fix it.
template<typename T, typename Share>
void Precompute::getBooleanBeaverTriples(Share &x, Share &y, Share &z) {

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
}

template<typename T, typename Share>
void Precompute::getMatrixBeaverTriple(Share &x, Share &y, Share &z,
	int a_rows, int a_cols, int b_rows, int b_cols,
	bool transpose_a, bool transpose_b, bool transpose_c) 
{
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
		T rx[x.size()], ry[y.size()], rz[z.size()];
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
	}
}

template<typename T, typename Share>
void Precompute::getConvBeaverTriple_fprop(Share &x, Share &y, Share &z,
	int batchSize, int imageHeight, int imageWidth, int Din,
	int Dout, int filterHeight, int filterWidth,
	int paddingHeight, int paddingWidth,
	int stride, int dilation) {

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
		T rx[x.size()], ry[y.size()], rz[z.size()];
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
	}
}

template<typename T, typename Share>
void Precompute::getConvBeaverTriple_dgrad(Share &x, Share &y, Share &z,
	int batchSize, int outputHeight, int outputWidth, int Dout,
	int filterHeight, int filterWidth, int Din,
	int paddingHeight, int paddingWidth, int stride, int dilation,
	int imageHeight, int imageWidth) {

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
		T rx[x.size()], ry[y.size()], rz[z.size()];
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
	}
}

template<typename T, typename Share>
void Precompute::getConvBeaverTriple_wgrad(Share &x, Share &y, Share &z,
	int batchSize, int outputHeight, int outputWidth, int Dout,
	int imageHeight, int imageWidth, int Din,
	int filterHeight, int filterWidth,
	int paddingHeight, int paddingWidth, int stride, int dilation) {

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
		T rx[x.size()], ry[y.size()], rz[z.size()];
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
	}
}      

// 	Delphi's linear layer offline phase protocol.
// 	output:
//		Server: out1 = rs, out2 = 0.
//		Client: out1 = w*rc-rs, out2 = rc.
template<typename T, typename ShareBase, typename Share>
void Precompute::getCorrelatedRandomness(
	const ShareBase& w, Share& out1, Share& out2
) {

	T myr[w.size()], otherr[w.size()];	
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
	int rows = transpose_a ? a_cols : a_rows;

	int shared = transpose_a ? a_rows : a_cols;
	assert(shared == (transpose_b ? b_cols : b_rows));

	int cols = transpose_b ? b_rows : b_cols;

	// random1: Rs. random2: Rc.
	T random1[out1.size()], random2[out2.size()];	
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
}
