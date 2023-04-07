#pragma once

#include "Precompute.h"

template<typename T, typename Share>
void Precompute::getBeaverTriples(Share &x, Share &y, Share &z) {

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
		std::cout << "my x: " << rx[1] << std::endl;
		std::cout << "my y: " << ry[1] << std::endl;
		std::cout << "my z: " << rz[1] << std::endl;
		T tempx = rx[1], tempy = ry[1], tempz;

		aes_objects[1-partyNum]->getRandom(rx, x.size());
		aes_objects[1-partyNum]->getRandom(ry, y.size());
		aes_objects[1-partyNum]->getRandom(rz, z.size());
		std::cout << "your x: " << rx[1] << std::endl;
		std::cout << "your y: " << ry[1] << std::endl;
		std::cout << "your z: " << rz[1] << std::endl;

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
			tempz = rz[1];
			vx += x;
			vy += y;
			vx *= *vy.getShare(0);
			z.zero();
			z += vx;
			z -= vz;
		}
		std::vector<T> open_z(z.size());
		thrust::copy(z.getShare(0)->begin(), z.getShare(0)->end(), open_z.begin());
		std::cout << "your z: " << tempz << std::endl;
		std::cout << "my final z: " << open_z[1] << std::endl;
	}
}

template<typename T, typename Share>
void Precompute::getMatrixBeaverTriple(Share &x, Share &y, Share &z,
	int a_rows, int a_cols, int b_rows, int b_cols,
	bool transpose_a, bool transpose_b) 
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
		aes_objects[partyNum]->getRandom(rx, x.size());
		aes_objects[partyNum]->getRandom(ry, y.size());
		aes_objects[partyNum]->getRandom(rz, z.size());
		thrust::copy(rx, rx + x.size(), x.getShare(0)->begin());
		thrust::copy(ry, ry + y.size(), y.getShare(0)->begin());
		thrust::copy(rz, rz + z.size(), z.getShare(0)->begin());
		x *= 1;
		y *= 1;
		z *= 1;
		// std::cout << rx[1] << std::endl;
		// std::cout << ry[1] << std::endl;
		// std::cout << rz[1] << std::endl;
		T tempx = rx[1], tempy = ry[1];

		aes_objects[1-partyNum]->getRandom(rx, x.size());
		aes_objects[1-partyNum]->getRandom(ry, y.size());
		aes_objects[1-partyNum]->getRandom(rz, z.size());
		// std::cout << rx[1] << std::endl;
		// std::cout << ry[1] << std::endl;
		// std::cout << rz[1] << std::endl;
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
			gpu::gemm(rows, cols, shared, vx.getShare(0), transpose_a, y.getShare(0), transpose_b, z.getShare(0), 0);
			z -= vz;
			gpu::gemm(rows, cols, shared, x.getShare(0), transpose_a, vy.getShare(0), transpose_b, vz.getShare(0), 0);
			z += vz;
			gpu::gemm(rows, cols, shared, x.getShare(0), transpose_a, y.getShare(0), transpose_b, vz.getShare(0), 0);
			z += vz;
			gpu::gemm(rows, cols, shared, vx.getShare(0), transpose_a, vy.getShare(0), transpose_b, vz.getShare(0), 0);
			z += vz;
		}
		std::vector<double> open_z(z.size());
		copyToHost(z, open_z, false);
		// std::cout << open_z[1] << std::endl;
	}
}


// void Precompute::getRandomBitShares(RSSVectorSmallType &a, size_t size)
// {
// 	assert(a.size() == size && "size mismatch for getRandomBitShares");
// 	for(auto &it : a)
// 		it = std::make_pair(0,0);
// }


//m_0 is random shares of 0, m_1 is random shares of 1 in RSSMyType. 
//This function generates random bits c and corresponding RSSMyType values m_c
/*
void Precompute::getSelectorBitShares(RSSVectorSmallType &c, RSSVectorMyType &m_c, size_t size)
{
	assert(c.size() == size && "size mismatch for getSelectorBitShares");
	assert(m_c.size() == size && "size mismatch for getSelectorBitShares");
	for(auto &it : c)
		it = std::make_pair(0,0);

	for(auto &it : m_c)
		it = std::make_pair(0,0);
}
*/

//Shares of random r, shares of bits of that, and shares of wrap3 of that.
/*
void Precompute::getShareConvertObjects(RSSVectorMyType &r, RSSVectorSmallType &shares_r, 
										RSSVectorSmallType &alpha, size_t size)
{
	assert(shares_r.size() == size*BIT_SIZE && "getShareConvertObjects size mismatch");
	for(auto &it : r)
		it = std::make_pair(0,0);

	for(auto &it : shares_r)
		it = std::make_pair(0,0);

	for(auto &it : alpha)
		it = std::make_pair(0,0);
}
*/

//Triplet verification myType
/*
void Precompute::getTriplets(RSSVectorMyType &a, RSSVectorMyType &b, RSSVectorMyType &c, 
						size_t rows, size_t common_dim, size_t columns)
{
	assert(((a.size() == rows*common_dim) 
		and (b.size() == common_dim*columns) 
		and (c.size() == rows*columns)) && "getTriplet size mismatch");
	
	for(auto &it : a)
		it = std::make_pair(0,0);

	for(auto &it : b)
		it = std::make_pair(0,0);

	for(auto &it : c)
		it = std::make_pair(0,0);
}
*/

//Triplet verification myType
/*
void Precompute::getTriplets(RSSVectorMyType &a, RSSVectorMyType &b, RSSVectorMyType &c, size_t size)
{
	assert(((a.size() == size) and (b.size() == size) and (c.size() == size)) && "getTriplet size mismatch");
	
	for(auto &it : a)
		it = std::make_pair(0,0);

	for(auto &it : b)
		it = std::make_pair(0,0);

	for(auto &it : c)
		it = std::make_pair(0,0);
}
*/

//Triplet verification smallType
/*
void Precompute::getTriplets(RSSVectorSmallType &a, RSSVectorSmallType &b, RSSVectorSmallType &c, size_t size)
{
	assert(((a.size() == size) and (b.size() == size) and (c.size() == size)) && "getTriplet size mismatch");
	
	for(auto &it : a)
		it = std::make_pair(0,0);

	for(auto &it : b)
		it = std::make_pair(0,0);

	for(auto &it : c)
		it = std::make_pair(0,0);
}
*/
