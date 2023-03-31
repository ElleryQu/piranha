/*
 * ROG.inl
 */

#pragma once

#include "Rogue.h"

#include <bitset>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include "../gpu/bitwise.cuh"
#include "../gpu/convolution.cuh"
#include "../gpu/conv.cuh"
#include "../gpu/DeviceData.h"
#include "../gpu/functors.cuh"
#include "../gpu/matrix.cuh"
#include "../gpu/gemm.cuh"
#include "../gpu/StridedRange.cuh"
#include "../globals.h"
#include "Precompute.h"
#include "../util/functors.h"
#include "../util/Profiler.h"

extern Precompute PrecomputeObject;
extern Profiler comm_profiler;
extern Profiler func_profiler;


// Functors

struct ROG_convex_comb_functor {
    const int party;
    ROG_convex_comb_functor(int _party) : party(_party) {}
    
    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple t) {
        // b, c share, d share
        if (thrust::get<0>(t) == 1) {
            switch(party) {
                case ROG<uint64_t>::SERVER: // doesn't really matter what type ROG is templated at here
                    thrust::get<2>(t) = 1 - thrust::get<1>(t);
                    break;
                case ROG<uint64_t>::CLIENT:
                    thrust::get<2>(t) = -thrust::get<1>(t);
                    break;
            }
        } else {
            thrust::get<2>(t) = thrust::get<1>(t);
        }
    }
};

// Prototypes

template<typename T>
void localMatMul(const ROG<T> &a, const ROG<T> &b, const ROG<T> &c,
        int M, int N, int K,
        bool transpose_a, bool transpose_b);

template<typename T, typename I, typename I2, typename I3>
void localConvolution(ROG<T, I> &im, ROG<T, I2> &filters, DeviceData<T, I3> &out,
        size_t imageWidth, size_t imageHeight, size_t filterSize, size_t Din, size_t Dout,
        size_t stride, size_t padding);

template<typename T, typename I, typename I2>
void carryOut(ROG<T, I> &p, ROG<T, I> &g, int k, ROG<T, I2> &out);

template<typename T, typename I, typename I2>
void getPowers(ROG<T, I> &in, DeviceData<T, I2> &pow);

template<typename T, typename I, typename I2, typename Functor>
void taylorSeries(ROG<T, I> &in, ROG<T, I2> &out,
        double a0, double a1, double a2,
        Functor fn);


template<typename T, typename U, typename I, typename I2>
void convex_comb(ROG<T, I> &a, ROG<T, I> &c, DeviceData<U, I2> &b);

// ROG class implementation 

template<typename T, typename I>
ROGBase<T, I>::ROGBase(DeviceData<T, I> *a, bool offline_known) : 
                shareA(a), offline_known(offline_known) {}

template<typename T, typename I>
void ROGBase<T, I>::set(DeviceData<T, I> *a) {
    shareA = a;
}

template<typename T, typename I>
size_t ROGBase<T, I>::size() const {
    return shareA->size();
}

template<typename T, typename I>
void ROGBase<T, I>::zero() {
    shareA->zero();
};

template<typename T, typename I>
void ROGBase<T, I>::fill(T val) {
    shareA->fill(partyNum == SERVER ? val : 0);
}

/// @brief Given an array(double) in CPU, embedding it on GPU.
/// @tparam T 
/// @tparam I 
/// @param v 
template<typename T, typename I>
void ROGBase<T, I>::setPublic(std::vector<double> &v) {
    std::vector<T> shifted_vals;
    for (double f : v) {
        shifted_vals.push_back((T) (f * (1 << FLOAT_PRECISION)));
    }

    switch (partyNum) {
        case SERVER:
            thrust::copy(shifted_vals.begin(), shifted_vals.end(), shareA->begin());
            break;
        case CLIENT:
            shareA->zero();
            break;
    }
};

template<typename T, typename I>
DeviceData<T, I> *ROGBase<T, I>::getShare(int i) {
    switch (i) {
        case 0:
            return shareA;
        default:
            return nullptr;
    }
}

template<typename T, typename I>
const DeviceData<T, I> *ROGBase<T, I>::getShare(int i) const {
    switch (i) {
        case 0:
            return shareA;
        default:
            return nullptr;
    }
}

template<typename T, typename I>
const std::string& ROGBase<T, I>::getProt() {
    const static std::string prot = "ROG";
    return prot;
}

template<typename T, typename I>
ROGBase<T, I> &ROGBase<T, I>::operator+=(const T rhs) {
    if (partyNum == SERVER) {
        *shareA += rhs;
    }
    return *this;
}

template<typename T, typename I>
ROGBase<T, I> &ROGBase<T, I>::operator-=(const T rhs) {
    if (partyNum == SERVER) {
        *shareA -= rhs;
    }
    return *this;
}

template<typename T, typename I>
ROGBase<T, I> &ROGBase<T, I>::operator*=(const T rhs) {
    *shareA *= rhs;
    return *this;
}

template<typename T, typename I>
ROGBase<T, I> &ROGBase<T, I>::operator%=(const T rhs) {
    *shareA %= rhs;
    return *this;
}

template<typename T, typename I>
ROGBase<T, I> &ROGBase<T, I>::operator>>=(const T rhs) {
    *shareA >>= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
ROGBase<T, I> &ROGBase<T, I>::operator+=(const DeviceData<T, I2> &rhs) {
    if (partyNum == SERVER) {
        *shareA += rhs;
    }
    return *this;
}

template<typename T, typename I>
template<typename I2>
ROGBase<T, I> &ROGBase<T, I>::operator-=(const DeviceData<T, I2> &rhs) {
    if (partyNum == SERVER) {
        *shareA -= rhs;
    }
    return *this;
}

template<typename T, typename I>
template<typename I2>
ROGBase<T, I> &ROGBase<T, I>::operator*=(const DeviceData<T, I2> &rhs) {
    *shareA *= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
ROGBase<T, I> &ROGBase<T, I>::operator^=(const DeviceData<T, I2> &rhs) {
    if (partyNum == SERVER) {
        *shareA ^= rhs;
    }
    return *this;
}

template<typename T, typename I>
template<typename I2>
ROGBase<T, I> &ROGBase<T, I>::operator&=(const DeviceData<T, I2> &rhs) {
    *shareA &= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
ROGBase<T, I> &ROGBase<T, I>::operator>>=(const DeviceData<T, I2> &rhs) {
    *shareA >>= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
ROGBase<T, I> &ROGBase<T, I>::operator<<=(const DeviceData<T, I2> &rhs) {
    *shareA <<= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
ROGBase<T, I> &ROGBase<T, I>::operator+=(const ROGBase<T, I2> &rhs) {
    *shareA += *rhs.getShare(0);
    return *this;
}

template<typename T, typename I>
template<typename I2>
ROGBase<T, I> &ROGBase<T, I>::operator-=(const ROGBase<T, I2> &rhs) {
    *shareA -= *rhs.getShare(0);
    return *this;
}

template<typename T, typename I>
template<typename I2>
ROGBase<T, I> &ROGBase<T, I>::operator*=(const ROGBase<T, I2> &rhs) {

    size_t size = rhs.size();

    if (!rhs.offline_known)
    {
        // Precomputation.
        ROG<T> x(size), y(size), z(size);
        PrecomputeObject.getBeaverTriples<T, ROG<T> >(x, y, z);
        DeviceData<T> e(size), f(size), temp(size);

        *x.getShare(0) += *this->getShare(0); 
        *y.getShare(0) += *rhs.getShare(0);
        reconstruct(x, e); reconstruct(y, f);
        *x.getShare(0) -= *this->getShare(0);
        *y.getShare(0) -= *rhs.getShare(0);
        
        this->zero();
        *this += z;

        temp.zero();
        temp += f;
        temp -= *y.getShare(0);
        temp *= e;
        *this += temp;

        temp.zero();
        temp -= *x.getShare(0);
        temp *= f;
        *this += temp;
    } 
    else 
    {
        // printf("-----------------\nOffline branch entered.\n-----------------\n");
        // TODO: Precomputation.
        // SERVER: r^S.     
        // CLIENT: w*r^C-r^S.
        DeviceData<T> offline_output(size);
        offline_output.fill(0);

        // SERVER: x-r^C.   
        // CLIENT: r^C.
        DeviceData<T> r(size);
        r.fill(0);
        if (partyNum == CLIENT) {
            *this->getShare(0) -= r;
            comm_profiler.start();
            this->getShare(0)->transmit(ROG<T>::otherParty(partyNum));
            this->getShare(0)->join();
            comm_profiler.accumulate("comm-time");
            this->getShare(0)->zero();
            *this->getShare(0) += offline_output;
        }
        else if (partyNum == SERVER) {
            comm_profiler.start();
            r.receive(ROG<T>::otherParty(partyNum));
            r.join();
            comm_profiler.accumulate("comm-time");
            *this->getShare(0) += r;
            *this->getShare(0) *= *rhs.getShare(0);
            *this->getShare(0) += offline_output;
        }

        func_profiler.add_comm_round();
    }
 
    return *this;
}

template<typename T, typename I>
template<typename I2>
ROGBase<T, I> &ROGBase<T, I>::operator^=(const ROGBase<T, I2> &rhs) {
    *shareA ^= *rhs.getShare(0);
    return *this;
}

template<typename T, typename I>
template<typename I2>
ROGBase<T, I> &ROGBase<T, I>::operator&=(const ROGBase<T, I2> &rhs) {

    size_t size = rhs.size();
    ROG<T> x(size), y(size), z(size);
    PrecomputeObject.getBooleanBeaverTriples<T, ROG<T> >(x, y, z);
    DeviceData<T> e(size), f(size), temp(size);

    *x.getShare(0) ^= *this->getShare(0); 
    *y.getShare(0) ^= *rhs.getShare(0);
    reconstruct(x, e); reconstruct(y, f);
    *x.getShare(0) ^= *this->getShare(0);
    *y.getShare(0) ^= *rhs.getShare(0);
    
    this->zero();
    *this ^= z;

    temp.zero();
    temp ^= f;
    temp ^= *y.getShare(0);
    temp &= e;
    *this ^= temp;

    temp.zero();
    temp ^= *x.getShare(0);
    temp &= f;
    *this ^= temp;
 
    return *this;
}

//TO_BE_DONE
template<typename T, typename I>
int ROGBase<T, I>::otherParty(int party) {
	switch(party) {
        case SERVER:
            return CLIENT;
        default: // CLIENT
            return SERVER;
    }	
}

template<typename T, typename I>
int ROGBase<T, I>::numShares() {
    return 1;
}

template<typename T, typename I>
ROG<T, I>::ROG(DeviceData<T, I> *a) : ROGBase<T, I>(a) {}

template<typename T>
ROG<T, BufferIterator<T> >::ROG(DeviceData<T> *a) :
    ROGBase<T, BufferIterator<T> >(a) {}

template<typename T>
ROG<T, BufferIterator<T> >::ROG(size_t n) :
    _shareA(n),
    ROGBase<T, BufferIterator<T> >(&_shareA) {}

template<typename T>
ROG<T, BufferIterator<T> >::ROG(std::initializer_list<double> il, bool convertToFixedPoint) :
    _shareA(il.size()),
    ROGBase<T, BufferIterator<T> >(&_shareA) {

    std::vector<T> shifted_vals;
    for (double f : il) {
        if (convertToFixedPoint) {
            shifted_vals.push_back((T) (f * (1 << FLOAT_PRECISION)));
        } else {
            shifted_vals.push_back((T) f);
        }
    }

    switch (partyNum) {
        case ROG<T>::SERVER:
            thrust::copy(shifted_vals.begin(), shifted_vals.end(), _shareA.begin());
            break;
        case ROG<T>::CLIENT:
            // nothing
            break;
    }
}

template<typename T>
void ROG<T, BufferIterator<T> >::resize(size_t n) {
    _shareA.resize(n);
}

template<typename T, typename I>
void dividePublic(ROG<T, I> &a, T denominator) {

    *a.getShare(0) /= denominator;
}

template<typename T, typename I, typename I2>
void dividePublic(ROG<T, I> &a, DeviceData<T, I2> &denominators) {

    assert(denominators.size() == a.size() && "ROG dividePublic powers size mismatch");

    *a.getShare(0) /= denominators;
}

template<typename T, typename U, typename I, typename I2>
void dividePublic_no_off1(ROG<T, I> &a, T denominator, ROG<U, I2> &result) {

    size_t size = a.size();

    DeviceData<T> d(size);
    d.fill(denominator);
    dividePublic_no_off1(a, d, result);
}

template<typename T, typename U, typename I, typename I2, typename I3>
void dividePublic_no_off1(ROG<T, I> &a, DeviceData<T, I2> &denominators, ROG<U, I3> &result) {

    assert(denominators.size() == a.size() && "ROG dividePublic powers size mismatch");

    // TODO: int8 or int4 support.
    size_t size = a.size();
    result.zero();

    // step 1:  SERVER samples r and send xs + r to CLIENT.
    //          CLIENT computes z = x + r.
    DeviceData<T> r(size);
    if (partyNum == ROG<uint32_t>::SERVER) {
        // generate r.
        // TODO: randomness.
        r.fill(0);
        
        *a.getShare(0) += r;
        comm_profiler.start();
        a.getShare(0)->transmit(ROG<T>::otherParty(partyNum));
        a.getShare(0)->join();
        comm_profiler.accumulate("comm-time");
    }
    else if (partyNum == ROG<uint32_t>::CLIENT) {
        // TODO: randomness.
        r.fill(0);
        comm_profiler.start();
        r.receive(ROG<T>::otherParty(partyNum));
        r.join();
        comm_profiler.accumulate("comm-time");
        // compute z.
        r += *a.getShare(0);
    }

    // step 2:  SERVER and CLIENT run a millionaire's protocol.
    using SRIterator = typename StridedRange<I>::iterator;
    DeviceData<T> rmodd(size);
    rmodd.fill(0);
    rmodd += r;
    rmodd %= denominators;

    // step 3: compute <1{rmodd <= zmodd}>_2
    ROG<T> rmodd_(&rmodd);
    ROG<U> compare_result(size);
    compare_result.zero();
    privateCompare(rmodd_, compare_result);

    // Step 4: the final step.
    r /= denominators;
    r &= 1;
    thrust::copy(r.begin(), r.end(), result.getShare(0)->begin());
    result ^= compare_result;

    func_profiler.add_comm_round();
}

/// @brief SERVER and CLIENT run a millionaire's protocool, output
/// <1{x>y}>_2, where the input of SERVER is x and CLIENT y.
/// @tparam T DeviceData datatype.
/// @tparam I DeviceData iterator.
/// @param a The input of SERVER or CLIENT.
template<typename T, typename U, typename I, typename I2>
void privateCompare(ROG<T, I> &input, ROG<U, I2> &result) {
    // TODO: int8 or int4 support.  uint8 √
    // notice: uint8 is enough to hold prexor.
    size_t T_bits_count = sizeof(T) * 8;
    size_t size = input.size();

    // Commom variable.
    int offset, stride;
    DeviceData<U> b(size*(T_bits_count));
    DeviceData<U> delta(size);
    b.fill(0), delta.fill(0);
    using SRIterator = typename StridedRange<I2>::iterator;

    // MILL step 1: bit expand. because r is known to SERVER, so the bit expand of r is trival.
    // TODO: int8 support.
    gpu::bitexpand(input.getShare(0), &b);
    
    if (partyNum == ROG<uint32_t>::SERVER) {
        // MILL step 2: SERVER sample deltas, compute alpha = 1 - 2*deltas. 
        // TODO: randomn number.
        // test passed. run test passed.
        DeviceData<U> alpha(size * T_bits_count);
        gpu::vectorExpand(&delta, &alpha, T_bits_count);
        alpha *= static_cast<U>(-2);
        alpha += p+1;
        alpha %= p;

        DeviceData<U> rbi(size * T_bits_count);
        DeviceData<U> rbn1(size);
        rbi.fill(1);
        rbn1.fill(1);

        // MILL step 3: SERVER and CLIENT evaluate bi together.
        stride = T_bits_count;
        ROG<U> bi_xor(size*T_bits_count);
        bi_xor.fill(0);
        *bi_xor.getShare(0) += b;
        bi_xor.offline_known = true;
        ROG<U> another_input(size*T_bits_count);
        another_input.fill(0);
        another_input *= bi_xor;
        another_input %= p;
        another_input *= static_cast<U>(-2);
        another_input += b;
        another_input %= p;
        ROG<U> &prefix_xor = another_input;
        prefix_xor *= 3;
        prefix_xor %= p;

        // note the output of bitexpand is little  endian.
        thrust::reverse_iterator<I2> reverse_prefix_xor_iter(prefix_xor.getShare(0)->end());
        thrust::counting_iterator<U> key_count_iter(0);
        DeviceData<U> key(size);
        thrust::copy(key_count_iter, key_count_iter + key.size(), key.begin());
        DeviceData<U> key_expand(size * T_bits_count);
        gpu::vectorExpand(&key, &key_expand, T_bits_count);
        thrust::inclusive_scan_by_key(key_expand.begin(), key_expand.end(), reverse_prefix_xor_iter, reverse_prefix_xor_iter);
        b += *prefix_xor.getShare(0);
        b += delta;
        b %= p;

        // MILL step 4: computes b_{-1}.
        // now, alpha becomes b_{-1}.
        // test pass. run test suspend.
        DeviceData<U> bn1(size);
        bn1.zero();
        bn1 += delta;
        bn1 *= 3;
        StridedRange<I2> bn1_range(prefix_xor.getShare(0)->begin(), prefix_xor.getShare(0)->end(), T_bits_count);
        DeviceData<U, SRIterator> reduce_xor(bn1_range.begin(), bn1_range.end());
        bn1 += reduce_xor;
        b *= rbi;
        bn1 *= rbn1;
        b %= p;
        bn1 %= p;

        // MILL step 5: transmission.
        comm_profiler.start();
        bn1.transmit(ROG<T>::otherParty(partyNum));
        bn1.join();
        b.transmit(ROG<T>::otherParty(partyNum));
        b.join();
        comm_profiler.accumulate("comm-time");

        // // MILL step 6: pass.
        // // PAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAASS.
        result.zero();
        *result.getShare(0) += delta;
    }
    else if (partyNum == ROG<uint32_t>::CLIENT) {

        // MILL step 2: pass.
        // TODO: randomn number.
        // PAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAASS.

        // MILL step 3: SERVER and CLIENT evaluate bi together.
        stride = T_bits_count;
        ROG<U> bi_xor(size*T_bits_count);
        bi_xor.fill(0);
        *bi_xor.getShare(0) += b;
        ROG<U> another_input(size*T_bits_count);
        another_input.fill(0);
        another_input.offline_known = true;
        bi_xor *= another_input;
        bi_xor %= p;
        bi_xor *= static_cast<U>(-2);
        *bi_xor.getShare(0) += b;
        bi_xor %= p;
        ROG<U> &prefix_xor = another_input;
        prefix_xor *= 3;
        prefix_xor %= p;
        // note the output of bitexpand is small endian.
        thrust::reverse_iterator<I2> reverse_prefix_xor_iter(prefix_xor.getShare(0)->end());
        thrust::counting_iterator<U> key_count_iter(0);
        DeviceData<U> key(size);
        thrust::copy(key_count_iter, key_count_iter + key.size(), key.begin());
        DeviceData<U> key_expand(size * T_bits_count);
        gpu::vectorExpand(&key, &key_expand, T_bits_count);
        thrust::inclusive_scan_by_key(key_expand.begin(), key_expand.end(), reverse_prefix_xor_iter, reverse_prefix_xor_iter);
        b *= static_cast<U>(-1);
        b += *prefix_xor.getShare(0);
        b %= p;

        // MILL step 4: computes b_{-1}.
        // now, alpha becomes b_{-1}.
        DeviceData<U> bn1(size);
        bn1.zero();
        bn1 *= 3;
        StridedRange<I2> bn1_range(prefix_xor.getShare(0)->begin(), prefix_xor.getShare(0)->end(), T_bits_count);
        DeviceData<U, SRIterator> reduce_xor(bn1_range.begin(), bn1_range.end());
        bn1 += reduce_xor;
        bn1 %= p;

        // MILL step 5: transmission.
        DeviceData<U> recvbi(size * T_bits_count);
        DeviceData<U> recvbn1(size);
        recvbi.fill(0), recvbn1.fill(0);
        comm_profiler.start();
        recvbn1.receive(ROG<T>::otherParty(partyNum));
        recvbn1.join();
        recvbi.receive(ROG<T>::otherParty(partyNum));
        recvbi.join();
        comm_profiler.accumulate("comm-time");
        b += recvbi;
        bn1 += recvbn1;
        b %= p;
        bn1 %= p;

        // MILL step 6: CLIENT check if there is any 0.
        // Lets work.        
        thrust::transform(b.begin(), b.end(), b.begin(), is_not_a_functor<T>(0));
        thrust::transform(bn1.begin(), bn1.end(), bn1.begin(), is_not_a_functor<T>(0));

        stride = 2;
        while (stride < T_bits_count) {
            offset = (size_t) (stride / 2);
            StridedRange<I2> b_even_range(b.begin(), b.end(), stride);
            StridedRange<I2> b_odd_range(b.begin()  + offset, b.end(), stride);
            DeviceData<U, SRIterator> b_even(b_even_range.begin(), b_even_range.end());
            DeviceData<U, SRIterator> b_odd(b_odd_range.begin(), b_odd_range.end());
            b_even *= b_odd;
            b_even %= p;
            stride *= 2;
        }
        StridedRange<I2> b_range(b.begin(), b.end(), stride);
        DeviceData<U, SRIterator> b_(b_range.begin(), b_range.end());
        bn1 *= b_;
        result.zero();
        *result.getShare(0) += bn1;
    }
    func_profiler.add_comm_round();
}

template<typename T, typename I, typename I2>
void reconstruct(ROG<T, I> &in, DeviceData<T, I2> &out) {

    comm_profiler.start();
    // 1 - send shareA to next party
    in.getShare(0)->transmit(ROG<T>::otherParty(partyNum));

    // 2 - receive shareA from previous party into DeviceBuffer 
    DeviceData<T> rxShare(in.size());
    rxShare.receive(ROG<T>::otherParty(partyNum));

    in.getShare(0)->join();
    rxShare.join();
    comm_profiler.accumulate("comm-time");

    // 3 - result is our shareB + received shareA
    out.zero();
    out += *in.getShare(0);
    out += rxShare;

    func_profiler.add_comm_round();
}

template<typename T>
void matmul(const ROG<T> &a, const ROG<T> &b, ROG<T> &c,
        int M, int N, int K,
        bool transpose_a, bool transpose_b, bool transpose_c, T truncation) {

    localMatMul(a, b, c, M, N, K, transpose_a, transpose_b, transpose_c);

    // truncate
    // dividePublic(c, (T)1 << truncation);
    c >>= truncation;
}

/**
 * return b*(y-x) + x. if b=0, return x; else return y. b is a boolean sharing.
*/
// template<typename T, typename U, typename I, typename I2, typename I3, typename I4>
// void selectShare(const ROG<T, I> &x, const ROG<T, I2> &y, const ROG<U, I3> &b, ROG<T, I4> &z) {

//     assert(x.size() == y.size() && x.size() == b.size() && x.size() == z.size() && "ROG selectShare input size mismatch");

//     //TO_BE_DONE
//     ROG<T> c(x.size());
//     ROG<U> cbits(b.size());

//     // b XOR c, then open -> e
//     cbits ^= b;

//     DeviceData<U> e(cbits.size());
//     reconstruct(cbits, e);

//     // d = 1-c if e=1 else d = c       ->        d = (e)(1-c) + (1-e)(c)
//     ROG<T> d(e.size());
//     convex_comb(d, c, e);

//     // z = ((y - x) * d) + x
//     ROG<T> result(x.size());
//     result += y;
//     result -= x;
//     result *= d;
//     result += x;
    
//     z.zero();
//     z += result;
// }

/**
 * return b*(y-x) + x. if b=0, return x; else return y. b is a arithmatic sharing.
*/
template<typename T, typename U, typename I, typename I2, typename I3, typename I4>
void selectShare(const ROG<T, I> &x, const ROG<T, I2> &y, const ROG<U, I3> &b, ROG<T, I4> &z) {

    assert(x.size() == y.size() && x.size() == b.size() && x.size() == z.size() && "ROG selectShare input size mismatch");

    //TO_BE_DONE
    ROG<T> b_T(x.size());
    b_T.zero();
    z.zero();
    z += y;
    z -= x;
    thrust::copy(b.getShare(0)->begin(), 
        b.getShare(0)->end(),
        b_T.getShare(0)->begin());
    z *= b_T;
    z += x;
}

// template<typename T, typename U, typename I, typename I2, typename I3, typename I4>
// void selectShare(const ROG<T, I> &x, const ROG<T, I2> &y, const ROG<U, I3> &b, ROG<T, I4> &z) {

//     assert(x.size() == y.size() && x.size() == b.size() && x.size() == z.size() && "ROG selectShare input size mismatch");

//     //TO_BE_DONE
//     ROG<T> c(x.size());
//     ROG<U> cbits(b.size());

//     // b XOR c, then open -> e
//     cbits ^= b;

//     DeviceData<U> e(cbits.size());
//     reconstruct(cbits, e);

//     // d = 1-c if e=1 else d = c       ->        d = (e)(1-c) + (1-e)(c)
//     ROG<T> d(e.size());
//     convex_comb(d, c, e);

//     // z = ((y - x) * d) + x
//     ROG<T> result(x.size());
//     result += y;
//     result -= x;
//     result *= d;
//     result += x;
    
//     z.zero();
//     z += result;
// }

template<typename T, typename I, typename I2>
void sqrt(ROG<T, I> &in, ROG<T, I2> &out) {
    /*
     * Approximations:
     *   > sqrt(x) = 0.424 + 0.584(x)
     *     sqrt(x) = 0.316 + 0.885(x) - 0.202(x^2)
     */
    taylorSeries(in, out, 0.424, 0.584, 0.0, sqrt_lambda());
}

template<typename T, typename I, typename I2>
void inverse(ROG<T, I> &in, ROG<T, I2> &out) {
    /*
     * Approximations:
     *     1/x = 2.838 - 1.935(x)
     *   > 1/x = 4.245 - 5.857(x) + 2.630(x^2)
     */
    taylorSeries(in, out, 4.245, -5.857, 2.630, inv_lambda());
}

template<typename T, typename I, typename I2>
void sigmoid(ROG<T, I> &in, ROG<T, I2> &out) {
    /*
     * Approximation:
     *   > sigmoid(x) = 0.494286 + 0.275589(x) + -0.038751(x^2)
     */
    taylorSeries(in, out, 0.494286, 0.275589, -0.038751, sigmoid_lambda());
}

template<typename T>
void localFprop(const ROG<T> &A, const ROG<T> &B, ROG<T> &C,
        int batchSize, int imageHeight, int imageWidth, int Din,
        int Dout, int filterHeight, int filterWidth,
        int paddingHeight, int paddingWidth,
        int stride, int dilation) {

    ROG<T> x(A.size()), y(B.size()), z(C.size());
    if (!B.offline_known)
    {
        PrecomputeObject.getConvBeaverTriple_fprop<T, ROG<T> >(x, y, z, 
            batchSize, imageHeight, imageWidth, Din,
            Dout, filterHeight, filterWidth,
            paddingHeight, paddingWidth,
            stride, dilation);
        DeviceData<T> e(x.size()), f(y.size()), temp(z.size());

        x += A; y += B;
        reconstruct(x, e); reconstruct(y, f);
        x -= A; y -= B;

        C.zero();
        C += z;

        gpu::conv_fprop(&e, &f, &temp, 
            batchSize, imageHeight, imageWidth, Din,
            Dout, filterHeight, filterWidth,
            paddingHeight, paddingWidth,
            stride, dilation);
        C += temp;
        temp.zero();

        gpu::conv_fprop(&e, y.getShare(0), &temp, 
            batchSize, imageHeight, imageWidth, Din,
            Dout, filterHeight, filterWidth,
            paddingHeight, paddingWidth,
            stride, dilation);
        *C.getShare(0) -= temp;
        temp.zero();

        gpu::conv_fprop(x.getShare(0), &f, &temp, 
            batchSize, imageHeight, imageWidth, Din,
            Dout, filterHeight, filterWidth,
            paddingHeight, paddingWidth,
            stride, dilation);
        *C.getShare(0) -= temp;

        cudaThreadSynchronize();
    }
    else
    {
        // printf("-----------------\nOffline branch entered.\n-----------------\n");
        if (partyNum == ROG<uint32_t>::CLIENT) {
            // TODO: offline phase.
            DeviceData<T> Zc(C.size()); 
            Zc.zero();

            C.zero();
            *C.getShare(0) += Zc;
        }
        else if (partyNum == ROG<uint32_t>::SERVER) {
            // TODO: offline phase.
            DeviceData<T> Sz(C.size()), Rs(C.size());
            Sz.zero(), Rs.zero();

            DeviceData<T> a_copy(A.size()), b_copy(B.size());
            a_copy.zero(), b_copy.zero();
            a_copy += *A.getShare(0), b_copy += *B.getShare(0);
            gpu::conv_fprop(&a_copy, &b_copy, C.getShare(0), 
                batchSize, imageHeight, imageWidth, Din,
                Dout, filterHeight, filterWidth,
                paddingHeight, paddingWidth,
                stride, dilation);

            *C.getShare(0) += Sz;
            *C.getShare(0) -= Rs;
        }
    }
}

template<typename T>
void localDgrad(const ROG<T> &A, const ROG<T> &B, ROG<T> &C,
        int batchSize, int outputHeight, int outputWidth, int Dout,
        int filterHeight, int filterWidth, int Din,
        int paddingHeight, int paddingWidth, int stride, int dilation,
        int imageHeight, int imageWidth) {

    ROG<T> x(A.size()), y(B.size()), z(C.size());
    if (!B.offline_known)
    {
        PrecomputeObject.getConvBeaverTriple_dgrad<T, ROG<T> >(x, y, z, 
            batchSize, outputHeight, outputWidth, Dout,
            filterHeight, filterWidth, Din,
            paddingHeight, paddingWidth, stride, dilation,
            imageHeight, imageWidth);
        DeviceData<T> e(x.size()), f(y.size()), temp(z.size());

        x += A; y += B;
        reconstruct(x, e); reconstruct(y, f);
        x -= A; y -= B;

        C.zero();
        C += z;

        gpu::conv_dgrad(&e, &f, &temp, 
            batchSize, outputHeight, outputWidth, Dout,
            filterHeight, filterWidth, Din,
            paddingHeight, paddingWidth, stride, dilation,
            imageHeight, imageWidth);
        C += temp;
        temp.zero();

        gpu::conv_dgrad(&e, y.getShare(0), &temp, 
            batchSize, outputHeight, outputWidth, Dout,
            filterHeight, filterWidth, Din,
            paddingHeight, paddingWidth, stride, dilation,
            imageHeight, imageWidth);
        *C.getShare(0) -= temp;
        temp.zero();

        gpu::conv_dgrad(x.getShare(0), &f, &temp, 
            batchSize, outputHeight, outputWidth, Dout,
            filterHeight, filterWidth, Din,
            paddingHeight, paddingWidth, stride, dilation,
            imageHeight, imageWidth);
        *C.getShare(0) -= temp;

        cudaThreadSynchronize();
    }
    else
    {
        // printf("-----------------\nOffline branch entered.\n-----------------\n");
        if (partyNum == ROG<uint32_t>::CLIENT) {
            // TODO: offline phase.
            DeviceData<T> Zc(C.size()); 
            Zc.zero();

            C.zero();
            *C.getShare(0) += Zc;
        }
        else if (partyNum == ROG<uint32_t>::SERVER) {
            // TODO: offline phase.
            DeviceData<T> Sz(C.size()), Rs(C.size());
            Sz.zero(), Rs.zero();

            DeviceData<T> a_copy(A.size()), b_copy(B.size());
            a_copy.zero(), b_copy.zero();
            a_copy += *A.getShare(0), b_copy += *B.getShare(0);
            gpu::conv_dgrad(&a_copy, &b_copy, C.getShare(0), 
                batchSize, outputHeight, outputWidth, Dout,
                filterHeight, filterWidth, Din,
                paddingHeight, paddingWidth, stride, dilation,
                imageHeight, imageWidth);

            *C.getShare(0) += Sz;
            *C.getShare(0) -= Rs;
        }
    }
}

template<typename T>
void localWgrad(const ROG<T> &A, const ROG<T> &B, ROG<T> &C,
        int batchSize, int outputHeight, int outputWidth, int Dout,
        int imageHeight, int imageWidth, int Din,
        int filterHeight, int filterWidth,
        int paddingHeight, int paddingWidth, int stride, int dilation) {

    ROG<T> x(A.size()), y(B.size()), z(C.size());
    if (!B.offline_known)
    {
        PrecomputeObject.getConvBeaverTriple_wgrad<T, ROG<T> >(x, y, z, 
            batchSize, outputHeight, outputWidth, Dout,
            filterHeight, filterWidth, Din,
            paddingHeight, paddingWidth, stride, dilation,
            imageHeight, imageWidth);
        DeviceData<T> e(x.size()), f(y.size()), temp(z.size());

        x += A; y += B;
        reconstruct(x, e); reconstruct(y, f);
        x -= A; y -= B;

        C.zero();
        C += z;

        gpu::conv_wgrad(&e, &f, &temp, 
            batchSize, outputHeight, outputWidth, Dout,
            imageHeight, imageWidth, Din,
            filterHeight, filterWidth,
            paddingHeight, paddingWidth, stride, dilation);
        C += temp;
        temp.zero();

        gpu::conv_wgrad(&e, y.getShare(0), &temp, 
            batchSize, outputHeight, outputWidth, Dout,
            imageHeight, imageWidth, Din,
            filterHeight, filterWidth,
            paddingHeight, paddingWidth, stride, dilation);
        *C.getShare(0) -= temp;
        temp.zero();

        gpu::conv_wgrad(x.getShare(0), &f, &temp, 
            batchSize, outputHeight, outputWidth, Dout,
            imageHeight, imageWidth, Din,
            filterHeight, filterWidth,
            paddingHeight, paddingWidth, stride, dilation);
        *C.getShare(0) -= temp;

        cudaThreadSynchronize();
    }
    else
    {
        // printf("-----------------\nOffline branch entered.\n-----------------\n");
        if (partyNum == ROG<uint32_t>::CLIENT) {
            // TODO: offline phase.
            DeviceData<T> Zc(C.size()); 
            Zc.zero();

            C.zero();
            *C.getShare(0) += Zc;
        }
        else if (partyNum == ROG<uint32_t>::SERVER) {
            // TODO: offline phase.
            DeviceData<T> Sz(C.size()), Rs(C.size());
            Sz.zero(), Rs.zero();

            DeviceData<T> a_copy(A.size()), b_copy(B.size());
            a_copy.zero(), b_copy.zero();
            a_copy += *A.getShare(0), b_copy += *B.getShare(0);
            gpu::conv_wgrad(&a_copy, &b_copy, C.getShare(0), 
                batchSize, outputHeight, outputWidth, Dout,
                imageHeight, imageWidth, Din,
                filterHeight, filterWidth,
                paddingHeight, paddingWidth, stride, dilation);

            *C.getShare(0) += Sz;
            *C.getShare(0) -= Rs;
        }
    }
}

template<typename T>
void convolution(const ROG<T> &A, const ROG<T> &B, ROG<T> &C,
        cutlass::conv::Operator op,
        int batchSize, int imageHeight, int imageWidth, int filterSize,
        int Din, int Dout, int stride, int padding, int truncation) {

    int outputHeight = (imageHeight + 2 * padding - filterSize) / stride + 1; 
    int outputWidth = (imageWidth + 2 * padding - filterSize) / stride + 1; 
    C.zero();
    // DeviceData<T> localResult(C.size());

    switch (op) {
        case cutlass::conv::Operator::kFprop:
            localFprop(A, B, C,
                    batchSize, imageHeight, imageWidth, Din,
                    Dout, filterSize, filterSize,
                    padding, padding,
                    stride, (T)1);
            break;
        case cutlass::conv::Operator::kDgrad:
            localDgrad(A, B, C,
                    batchSize, outputHeight, outputWidth, Dout,
                    filterSize, filterSize, Din,
                    padding, padding, stride, (T)1,
                    imageHeight, imageWidth);
            break;
        case cutlass::conv::Operator::kWgrad:
            localWgrad(A, B, C,
                    batchSize, outputHeight, outputWidth, Dout,
                    imageHeight, imageWidth, Din,
                    filterSize, filterSize,
                    padding, padding, stride, (T)1);
            break;
    }

    // *C.getShare(0) += localResult;
    // dividePublic(C, (T)1 << truncation);
    C >>= truncation;
}

// TODO change into 2 arguments with subtraction, pointer NULL indicates compare w/ 0
/// @brief dReLU(x) = 1{x>=0} = DevideC(x, bound)
/// @tparam T       input datatype.
/// @tparam U       Output datatype.
/// @tparam I       Input iterator.
/// @tparam I2      Output iterator.
/// @param input    <x>.
/// @param result   <1{x>=0}>_b.
template<typename T, typename U, typename I, typename I2>
void dReLU(const ROG<T, I> &input, ROG<U, I2> &result) {

    size_t size = input.size();

    // TODO: how to constrain in input x's range?
    T bound = ROGUE_BOUND;
    ROG<T> input_(size);
    input_.zero();
    input_ += input;
    input_ += bound;
    dividePublic_no_off1(input_, bound, result);
}
    
template<typename T, typename U, typename I, typename I2, typename I3>
void ReLU(const ROG<T, I> &input, ROG<T, I2> &result, ROG<U, I3> &dresult) {
    MTPC<T> minput(input.size());
    minput.zero();

    func_profiler.start();
    reshare(input, minput);
    dReLU(input, dresult);
    FusionMux(minput, dresult, result);
    func_profiler.accumulate("relu-drelu");
}

template<typename T, typename U, typename I, typename I2, typename I3>
void FusionMux(MTPC<T, I> &x, ROG<U, I2> &b, ROG<T, I3> &result) {
    int size = x.size();

    // TODO: offline.
    ROG<T> dbdx(size), tb(size), db(size);
    MTPC<T> bang(size);
    dbdx.zero(), bang.zero(), result.zero(), db.zero();
    thrust::copy(b.getShare(0)->begin(), 
        b.getShare(0)->end(),  
        tb.getShare(0)->begin());
    if (partyNum == ROG<uint32_t>::SERVER) {
        *bang.getShare(1) += *tb.getShare(0);
    }
    *db.getShare(0) += *bang.getShare(1);

    // online.
    *bang.getShare(0) ^= *bang.getShare(1);
    *bang.getShare(0) ^= *tb.getShare(0);
    
    // Compute D_b and [\zeta]^C. 
    if (partyNum == ROG<uint32_t>::CLIENT) {
        *db.getShare(0) *= *x.getShare(0);
        *db.getShare(0) -= *dbdx.getShare(0);

        // now, dbdx holds (1-2D_b).
        dbdx.getShare(0)->zero();
        *dbdx.getShare(0) += *bang.getShare(0);
        *dbdx.getShare(0) *= static_cast<T>(-2);
        *dbdx.getShare(0) += 1;

        *db.getShare(0) *= *dbdx.getShare(0);
        dbdx.getShare(0)->zero();
        *dbdx.getShare(0) -= *result.getShare(0);
        *dbdx.getShare(0) += *db.getShare(0);

        db.getShare(0)->zero();
        *db.getShare(0) += *bang.getShare(0);
        *db.getShare(0) *= *x.getShare(1);
        *dbdx.getShare(0) -= *db.getShare(0);

        bang.getShare(0)->transmit(ROG<T>::otherParty(partyNum));
        bang.getShare(0)->join();
        dbdx.getShare(0)->transmit(ROG<T>::otherParty(partyNum));
        dbdx.getShare(0)->join();
    }
    else if (partyNum == ROG<uint32_t>::SERVER) {
        bang.getShare(0)->receive(ROG<T>::otherParty(partyNum));
        bang.getShare(0)->join();
        result.getShare(0)->receive(ROG<T>::otherParty(partyNum));
        result.getShare(0)->join();

        *db.getShare(0) *= *x.getShare(0);
        *db.getShare(0) -= *dbdx.getShare(0);

        // now, dbdx holds (1-2D_b).
        dbdx.getShare(0)->zero();
        *dbdx.getShare(0) += *bang.getShare(0);
        *dbdx.getShare(0) *= static_cast<T>(-2);
        *dbdx.getShare(0) += 1;
        *db.getShare(0) *= *dbdx.getShare(0);

        dbdx.getShare(0)->zero();
        *dbdx.getShare(0) += *x.getShare(0);
        *dbdx.getShare(0) -= *x.getShare(1);
        *bang.getShare(0) *= *dbdx.getShare(0);

        *result.getShare(0) += *bang.getShare(0);
        *result.getShare(0) += *db.getShare(0);
    }
}

template<typename T, typename U, typename I, typename I2, typename I3>
void maxpool(ROG<T, I> &input, ROG<T, I2> &result, ROG<U, I3> &dresult, int k) {

    //TO_BE_DONE

    // d(Maxpool) setup
    dresult.fill(1);

    // split input into even, odd
    using SRIterator = typename StridedRange<I>::iterator;

    int stride = 2;
    int offset = 1;

    func_profiler.start();
    StridedRange<I> even0Range(input.getShare(0)->begin(), input.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> even0(even0Range.begin(), even0Range.end());
    ROG<T, SRIterator> even(&even0);

    StridedRange<I> odd0Range(input.getShare(0)->begin() + offset, input.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> odd0(odd0Range.begin(), odd0Range.end());
    ROG<T, SRIterator> odd(&odd0);
    func_profiler.accumulate("range creation");

    //printf("func-maxpool-post-rangecreate\n");
    //printMemUsage();

    while(k > 2) {

        // -- MP --

        // diff = even - odd
        func_profiler.start();
        ROG<T> diff(even.size());
        diff.zero();
        diff += even;
        diff -= odd;
        func_profiler.accumulate("maxpool-diff");

        //printf("func-maxpool-post-diff-k=%d\n", k);
        //printMemUsage();

        // DRELU diff -> b
        func_profiler.start();
        ROG<U> b(even.size());
        dReLU(diff, b);
        func_profiler.accumulate("maxpool-drelu");

        //printf("func-maxpool-post-drelu-k=%d\n", k);
        //printMemUsage();
     
        // TODO: remove copy.
        ROG<T> b_T(b.size());
        thrust::copy(b.getShare(0)->begin(), 
            b.getShare(0)->end(),
            b_T.getShare(0)->begin());
            
        // selectShare(odd, even, b, even);
        even -= odd;
        even *= b_T;
        even += odd;

        // unzip even -> into even, odd
        stride *= 2;

        //printf("func-maxpool-pre-rangeupdate-k=%d\n", k);
        //printMemUsage();

        func_profiler.start();
        even0Range.set(input.getShare(0)->begin(), input.getShare(0)->end(), stride);
        even0.set(even0Range.begin(), even0Range.end());
        even.set(&even0);

        odd0Range.set(input.getShare(0)->begin() + stride/2, input.getShare(0)->end(), stride);
        odd0.set(odd0Range.begin(), odd0Range.end());
        odd.set(&odd0);
        func_profiler.accumulate("maxpool-unzip");
     
        // -- dMP --

        //printf("func-maxpool-pre-expand-k=%d\n", k);
        //printMemUsage();

        // expandCompare b -> expandedB
        func_profiler.start();
        ROG<U> negated(b.size());
        negated.fill(1);
        negated -= b;
        ROG<U> expandedB(input.size());

        gpu::expandCompare(*b.getShare(0), *negated.getShare(0), *expandedB.getShare(0));

        func_profiler.accumulate("maxpool-expandCompare");

        //printf("func-maxpool-post-expand-k=%d\n", k);
        //printMemUsage();
     
        // dresult &= expandedB
        func_profiler.start();
        dresult *= expandedB;
        func_profiler.accumulate("maxpool-dcalc");

        k /= 2;
    }

    // Fencepost - don't unzip the final results after the last comparison and finish
    // calculating derivative.
 
    // -- MP --
 
    // diff = even - odd
    func_profiler.start();
    ROG<T> diff(even.size());
    diff.zero();
    diff += even;
    diff -= odd;
    func_profiler.accumulate("maxpool-z-diff");

    // DRELU diff -> b
    func_profiler.start();
    ROG<U> b(even.size());
    dReLU(diff, b);
    func_profiler.accumulate("maxpool-z-drelu");
 
    func_profiler.start();
    // TODO: remove copy.
    ROG<T> b_T(b.size());
    thrust::copy(b.getShare(0)->begin(), 
        b.getShare(0)->end(),
        b_T.getShare(0)->begin());
        
    // selectShare(odd, even, b, even);
    even -= odd;
    even *= b_T;
    even += odd;
    func_profiler.accumulate("maxpool-selectShare");

    func_profiler.start();
    //even *= b;
    //odd *= negated;
    //even += odd;

    result.zero();
    result += even;
    func_profiler.accumulate("maxpool-z-calc");

    // -- dMP --

    // expandCompare b -> expandedB
    func_profiler.start();
    ROG<U> negated(b.size());
    negated.fill(1);
    negated -= b;
    ROG<U> expandedB(input.size());
    gpu::expandCompare(*b.getShare(0), *negated.getShare(0), *expandedB.getShare(0));
    func_profiler.accumulate("maxpool-z-expandCompare");
 
    // dresult &= expandedB
    func_profiler.start();
    dresult *= expandedB;
    func_profiler.accumulate("maxpool-z-dcalc");
}

template<typename T>
void localMatMul(const ROG<T> &a, const ROG<T> &b, ROG<T> &c,
        int M, int N, int K,
        bool transpose_a, bool transpose_b, bool transpose_c) {
    
    ROG<T> x(a.size()), y(b.size()), z(c.size());

    int a_rows = transpose_a ? K : M; int a_cols = transpose_a ? M : K;
    int b_rows = transpose_b ? N : K; int b_cols = transpose_b ? K : N;
    if (!b.offline_known)
    {
        PrecomputeObject.getMatrixBeaverTriple<T, ROG<T> >(x, y, z, a_rows, a_cols, b_rows, b_cols, transpose_a, transpose_b);

        DeviceData<T> e(x.size()), f(y.size()), temp(z.size());

        x += a; y += b;
        reconstruct(x, e);
        reconstruct(y, f);
        x -= a; y -= b;

        c.zero();
        c += z;

        gpu::gemm(M, N, K, &e, transpose_a, &f, transpose_b, &temp, transpose_c);
        c += temp;
        temp.zero();

        gpu::gemm(M, N, K, &e, transpose_a, y.getShare(0), transpose_b, &temp, transpose_c);
        c -= temp;
        temp.zero();

        gpu::gemm(M, N, K, x.getShare(0), transpose_a, &f, transpose_b, &temp, transpose_c);
        c -= temp;  
    }
    else 
    {
        // printf("-----------------\nOffline branch entered.\n-----------------\n");
        if (partyNum == ROG<uint32_t>::CLIENT) {
            // TODO: offline phase.
            DeviceData<T> Zc(c.size()); 
            Zc.zero();

            c.zero();
            *c.getShare(0) += Zc;
        }
        else if (partyNum == ROG<uint32_t>::SERVER) {
            // TODO: offline phase.
            DeviceData<T> Sz(c.size()), Rs(c.size());
            Sz.zero(), Rs.zero();

            DeviceData<T> a_copy(a.size()), b_copy(b.size());
            a_copy.zero(), b_copy.zero();
            a_copy += *a.getShare(0), b_copy += *b.getShare(0);
            gpu::gemm(M, N, K, &a_copy, transpose_a, &b_copy, transpose_b, c.getShare(0), transpose_c);

            *c.getShare(0) += Sz;
            *c.getShare(0) -= Rs;
        }
    }
}

template<typename T, typename I, typename I2>
void carryOut(ROG<T, I> &p, ROG<T, I> &g, int k, ROG<T, I2> &out) {

    // get zip iterators on both p and g
    //  -> pEven, pOdd, gEven, gOdd
 
    int stride = 2;
    int offset = 1;

    using SRIterator = typename StridedRange<I>::iterator;

    StridedRange<I> pEven0Range(p.getShare(0)->begin(), p.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> pEven0(pEven0Range.begin(), pEven0Range.end());
    ROG<T, SRIterator> pEven(&pEven0);

    StridedRange<I> pOdd0Range(p.getShare(0)->begin() + offset, p.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> pOdd0(pOdd0Range.begin(), pOdd0Range.end());
    ROG<T, SRIterator> pOdd(&pOdd0);

    StridedRange<I> gEven0Range(g.getShare(0)->begin(), g.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> gEven0(gEven0Range.begin(), gEven0Range.end());
    ROG<T, SRIterator> gEven(&gEven0);

    StridedRange<I> gOdd0Range(g.getShare(0)->begin() + offset, g.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> gOdd0(gOdd0Range.begin(), gOdd0Range.end());
    ROG<T, SRIterator> gOdd(&gOdd0);

    while(k > 1) {

        // gTemp = pOdd & gEven
        //  store result in gEven
        gEven &= pOdd;

        // pEven & pOdd
        //  store result in pEven
        pEven &= pOdd;

        // gOdd ^ gTemp
        //  store result in gOdd
        gOdd ^= gEven;
     
        // regenerate zip iterators to p and g
     
        //  gOdd -> gEven, gOdd
        gEven0Range.set(g.getShare(0)->begin() + offset, g.getShare(0)->end(), stride*2);
        gEven0.set(gEven0Range.begin(), gEven0Range.end());
        gEven.set(&gEven0);

        offset += stride;

        gOdd0Range.set(g.getShare(0)->begin() + offset, g.getShare(0)->end(), stride*2);
        gOdd0.set(gOdd0Range.begin(), gOdd0Range.end());
        gOdd.set(&gOdd0);

        //  pEven -> pEven, pOdd
        stride *= 2;

        pEven0Range.set(p.getShare(0)->begin(), p.getShare(0)->end(), stride);
        pEven0.set(pEven0Range.begin(), pEven0Range.end());
        pEven.set(&pEven0);

        pOdd0Range.set(p.getShare(0)->begin() + stride/2, p.getShare(0)->end(), stride);
        pOdd0.set(pOdd0Range.begin(), pOdd0Range.end());
        pOdd.set(&pOdd0);
     
        k /= 2;
    }

    // copy output to destination
    // out.zip(gEven, gOdd);
    StridedRange<I> outputEven0Range(out.getShare(0)->begin(), out.getShare(0)->end(), 2);
    thrust::copy(gEven.getShare(0)->begin(), gEven.getShare(0)->end(), outputEven0Range.begin());

    StridedRange<I> outputOdd0Range(out.getShare(0)->begin() + 1, out.getShare(0)->end(), 2);
    thrust::copy(gOdd.getShare(0)->begin(), gOdd.getShare(0)->end(), outputOdd0Range.begin());
}

template<typename T, typename I, typename I2>
void getPowers(ROG<T, I> &in, DeviceData<T, I2> &pow) {

    ROG<T> powers(pow.size()); // accumulates largest power yet tested that is less than the input val
    ROG<T> currentPowerBit(in.size()); // current power
    ROG<T> diff(in.size());
    ROG<uint8_t> comparisons(in.size());

    for (int bit = 0; bit < sizeof(T) * 8; bit++) {
        currentPowerBit.fill(bit);

        diff.zero();
        diff += in;
        diff -= (((T)1) << bit);

        comparisons.zero();
        dReLU(diff, comparisons); // 0 -> current power is larger than input val, 1 -> input val is larger than current power

        // 0 -> keep val, 1 -> update to current known largest power less than input
        // TODO: remove copy.
        ROG<T> b(comparisons.size());
        thrust::copy(comparisons.getShare(0)->begin(), 
            comparisons.getShare(0)->end(),
            b.getShare(0)->begin());
            
        // selectShare(odd, even, b, even);
        currentPowerBit -= powers;
        currentPowerBit *= b;
        powers += currentPowerBit;
        // selectShare(powers, currentPowerBit, comparisons, powers);
    }

    reconstruct(powers, pow);
}

template<typename T, typename I, typename I2, typename Functor>
void taylorSeries(ROG<T, I> &in, ROG<T, I2> &out,
        double a0, double a1, double a2,
        Functor fn) {

    out.zero();
    ROG<T> scratch(out.size());

    DeviceData<T> pow(out.size());
    getPowers(in, pow);
    pow += 1;

    DeviceData<T> ones(pow.size());
    ones.fill(1);
    ones <<= pow;

    if (a2 != 0.0) {
        DeviceData<T> a2Coeff(out.size());
        thrust::transform(
            pow.begin(), pow.end(), a2Coeff.begin(),
            tofixed_variable_precision_functor<T>(a2));

        scratch.zero();
        scratch += in;
        scratch *= in;
        dividePublic(scratch, ones);

        scratch *= a2Coeff;
        dividePublic(scratch, ones);
        out += scratch;
    }

    if (a1 != 0.0) {

        DeviceData<T> a1Coeff(out.size());
        thrust::transform(
            pow.begin(), pow.end(), a1Coeff.begin(),
            tofixed_variable_precision_functor<T>(a1));

        scratch.zero();
        scratch += in;
        scratch *= a1Coeff;
        dividePublic(scratch, ones);

        out += scratch;
    }

    DeviceData<T> a0Coeff(out.size());
    thrust::transform(
        pow.begin(), pow.end(), a0Coeff.begin(),
        tofixed_variable_precision_functor<T>(a0));
    out += a0Coeff;

    DeviceData<T> powCoeff(out.size());
    thrust::transform(
        pow.begin(), pow.end(), powCoeff.begin(),
        calc_fn<T, Functor>(fn));
    out *= powCoeff;

    dividePublic(out, ones);

    // turn values back to base (e.g. 20 bit) precision

    pow -= FLOAT_PRECISION;

    DeviceData<T> positivePow(pow.size());
    thrust::transform(
        pow.begin(), pow.end(), positivePow.begin(),
        filter_positive_powers<T>());

    ones.fill(1);
    ones <<= positivePow;

    dividePublic(out, ones);

    DeviceData<T> negativePow(pow.size());
    thrust::transform(
        pow.begin(), pow.end(), negativePow.begin(),
        filter_negative_powers<T>());

    for (int share = 0; share < ROG<T>::numShares(); share++) {
        thrust::transform(
            out.getShare(share)->begin(), out.getShare(share)->end(), negativePow.begin(), out.getShare(share)->begin(),
            lshift_functor<T>()); 
    }
}

template<typename T, typename U, typename I, typename I2>
void convex_comb(ROG<T, I> &a, ROG<T, I> &c, DeviceData<U, I2> &b) {

    thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(b.begin(), c.getShare(0)->begin(), a.getShare(0)->begin())),
        thrust::make_zip_iterator(thrust::make_tuple(b.end(), c.getShare(0)->end(), a.getShare(0)->end())),
        ROG_convex_comb_functor(partyNum)
    );
}

// convert a [C, .]-sharing to <.>-sharing.
template<typename T, typename I, typename I2>
void reshare(const ROG<T,I> &in, MTPC<T, I2> &out){
    // TODO: offline.
    int size = in.size();
    
    if (partyNum == ROG<T>::CLIENT){
        out.getShare(1)->zero();
        *out.getShare(1) -= *in.getShare(0);
    }

    // online.
    if (partyNum == ROG<T>::SERVER){
        out.getShare(0)->zero();
        *out.getShare(0) += *in.getShare(0);
        *out.getShare(0) += *out.getShare(1);

        out.getShare(0)->transmit(ROG<T>::otherParty(partyNum));
        out.getShare(0)->join();
    }
    else if (partyNum == ROG<T>::CLIENT){
        out.getShare(0)->receive(ROG<T>::otherParty(partyNum));
        out.getShare(0)->join();
    }
}