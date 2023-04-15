/*
 * GFO.inl
 */

#pragma once

#include "GForce.h"

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

struct GFO_convex_comb_functor {
    const int party;
    GFO_convex_comb_functor(int _party) : party(_party) {}
    
    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple t) {
        // b, c share, d share
        if (thrust::get<0>(t) == 1) {
            switch(party) {
                case GFO<uint64_t>::SERVER: // doesn't really matter what type GFO is templated at here
                    thrust::get<2>(t) = 1 - thrust::get<1>(t);
                    break;
                case GFO<uint64_t>::CLIENT:
                    thrust::get<2>(t) = -thrust::get<1>(t);
                    break;
            }
        } else {
            thrust::get<2>(t) = thrust::get<1>(t);
        }
    }
};

template<typename T>
struct GFO_catch_functor {

    const T a;

    GFO_catch_functor(T _a): a(_a) {}
    __host__ __device__ T operator()(const T &x) const {
        return x + x/a + 1;
    }
};

template<typename T>
struct GFO_key_functor {

    const T a;

    GFO_key_functor(T _a): a(_a) {}
    __host__ __device__ T operator()(const T &x) const {
        return x / (a + 1);
    }
};

template<typename T>
struct GFO_mult_and_modular_functor {

    const T prime;

    GFO_mult_and_modular_functor(T _prime): prime(_prime) {}
    __host__ __device__ T operator()(const T &x, const T &y) const {
        return (x * y) % prime;
    }
};

// Prototypes

template<typename T>
void localMatMul(const GFO<T> &a, const GFO<T> &b, const GFO<T> &c,
        int M, int N, int K,
        bool transpose_a, bool transpose_b);

template<typename T, typename I, typename I2, typename I3>
void localConvolution(GFO<T, I> &im, GFO<T, I2> &filters, DeviceData<T, I3> &out,
        size_t imageWidth, size_t imageHeight, size_t filterSize, size_t Din, size_t Dout,
        size_t stride, size_t padding);

template<typename T, typename I, typename I2>
void carryOut(GFO<T, I> &p, GFO<T, I> &g, int k, GFO<T, I2> &out);

template<typename T, typename I, typename I2>
void getPowers(GFO<T, I> &in, DeviceData<T, I2> &pow);

template<typename T, typename I, typename I2, typename Functor>
void taylorSeries(GFO<T, I> &in, GFO<T, I2> &out,
        double a0, double a1, double a2,
        Functor fn);


template<typename T, typename U, typename I, typename I2>
void convex_comb(GFO<T, I> &a, GFO<T, I> &c, DeviceData<U, I2> &b);

// GFO class implementation 

template<typename T, typename I>
GFOBase<T, I>::GFOBase(DeviceData<T, I> *a, bool offline_known, T prime_) : 
                shareA(a), offline_known(offline_known), prime(prime_) {}

template<typename T, typename I>
void GFOBase<T, I>::set(DeviceData<T, I> *a) {
    shareA = a;
}

template<typename T, typename I>
size_t GFOBase<T, I>::size() const {
    return shareA->size();
}

template<typename T, typename I>
void GFOBase<T, I>::zero() {
    shareA->zero();
};

template<typename T, typename I>
void GFOBase<T, I>::fill(T val) {
    shareA->fill(partyNum == SERVER ? val : 0);
}

/// @brief Given an array(double) in CPU, embedding it on GPU.
/// @tparam T 
/// @tparam I 
/// @param v 
template<typename T, typename I>
void GFOBase<T, I>::setPublic(std::vector<double> &v) {
    // typedef typename std::make_signed<T>::type S;
    std::vector<T> shifted_vals;
    for (double f : v) {
        shifted_vals.push_back(
            (static_cast<T>((f * (1 << FLOAT_PRECISION))) + prime) % prime
        );
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
DeviceData<T, I> *GFOBase<T, I>::getShare(int i) {
    switch (i) {
        case 0:
            return shareA;
        default:
            return nullptr;
    }
}

template<typename T, typename I>
const DeviceData<T, I> *GFOBase<T, I>::getShare(int i) const {
    switch (i) {
        case 0:
            return shareA;
        default:
            return nullptr;
    }
}

template<typename T, typename I>
const std::string& GFOBase<T, I>::getProt() {
    const static std::string prot = "GFO";
    return prot;
}

// DO NOT write GFO_A += -1! use GFO_A -= 1!
template<typename T, typename I>
GFOBase<T, I> &GFOBase<T, I>::operator+=(const T rhs) {
    if (partyNum == SERVER) {
        *shareA += rhs;
        *shareA %= prime;
    }
    return *this;
}

template<typename T, typename I>
GFOBase<T, I> &GFOBase<T, I>::operator-=(const T rhs) {
    if (partyNum == SERVER) {
        *shareA += prime - rhs;
        *shareA %= prime;
    }
    return *this;
}

// for any multiple rhs < 0, please use x *= -1; x*= -rhs. Do not use x *= rhs.
template<typename T, typename I>
GFOBase<T, I> &GFOBase<T, I>::operator*=(const T rhs) {
    *shareA *= prime + rhs;
    *shareA %= prime;
    return *this;
}

template<typename T, typename I>
GFOBase<T, I> &GFOBase<T, I>::operator%=(const T rhs) {
    *shareA %= rhs;
    return *this;
}

template<typename T, typename I>
GFOBase<T, I> &GFOBase<T, I>::operator>>=(const T rhs) {
    *shareA >>= rhs;
    return *this;
}

template<typename T, typename I>
GFOBase<T, I> &GFOBase<T, I>::operator^=(const T rhs) {
    *shareA ^= rhs;
    return *this;
}

template<typename T, typename I>
GFOBase<T, I> &GFOBase<T, I>::operator&=(const T rhs) {
    *shareA &= rhs;
    return *this;
}

// op with another DeviceData rhs. rhs must consist of field element.
template<typename T, typename I>
template<typename I2>
GFOBase<T, I> &GFOBase<T, I>::operator+=(const DeviceData<T, I2> &rhs) {
    if (partyNum == SERVER) {
        *shareA += rhs;
        *shareA %= prime;
    }
    return *this;
}

template<typename T, typename I>
template<typename I2>
GFOBase<T, I> &GFOBase<T, I>::operator-=(const DeviceData<T, I2> &rhs) {
    if (partyNum == SERVER) {
        *shareA -= rhs;
        *shareA += prime;
        *shareA %= prime;
    }
    return *this;
}

template<typename T, typename I>
template<typename I2>
GFOBase<T, I> &GFOBase<T, I>::operator*=(const DeviceData<T, I2> &rhs) {
    *shareA *= rhs;
    *shareA %= prime;
    return *this;
}

template<typename T, typename I>
template<typename I2>
GFOBase<T, I> &GFOBase<T, I>::operator^=(const DeviceData<T, I2> &rhs) {
    if (partyNum == SERVER) {
        *shareA ^= rhs;
    }
    return *this;
}

template<typename T, typename I>
template<typename I2>
GFOBase<T, I> &GFOBase<T, I>::operator&=(const DeviceData<T, I2> &rhs) {
    *shareA &= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
GFOBase<T, I> &GFOBase<T, I>::operator>>=(const DeviceData<T, I2> &rhs) {
    *shareA >>= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
GFOBase<T, I> &GFOBase<T, I>::operator<<=(const DeviceData<T, I2> &rhs) {
    *shareA <<= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
GFOBase<T, I> &GFOBase<T, I>::operator+=(const GFOBase<T, I2> &rhs) {
    *shareA += *rhs.getShare(0);
    *shareA %= prime;
    return *this;
}

template<typename T, typename I>
template<typename I2>
GFOBase<T, I> &GFOBase<T, I>::operator-=(const GFOBase<T, I2> &rhs) {
    *shareA -= *rhs.getShare(0);
    *shareA += prime;
    *shareA %= prime;
    return *this;
}

template<typename T, typename I>
template<typename I2>
GFOBase<T, I> &GFOBase<T, I>::operator*=(const GFOBase<T, I2> &rhs) {

    size_t size = rhs.size();
    size_t prime = rhs.prime;

    // T test[size];

    if (!rhs.offline_known)
    {
        // Precomputation.
        GFO<T> x(size, prime), y(size, prime), z(size, prime);
        PrecomputeObject.getBeaverTriples<T, GFO<T> >(x, y, z);
        DeviceData<T> e(size), f(size), temp(size);

        *x.getShare(0) += *this->getShare(0); 
        *x.getShare(0) %= prime;
        *y.getShare(0) += *rhs.getShare(0);
        *y.getShare(0) %= prime;
        reconstruct(x, e, false); reconstruct(y, f, false);
        *x.getShare(0) -= *this->getShare(0);
        *x.getShare(0) += prime;
        *x.getShare(0) %= prime;
        *y.getShare(0) -= *rhs.getShare(0);
        *y.getShare(0) += prime;
        *y.getShare(0) %= prime;

        // thrust::copy(this->getShare(0)->begin(), this->getShare(0)->end(), test);
		// std::cout << "----------- a --------------" << std::endl;
		// for (T t: test) {
		// 	std::cout << t << " ";
		// }
		// std::cout << std::endl;
        // thrust::copy(rhs.getShare(0)->begin(), rhs.getShare(0)->end(), test);
		// std::cout << "----------- b --------------" << std::endl;
		// for (T t: test) {
		// 	std::cout << t << " ";
		// }
		// std::cout << std::endl;
        // thrust::copy(e.begin(), e.end(), test);
		// std::cout << "----------- e --------------" << std::endl;
		// for (T t: test) {
		// 	std::cout << t << " ";
		// }
		// std::cout << std::endl;
        // thrust::copy(f.begin(), f.end(), test);
		// std::cout << "----------- f --------------" << std::endl;
		// for (T t: test) {
		// 	std::cout << t << " ";
		// }
		// std::cout << std::endl;
        
        this->zero();
        *this += z;

        temp.zero();
        temp += f;
        temp *= e;
        temp %= prime;
        *this += temp;
        *this %= prime;
        // thrust::copy(temp.begin(), temp.end(), test);
		// std::cout << "----------- e*f --------------" << std::endl;
		// for (T t: test) {
		// 	std::cout << t << " ";
		// }
		// std::cout << std::endl;

        temp.zero();
        temp -= *y.getShare(0);
        temp += prime;
        temp *= e;
        temp %= prime;
        *this->getShare(0) += temp;
        *this %= prime;
        // thrust::copy(temp.begin(), temp.end(), test);
		// std::cout << "----------- -e*y --------------" << std::endl;
		// for (T t: test) {
		// 	std::cout << t << " ";
		// }
		// std::cout << std::endl;

        temp.zero();
        temp -= *x.getShare(0);
        temp += prime;
        temp *= f;
        temp %= prime;
        *this->getShare(0) += temp;
        *this %= prime;
        // thrust::copy(this->getShare(0)->begin(), this->getShare(0)->end(), test);
		// std::cout << "----------- final --------------" << std::endl;
		// for (T t: test) {
		// 	std::cout << t << " ";
		// }
		// std::cout << std::endl;
    } 
    else 
    {
        // printf("-----------------\nOffline branch entered.\n-----------------\n");
        // SERVER: r^S.     
        // CLIENT: w*r^C-r^S.
        GFO<T> offline_output(size, prime);
        // SERVER: x-r^C.   
        // CLIENT: r^C.
        GFO<T> r(size, prime);
        PrecomputeObject.getCorrelatedRandomness<T, GFOBase<T, I2>, GFO<T>>(const_cast<GFOBase<T, I2> &>(rhs), offline_output, r);

        if (partyNum == CLIENT) {
            *this -= r;
            comm_profiler.start();
            this->getShare(0)->transmit(GFO<T>::otherParty(partyNum));
            this->getShare(0)->join();
            comm_profiler.accumulate("comm-time");
            this->zero();
            *this += offline_output;
        }
        else if (partyNum == SERVER) {
            comm_profiler.start();
            r.getShare(0)->receive(GFO<T>::otherParty(partyNum));
            r.getShare(0)->join();
            comm_profiler.accumulate("comm-time");
            *this += r;
            *this *= *rhs.getShare(0);
            *this += offline_output;
        }

        func_profiler.add_comm_round();
    }
 
    return *this;
}

template<typename T, typename I>
template<typename I2>
GFOBase<T, I> &GFOBase<T, I>::operator^=(const GFOBase<T, I2> &rhs) {
    *shareA ^= *rhs.getShare(0);
    return *this;
}

template<typename T, typename I>
template<typename I2>
GFOBase<T, I> &GFOBase<T, I>::operator&=(const GFOBase<T, I2> &rhs) {

    size_t size = rhs.size();
    GFO<T> x(size), y(size), z(size);
    PrecomputeObject.getBooleanBeaverTriples<T, GFO<T> >(x, y, z);
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
int GFOBase<T, I>::otherParty(int party) {
	switch(party) {
        case SERVER:
            return CLIENT;
        default: // CLIENT
            return SERVER;
    }	
}

template<typename T, typename I>
int GFOBase<T, I>::numShares() {
    return 1;
}

template<typename T, typename I>
GFO<T, I>::GFO(DeviceData<T, I> *a, T prime_) : GFOBase<T, I>(a, false, prime_) {}

template<typename T>
GFO<T, BufferIterator<T> >::GFO(DeviceData<T> *a, T prime_) :
    GFOBase<T, BufferIterator<T> >(a, false, prime_) {}

template<typename T>
GFO<T, BufferIterator<T> >::GFO(size_t n, T prime_) :
    _shareA(n),
    GFOBase<T, BufferIterator<T> >(&_shareA, false, prime_) {}

template<typename T>
GFO<T, BufferIterator<T> >::GFO(std::initializer_list<double> il, bool convertToFixedPoint, T prime_) :
    _shareA(il.size()),
    GFOBase<T, BufferIterator<T> >(&_shareA, false, prime_) {

    typedef typename std::make_signed<T>::type S;
    std::vector<T> shifted_vals;
    for (double f : il) {
        if (convertToFixedPoint) {
            shifted_vals.push_back(
                (static_cast<T>(f * (1 << FLOAT_PRECISION)) + this->prime) % this->prime
                );
        } else {
            shifted_vals.push_back(((static_cast<T>(f)) + this->prime) % this->prime);
        }
    }

    switch (partyNum) {
        case GFO<T>::SERVER:
            thrust::copy(shifted_vals.begin(), shifted_vals.end(), _shareA.begin());
            break;
        case GFO<T>::CLIENT:
            // nothing
            break;
    }

    *this *= 1;
}

template<typename T>
void GFO<T, BufferIterator<T> >::resize(size_t n) {
    _shareA.resize(n);
}

template<typename T, typename I>
void dividePublic(GFO<T, I> &a, T denominator) {
    int size = a.size();
    size_t prime = a.prime;

    #ifdef HIGH_Q
    // x/d = (r_x-m_x*q)/d. if x<0, then x/d = (r_x-q)/d.
    // qdd = 1{x<0}*(q/d - q).
    DeviceData<T> qdd(size);
    qdd.zero();
    // wrong.
    qdd += *a.getShare(0);
    qdd /= static_cast<T>((prime + 1) / 2);
    qdd *= static_cast<T>(prime / denominator);
    *a.getShare(0) /= denominator;
    *a.getShare(0) -= qdd;
    *a.getShare(0) += prime;
    a %= prime;

    // *a.getShare(0) /= denominator;
    #else
    // TODO: implement GForce native solution.
    
    #endif
}

template<typename T, typename I, typename I2>
void dividePublic(GFO<T, I> &a, DeviceData<T, I2> &denominators) {

    assert(denominators.size() == a.size() && "GFO dividePublic powers size mismatch");

    int size = a.size();

    DeviceData<T> xsign(size);
    DeviceData<T> qdd(size);
    xsign.zero();
    qdd.fill(a.prime);
    // wrong.
    xsign += *a.getShare(0);
    xsign /= (a.prime-1)/2 + 1;
    qdd /= denominators;
    qdd -= a.prime;
    qdd *= xsign;
    *a.getShare(0) /= denominators;
    *a.getShare(0) -= qdd;
}

template<typename T, typename U, typename I, typename I2>
void dividePublic_no_off1(GFO<T, I> &a, T denominator, GFO<T, I2> &result) {

    size_t size = a.size();

    DeviceData<T> d(size);
    d.fill(denominator);
    dividePublic_no_off1<T, U, I, I2>(a, d, result);
}

template<typename T, typename U, typename I, typename I2, typename I3>
void dividePublic_no_off1(GFO<T, I> &a, DeviceData<T, I2> &denominators, GFO<T, I3> &result) {

    assert(denominators.size() == a.size() && "GFO dividePublic powers size mismatch");

    // TODO: int8 or int4 support.
    size_t size = a.size();
    result.zero();
    auto prime = a.prime;

    // step 1:  SERVER samples r and send xs + r to CLIENT.
    //          CLIENT computes z = x + r.
    GFO<T> r(size);
    PrecomputeObject.getRandomNumber<T, GFO<T>>(r);
    if (partyNum == GFO<uint32_t>::SERVER) {
        a += r;
        comm_profiler.start();
        a.getShare(0)->transmit(GFO<T>::otherParty(partyNum));
        a.getShare(0)->join();
        comm_profiler.accumulate("comm-time");
    }
    else if (partyNum == GFO<uint32_t>::CLIENT) {
        comm_profiler.start();
        r.getShare(0)->receive(GFO<T>::otherParty(partyNum));
        r.getShare(0)->join();
        comm_profiler.accumulate("comm-time");
        // compute z.
        r += a;
    }

    // step 2:  SERVER and CLIENT run a millionaire's protocol.
    GFO<T> compare_result(size);

    {
        GFO<T> rmodd(size);
        rmodd.zero();
        rmodd += r;
        *rmodd.getShare(0) %= denominators;

        // step 3: compute <1{rmodd <= zmodd}>_2
        GFO<U> bool_result(size);
        privateCompare<T, U, I, BufferIterator<U>>(rmodd, bool_result);
        thrust::copy(bool_result.getShare(0)->begin(), bool_result.getShare(0)->end(), compare_result.getShare(0)->begin());
    }

    // step 4: wrap.
    if (partyNum == GFO<uint32_t>::SERVER) {
        GFO<T> r2(size);
        r2.offline_known = true;
        r2.zero();
        r2 += r;
        thrust::transform(
            r2.getShare(0)->begin(), r2.getShare(0)->end(),
            thrust::make_constant_iterator((q - 1) / 2),
            r2.getShare(0)->begin(),
            thrust::greater_equal<T>()
        );

        GFO<T> anotherr(size);
        anotherr.zero();
        anotherr *= r2;
        DeviceData<T>* ddq = r2.getShare(0);
        ddq->fill(q);
        *ddq /= denominators;
        anotherr *= *ddq;
        result += anotherr;
    }
    else if (partyNum == GFO<uint32_t>::CLIENT) {
        GFO<T> r2(size);
        r2.offline_known = true;
        r2.zero();
        r2 += r;
        thrust::transform(
            r2.getShare(0)->begin(), r2.getShare(0)->end(),
            thrust::make_constant_iterator((q - 1) / 2),
            r2.getShare(0)->begin(),
            thrust::less<T>()
        );

        GFO<T> anotherr(size);
        anotherr.offline_known = true;
        r2 *= anotherr;
        DeviceData<T>* ddq = anotherr.getShare(0);
        ddq->fill(q);
        *ddq /= denominators;
        r2 *= *ddq;
        result += r2;
    }

    // step 5: the final step.
    *r.getShare(0) /= denominators;
    if (partyNum == GFO<uint32_t>::SERVER) {   
        r *= static_cast<T>(-1);

        DeviceData<T> temp(size);
        temp.fill(0);
        temp += *compare_result.getShare(0);

        // bit2A.
        // placeholder for client's input.
        GFO<T> another_input(size); 
        another_input.fill(0);
        compare_result.offline_known = true;
        compare_result *= static_cast<T>(-2);
        compare_result += 1;

        another_input *= compare_result;
        another_input += temp;
        result -= another_input;
    }
    if (partyNum == GFO<uint32_t>::CLIENT) {
        // bit2A.
        // placeholder for server's input.
        GFO<T> another_input(size); 
        another_input.fill(0);
        another_input.offline_known = true;
        compare_result *= another_input;
        result -= compare_result;
    }
    result += r;

    func_profiler.add_comm_round();
}

/// @brief SERVER and CLIENT run a millionaire's protocool, output
/// <1{x>y}>_2, where the input of SERVER is x and CLIENT y.
/// @tparam T DeviceData datatype.
/// @tparam I DeviceData iterator.
/// @param a The input of SERVER or CLIENT.
template<typename T, typename U, typename I, typename I2>
void privateCompare(GFO<T, I> &input, GFO<U, I2> &result) {
    // TODO: int8 or int4 support.  uint8 √
    // notice: uint8 is enough to hold prexor.
    size_t T_bits_count = PC_BITS;
    size_t size = input.size();
    size_t prime = p;

    // b is used to hold medium result. [[b_{-1}^0, b_0^0, b_1^0, ...], [b_{-1}^1, ...], ...].
    DeviceData<U> b(size * (T_bits_count + 1));
    DeviceData<U> delta(size);
    PrecomputeObject.getCoin<U>(delta);
    b.fill(0), delta.fill(0), result.zero();

    // construct a iterator to catch the last T_bits_count bits of b for each element.
    thrust::counting_iterator<U> count_iter(0);
    // TODO: optimize this with thrust placeholder expressions.
    auto catch_iter = thrust::make_transform_iterator(count_iter, GFO_catch_functor<U>(T_bits_count));
    auto bi_catched_iter = thrust::make_permutation_iterator(b.begin(), catch_iter);
    DeviceData<U, decltype(bi_catched_iter)> bi(bi_catched_iter, bi_catched_iter + size * T_bits_count);

    // MILL step 1: bit expand. because r is known to SERVER, so the bit expand of r is trival.
    // expand to an addition bit 0.
    // TODO: int8 support. 
    gpu::bitexpand(input.getShare(0), &bi, T_bits_count + 1);

    
    if (partyNum == GFO<uint32_t>::SERVER) {

        // MILL step 2: SERVER sample deltas, compute alpha = 1 - 2*deltas. 
        // TODO: randomn number.
        DeviceData<U> alpha(size * (T_bits_count + 1));
        gpu::vectorExpand(&delta, &alpha, T_bits_count + 1);
        auto alpha_catched_iter = thrust::make_permutation_iterator(alpha.begin(), catch_iter);
        DeviceData<U, decltype(alpha_catched_iter)> alpha_catched(alpha_catched_iter, alpha_catched_iter + size * T_bits_count);
        alpha_catched *= static_cast<U>(-2 + prime);
        alpha_catched += prime + 1;
        alpha_catched %= prime;

        // TODO: offline known random mask to bi.
        DeviceData<U> rb(size * (T_bits_count + 1));
        rb.fill(1);

        // MILL step 3: SERVER and CLIENT evaluate xor result together.
        // TODO: initialize by DeviceData*.
        GFO<U> bi_xor(size * T_bits_count, prime);
        bi_xor.fill(0);
        *bi_xor.getShare(0) += bi;
        bi_xor.offline_known = true;  
        GFO<U> client_input(size * T_bits_count, prime);
        client_input.zero();
        client_input *= bi_xor;
        client_input *= static_cast<U>(-2);
        client_input += bi;
        client_input *= 3;

        // prefix_xor holds \phi_i for each element.
        DeviceData<U> prefix_xor(size * (T_bits_count + 1));
        prefix_xor.fill(0);
        // construct a iterator to catch the last T_bits_count bits of prefix_xor for each element.
        auto prefix_catched_iter = thrust::make_permutation_iterator(prefix_xor.begin(), catch_iter);
        thrust::copy(client_input.getShare(0)->begin(), client_input.getShare(0)->end(), prefix_catched_iter);
        // considering the output of bitexpand is small endian, we need to reverse the prefix_xor's order.
        thrust::reverse_iterator<BufferIterator<U>> prefix_reversed_iter(prefix_xor.end());
        // construct a repeat iterator.
        auto key_iter = thrust::make_transform_iterator(count_iter, GFO_key_functor<U>(T_bits_count));
        thrust::exclusive_scan_by_key(key_iter, key_iter + size * (T_bits_count + 1), prefix_reversed_iter, prefix_reversed_iter);
        prefix_xor %= prime;
        b += prefix_xor;  
        b += alpha;  
        b %= prime;


        // MILL step 4: transmission.
        comm_profiler.start();
        b.transmit(GFO<T>::otherParty(partyNum));
        b.join();
        comm_profiler.accumulate("comm-time");

        // MILL step 5: pass.
        thrust::copy(delta.begin(), delta.end(), result.getShare(0)->begin());
    }
    else if (partyNum == GFO<uint32_t>::CLIENT) {

        // MILL step 2: server samples delta.
        // pass.

        // MILL step 3: SERVER and CLIENT evaluate bi together.
        GFO<U> bi_xor(size*T_bits_count, prime);
        bi_xor.fill(0);
        *bi_xor.getShare(0) += bi;
        GFO<U> server_input(size*T_bits_count, prime);
        server_input.fill(0);
        server_input.offline_known = true;
        bi_xor *= server_input;
        bi_xor *= static_cast<U>(-2);
        *bi_xor.getShare(0) += bi;
        bi_xor *= 3;
        
        // prefix_xor holds \phi_i for each element.
        DeviceData<U> prefix_xor(size * (T_bits_count + 1));
        prefix_xor.fill(0);
        // construct a iterator to catch the last T_bits_count bits of prefix_xor for each element.
        auto prefix_catched_iter = thrust::make_permutation_iterator(prefix_xor.begin(), catch_iter);
        thrust::copy(bi_xor.getShare(0)->begin(), bi_xor.getShare(0)->end(), prefix_catched_iter);
        // considering the output of bitexpand is small endian, we need to reverse the prefix_xor's order.
        thrust::reverse_iterator<BufferIterator<U>> prefix_reversed_iter(prefix_xor.end());
        // construct a repeat iterator.
        auto key_iter = thrust::make_transform_iterator(count_iter, GFO_key_functor<U>(T_bits_count));
        thrust::exclusive_scan_by_key(key_iter, key_iter + size * (T_bits_count + 1), prefix_reversed_iter, prefix_reversed_iter);
        prefix_xor %= prime;
        b *= static_cast<U>(prime - 1);
        b += prefix_xor;  
        b %= prime;

        // MILL step 4: recieve.
        DeviceData<U> recvb(size * (T_bits_count + 1));
        comm_profiler.start();
        recvb.receive(GFO<T>::otherParty(partyNum));
        recvb.join();
        comm_profiler.accumulate("comm-time");
        b += recvb;
        b %= prime;

        // MILL step 5: CLIENT check if there is any 0.
        // TODO: combine transform and reduce.
        thrust::transform(b.begin(), b.end(), b.begin(), is_not_a_functor<T>(0));
        thrust::equal_to<U> binary_pred;
        auto binary_op = [prime](U x, U y) -> U {U z = x * y; return z % prime;};
        thrust::reduce_by_key(key_iter, key_iter + size * (T_bits_count + 1), b.begin(), b.begin(), result.getShare(0)->begin(), binary_pred, GFO_mult_and_modular_functor<U>(prime));
    }
    func_profiler.add_comm_round(2);
}

// open a field element 'in' into 'out'. If 'to_fxp' is true, convert 'out' into ring.
template<typename T, typename I, typename I2>
void reconstruct(GFO<T, I> &in, DeviceData<T, I2> &out, bool to_fxp) {

    auto prime = in.prime;
    comm_profiler.start();

    // 1 - send shareA to next party
    in.getShare(0)->transmit(GFO<T>::otherParty(partyNum));

    // 2 - receive shareA from previous party into DeviceBuffer 
    DeviceData<T> rxShare(in.size());
    rxShare.receive(GFO<T>::otherParty(partyNum));

    in.getShare(0)->join();
    rxShare.join();
    comm_profiler.accumulate("comm-time");

    // 3 - result is our shareB + received shareA
    out.zero();
    out += *in.getShare(0);
    out += rxShare;
    out %= prime;

    if (to_fxp) {
        thrust::transform(
            out.begin(), out.end(), out.begin(),
            field_restruct_functor<T>(prime));
    }

    func_profiler.add_comm_round();
}

template<typename T>
void matmul(const GFO<T> &a, const GFO<T> &b, GFO<T> &c,
        int M, int N, int K,
        bool transpose_a, bool transpose_b, bool transpose_c, T truncation) {

    localMatMul(a, b, c, M, N, K, transpose_a, transpose_b, transpose_c);

    // truncate
    dividePublic(c, (T)1 << truncation);
}

/**
 * return b*(y-x) + x. if b=0, return x; else return y. b is a boolean sharing.
*/
// template<typename T, typename U, typename I, typename I2, typename I3, typename I4>
// void selectShare(const GFO<T, I> &x, const GFO<T, I2> &y, const GFO<U, I3> &b, GFO<T, I4> &z) {

//     assert(x.size() == y.size() && x.size() == b.size() && x.size() == z.size() && "GFO selectShare input size mismatch");

//     //TO_BE_DONE
//     GFO<T> c(x.size());
//     GFO<U> cbits(b.size());

//     // b XOR c, then open -> e
//     cbits ^= b;

//     DeviceData<U> e(cbits.size());
//     reconstruct(cbits, e);

//     // d = 1-c if e=1 else d = c       ->        d = (e)(1-c) + (1-e)(c)
//     GFO<T> d(e.size());
//     convex_comb(d, c, e);

//     // z = ((y - x) * d) + x
//     GFO<T> result(x.size());
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
template<typename T, typename I, typename I2, typename I3, typename I4>
void selectShare(const GFO<T, I> &x, const GFO<T, I2> &y, const GFO<T, I3> &b, GFO<T, I4> &z) {

    assert(x.size() == y.size() && x.size() == b.size() && x.size() == z.size() && "GFO selectShare input size mismatch");

    //TO_BE_DONE
    GFO<T> b_T(x.size());
    b_T.zero();
    z.zero();
    z += y;
    z -= x;
    // thrust::copy(b.getShare(0)->begin(), 
    //     b.getShare(0)->end(),
    //     b_T.getShare(0)->begin());
    z *= b;
    z += x;
}

// template<typename T, typename U, typename I, typename I2, typename I3, typename I4>
// void selectShare(const GFO<T, I> &x, const GFO<T, I2> &y, const GFO<U, I3> &b, GFO<T, I4> &z) {

//     assert(x.size() == y.size() && x.size() == b.size() && x.size() == z.size() && "GFO selectShare input size mismatch");

//     //TO_BE_DONE
//     GFO<T> c(x.size());
//     GFO<U> cbits(b.size());

//     // b XOR c, then open -> e
//     cbits ^= b;

//     DeviceData<U> e(cbits.size());
//     reconstruct(cbits, e);

//     // d = 1-c if e=1 else d = c       ->        d = (e)(1-c) + (1-e)(c)
//     GFO<T> d(e.size());
//     convex_comb(d, c, e);

//     // z = ((y - x) * d) + x
//     GFO<T> result(x.size());
//     result += y;
//     result -= x;
//     result *= d;
//     result += x;
    
//     z.zero();
//     z += result;
// }

template<typename T, typename I, typename I2>
void sqrt(GFO<T, I> &in, GFO<T, I2> &out) {
    /*
     * Approximations:
     *   > sqrt(x) = 0.424 + 0.584(x)
     *     sqrt(x) = 0.316 + 0.885(x) - 0.202(x^2)
     */
    taylorSeries(in, out, 0.424, 0.584, 0.0, sqrt_lambda());
}

template<typename T, typename I, typename I2>
void inverse(GFO<T, I> &in, GFO<T, I2> &out) {
    /*
     * Approximations:
     *     1/x = 2.838 - 1.935(x)
     *   > 1/x = 4.245 - 5.857(x) + 2.630(x^2)
     */
    taylorSeries(in, out, 4.245, -5.857, 2.630, inv_lambda());
}

template<typename T, typename I, typename I2>
void sigmoid(GFO<T, I> &in, GFO<T, I2> &out) {
    /*
     * Approximation:
     *   > sigmoid(x) = 0.494286 + 0.275589(x) + -0.038751(x^2)
     */
    taylorSeries(in, out, 0.494286, 0.275589, -0.038751, sigmoid_lambda());
}

template<typename T>
void localFprop(const GFO<T> &A, const GFO<T> &B, GFO<T> &C,
        int batchSize, int imageHeight, int imageWidth, int Din,
        int Dout, int filterHeight, int filterWidth,
        int paddingHeight, int paddingWidth,
        int stride, int dilation) {

    GFO<T> x(A.size()), y(B.size()), z(C.size());
    auto prime = A.prime;
    if (!B.offline_known)
    {
        PrecomputeObject.getConvBeaverTriple_fprop<T, GFO<T> >(x, y, z, 
            batchSize, imageHeight, imageWidth, Din,
            Dout, filterHeight, filterWidth,
            paddingHeight, paddingWidth,
            stride, dilation);
        DeviceData<T> e(x.size()), f(y.size()), temp(z.size());

        x += A; y += B;
        reconstruct(x, e, false); reconstruct(y, f, false);
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
        // SERVER: r^S.     
        // CLIENT: w*r^C-r^S.
        GFO<T> offline_output(C.size());
        // SERVER: 0.   
        // CLIENT: r^C.
        GFO<T> r(A.size());
        PrecomputeObject.getCorrelatedRandomness_fprop<T, GFO<T> >(
            B, offline_output, r,
            batchSize, imageHeight, imageWidth, Din,
            Dout, filterHeight, filterWidth,
            paddingHeight, paddingWidth,
            stride, dilation
        );

        if (partyNum == GFO<uint32_t>::CLIENT) {
            r -= A;
            r *= static_cast<T>(-1);
            comm_profiler.start();
            r.getShare(0)->transmit(GFO<T>::otherParty(partyNum));
            r.getShare(0)->join();
            comm_profiler.accumulate("comm-time");
            C += offline_output;
        }
        else if (partyNum == GFO<uint32_t>::SERVER) {
            r.getShare(0)->receive(GFO<T>::otherParty(partyNum));
            comm_profiler.start();
            r.getShare(0)->join();
            comm_profiler.accumulate("comm-time");
            r += A;
            gpu::conv_fprop(r.getShare(0), B.getShare(0), C.getShare(0), 
                batchSize, imageHeight, imageWidth, Din,
                Dout, filterHeight, filterWidth,
                paddingHeight, paddingWidth,
                stride, dilation);
            C += offline_output;
        }
        cudaThreadSynchronize();

        func_profiler.add_comm_round();
    }
}

template<typename T>
void localDgrad(const GFO<T> &A, const GFO<T> &B, GFO<T> &C,
        int batchSize, int outputHeight, int outputWidth, int Dout,
        int filterHeight, int filterWidth, int Din,
        int paddingHeight, int paddingWidth, int stride, int dilation,
        int imageHeight, int imageWidth) {

    GFO<T> x(A.size()), y(B.size()), z(C.size());
    if (!B.offline_known)
    {
        PrecomputeObject.getConvBeaverTriple_dgrad<T, GFO<T> >(x, y, z, 
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
        // SERVER: r^S.     
        // CLIENT: w*r^C-r^S.
        DeviceData<T> offline_output(C.size());
        offline_output.fill(0);

        // SERVER: x-r^C.   
        // CLIENT: r^C.
        DeviceData<T> r(A.size());
        r.fill(0);
        if (partyNum == GFO<uint32_t>::CLIENT) {
            r -= *A.getShare(0);
            r *= static_cast<T>(-1);
            comm_profiler.start();
            r.transmit(GFO<T>::otherParty(partyNum));
            r.join();
            comm_profiler.accumulate("comm-time");
            *C.getShare(0) += offline_output;
        }
        else if (partyNum == GFO<uint32_t>::SERVER) {
            comm_profiler.start();
            r.receive(GFO<T>::otherParty(partyNum));
            r.join();
            comm_profiler.accumulate("comm-time");
            r += *A.getShare(0);
            DeviceData<T> b_copy(B.size());
            b_copy += *B.getShare(0);
            gpu::conv_dgrad(&r, &b_copy, C.getShare(0), 
                batchSize, outputHeight, outputWidth, Dout,
                filterHeight, filterWidth, Din,
                paddingHeight, paddingWidth, stride, dilation,
                imageHeight, imageWidth);
            *C.getShare(0) += offline_output;
        }
        cudaThreadSynchronize();

        func_profiler.add_comm_round();
    }
}

template<typename T>
void localWgrad(const GFO<T> &A, const GFO<T> &B, GFO<T> &C,
        int batchSize, int outputHeight, int outputWidth, int Dout,
        int imageHeight, int imageWidth, int Din,
        int filterHeight, int filterWidth,
        int paddingHeight, int paddingWidth, int stride, int dilation) {

    GFO<T> x(A.size()), y(B.size()), z(C.size());
    if (!B.offline_known)
    {
        PrecomputeObject.getConvBeaverTriple_wgrad<T, GFO<T> >(x, y, z, 
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
        // SERVER: r^S.     
        // CLIENT: w*r^C-r^S.
        DeviceData<T> offline_output(C.size());
        offline_output.fill(0);

        // SERVER: x-r^C.   
        // CLIENT: r^C.
        DeviceData<T> r(A.size());
        r.fill(0);
        if (partyNum == GFO<uint32_t>::CLIENT) {
            r -= *A.getShare(0);
            r *= static_cast<T>(-1);
            comm_profiler.start();
            r.transmit(GFO<T>::otherParty(partyNum));
            r.join();
            comm_profiler.accumulate("comm-time");
            *C.getShare(0) += offline_output;
        }
        else if (partyNum == GFO<uint32_t>::SERVER) {
            comm_profiler.start();
            r.receive(GFO<T>::otherParty(partyNum));
            r.join();
            comm_profiler.accumulate("comm-time");
            r += *A.getShare(0);
            DeviceData<T> b_copy(B.size());
            b_copy += *B.getShare(0);
            gpu::conv_wgrad(&r, &b_copy, C.getShare(0), 
                batchSize, outputHeight, outputWidth, Dout,
                imageHeight, imageWidth, Din,
                filterHeight, filterWidth,
                paddingHeight, paddingWidth, stride, dilation);
            *C.getShare(0) += offline_output;
        }
        cudaThreadSynchronize();

        func_profiler.add_comm_round();
    }
}

template<typename T>
void convolution(const GFO<T> &A, const GFO<T> &B, GFO<T> &C,
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
                    stride, static_cast<T>(1));
            break;
        case cutlass::conv::Operator::kDgrad:
            localDgrad(A, B, C,
                    batchSize, outputHeight, outputWidth, Dout,
                    filterSize, filterSize, Din,
                    padding, padding, stride, static_cast<T>(1),
                    imageHeight, imageWidth);
            break;
        case cutlass::conv::Operator::kWgrad:
            localWgrad(A, B, C,
                    batchSize, outputHeight, outputWidth, Dout,
                    imageHeight, imageWidth, Din,
                    filterSize, filterSize,
                    padding, padding, stride, (static_cast<T>(1)));
            break;
    }

    // *C.getShare(0) += localResult;
    dividePublic(C, static_cast<T>(1 << truncation));
    C %= q;
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
void dReLU(const GFO<T, I> &input, GFO<T, I2> &result) {

    size_t size = input.size();

    // TODO: how to constrain in input x's range?
    T bound = GFORCE_BOUND;
    GFO<T> input_(size);
    input_.zero();
    input_ += input;
    input_ += bound;
    dividePublic_no_off1<T, U, I, I2>(input_, bound, result);
}
    
template<typename T, typename U, typename I, typename I2, typename I3>
void ReLU(const GFO<T, I> &input, GFO<T, I2> &result, GFO<T, I3> &dresult) {

    //TO_BE_DONE

    func_profiler.start();
    dReLU<T, U, I, I3>(input, dresult);
    func_profiler.accumulate("relu-drelu");

    func_profiler.start();
    // TODO: can we eliminate this copy op?
    thrust::copy(dresult.getShare(0)->begin(), 
        dresult.getShare(0)->end(),
        result.getShare(0)->begin());
    result *= input;
    func_profiler.accumulate("relu-selectshare");
}

template<typename T, typename U, typename I, typename I2, typename I3>
void maxpool(GFO<T, I> &input, GFO<T, I2> &result, GFO<T, I3> &dresult, int k) {

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
    GFO<T, SRIterator> even(&even0);

    StridedRange<I> odd0Range(input.getShare(0)->begin() + offset, input.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> odd0(odd0Range.begin(), odd0Range.end());
    GFO<T, SRIterator> odd(&odd0);
    func_profiler.accumulate("range creation");

    //printf("func-maxpool-post-rangecreate\n");
    //printMemUsage();

    while(k > 2) {

        // -- MP --

        // diff = even - odd
        func_profiler.start();
        GFO<T> diff(even.size());
        diff.zero();
        diff += even;
        diff -= odd;
        func_profiler.accumulate("maxpool-diff");

        //printf("func-maxpool-post-diff-k=%d\n", k);
        //printMemUsage();

        // DRELU diff -> b
        func_profiler.start();
        GFO<T> b(even.size());
        dReLU(diff, b);
        func_profiler.accumulate("maxpool-drelu");

        //printf("func-maxpool-post-drelu-k=%d\n", k);
        //printMemUsage();
            
        // selectShare(odd, even, b, even);
        even -= odd;
        even *= b;
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
        // TODO: potential mistake(wrong). GForce's drelu output is arith sharing.
        func_profiler.start();
        GFO<T> negated(b.size());
        negated.fill(1);
        negated -= b;
        GFO<T> expandedB(input.size());

        // expanded choose bit.
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
    GFO<T> diff(even.size());
    diff.zero();
    diff += even;
    diff -= odd;
    func_profiler.accumulate("maxpool-z-diff");

    // DRELU diff -> b
    func_profiler.start();
    GFO<T> b(even.size());
    dReLU(diff, b);
    func_profiler.accumulate("maxpool-z-drelu");
 
    func_profiler.start();
        
    // selectShare(odd, even, b, even);
    even -= odd;
    even *= b;
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
    GFO<T> negated(b.size());
    negated.fill(1);
    negated -= b;
    GFO<T> expandedB(input.size());
    gpu::expandCompare(*b.getShare(0), *negated.getShare(0), *expandedB.getShare(0));
    func_profiler.accumulate("maxpool-z-expandCompare");
 
    // dresult &= expandedB
    func_profiler.start();
    dresult *= expandedB;
    func_profiler.accumulate("maxpool-z-dcalc");
}

template<typename T>
void localMatMul(const GFO<T> &a, const GFO<T> &b, GFO<T> &c,
        int M, int N, int K,
        bool transpose_a, bool transpose_b, bool transpose_c) {
    
    auto prime = a.prime;

    int a_rows = transpose_a ? K : M; int a_cols = transpose_a ? M : K;
    int b_rows = transpose_b ? N : K; int b_cols = transpose_b ? K : N;
    if (!b.offline_known)
    {
        GFO<T> x(a.size()), y(b.size()), z(c.size());
        PrecomputeObject.getMatrixBeaverTriple<T, GFO<T> >(x, y, z, a_rows, a_cols, b_rows, b_cols, transpose_a, transpose_b, transpose_c);

        DeviceData<T> e(x.size()), f(y.size()), temp(z.size());

        x += a; y += b;
        reconstruct(x, e, false);
        reconstruct(y, f, false);
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
        // SERVER: r^S.     
        // CLIENT: w*r^C-r^S.
        GFO<T> offline_output(c.size());
        // SERVER: 0.   
        // CLIENT: r^C.
        GFO<T> r(a.size());
        PrecomputeObject.getCorrelatedRandomness_matmul<T, GFO<T> >(
            const_cast<GFO<T>& >(b), offline_output, r,
            a_rows, a_cols, b_rows, b_cols,
            transpose_a, transpose_b, transpose_c
        );
        
        if (partyNum == GFO<uint32_t>::CLIENT) {
            r -= a;
            r *= static_cast<T>(-1);
            comm_profiler.start();
            r.getShare(0)->transmit(GFO<T>::otherParty(partyNum));
            r.getShare(0)->join();
            comm_profiler.accumulate("comm-time");
            c += offline_output;
        }
        else if (partyNum == GFO<uint32_t>::SERVER) {
            comm_profiler.start();
            r.getShare(0)->receive(GFO<T>::otherParty(partyNum));
            r.getShare(0)->join();
            comm_profiler.accumulate("comm-time");
            r += a;
            gpu::gemm(M, N, K, r.getShare(0), transpose_a, b.getShare(0), transpose_b, c.getShare(0), transpose_c);
            c += offline_output;
        }

        func_profiler.add_comm_round();
    }
}

template<typename T, typename I, typename I2>
void carryOut(GFO<T, I> &p, GFO<T, I> &g, int k, GFO<T, I2> &out) {

    // get zip iterators on both p and g
    //  -> pEven, pOdd, gEven, gOdd
 
    int stride = 2;
    int offset = 1;

    using SRIterator = typename StridedRange<I>::iterator;

    StridedRange<I> pEven0Range(p.getShare(0)->begin(), p.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> pEven0(pEven0Range.begin(), pEven0Range.end());
    GFO<T, SRIterator> pEven(&pEven0);

    StridedRange<I> pOdd0Range(p.getShare(0)->begin() + offset, p.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> pOdd0(pOdd0Range.begin(), pOdd0Range.end());
    GFO<T, SRIterator> pOdd(&pOdd0);

    StridedRange<I> gEven0Range(g.getShare(0)->begin(), g.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> gEven0(gEven0Range.begin(), gEven0Range.end());
    GFO<T, SRIterator> gEven(&gEven0);

    StridedRange<I> gOdd0Range(g.getShare(0)->begin() + offset, g.getShare(0)->end(), stride);
    DeviceData<T, SRIterator> gOdd0(gOdd0Range.begin(), gOdd0Range.end());
    GFO<T, SRIterator> gOdd(&gOdd0);

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
void getPowers(GFO<T, I> &in, DeviceData<T, I2> &pow) {

    GFO<T> powers(pow.size()); // accumulates largest power yet tested that is less than the input val
    GFO<T> currentPowerBit(in.size()); // current power
    GFO<T> diff(in.size());
    GFO<T> comparisons(in.size());

    for (int bit = 0; bit < sizeof(T) * 8; bit++) {
        currentPowerBit.fill(bit);

        diff.zero();
        diff += in;
        diff -= ((static_cast<T>(1)) << bit);

        comparisons.zero();
        dReLU(diff, comparisons); // 0 -> current power is larger than input val, 1 -> input val is larger than current power

        // 0 -> keep val, 1 -> update to current known largest power less than input
        // TODO: remove copy.
        GFO<T> b(comparisons.size());
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
void taylorSeries(GFO<T, I> &in, GFO<T, I2> &out,
        double a0, double a1, double a2,
        Functor fn) {

    out.zero();
    GFO<T> scratch(out.size());

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

    for (int share = 0; share < GFO<T>::numShares(); share++) {
        thrust::transform(
            out.getShare(share)->begin(), out.getShare(share)->end(), negativePow.begin(), out.getShare(share)->begin(),
            lshift_functor<T>()); 
    }
}

template<typename T, typename U, typename I, typename I2>
void convex_comb(GFO<T, I> &a, GFO<T, I> &c, DeviceData<U, I2> &b) {

    thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(b.begin(), c.getShare(0)->begin(), a.getShare(0)->begin())),
        thrust::make_zip_iterator(thrust::make_tuple(b.end(), c.getShare(0)->end(), a.getShare(0)->end())),
        GFO_convex_comb_functor(partyNum)
    );
}


