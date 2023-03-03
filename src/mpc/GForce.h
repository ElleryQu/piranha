/*
 * GFO.h
 */

#pragma once

#include <cstddef>
#include <initializer_list>

#include <cutlass/conv/convolution.h>

#include "../gpu/DeviceData.h"
#include "../globals.h"

template <typename T, typename I>
class GFOBase {

    protected:
        
        GFOBase(DeviceData<T, I> *a, bool offline_known=false);

    public:

        enum Party { SERVER, CLIENT };
        static const int numParties = 2;

        void set(DeviceData<T, I> *a);
        size_t size() const;
        void zero();
        void fill(T val);
        void setPublic(std::vector<double> &v);
        DeviceData<T, I> *getShare(int i);
        const DeviceData<T, I> *getShare(int i) const;
        static int numShares();
        static int otherParty(int party);
        typedef T share_type;
        typedef I iterator_type;
        bool offline_known;

        GFOBase<T, I> &operator+=(const T rhs);
        GFOBase<T, I> &operator-=(const T rhs);
        GFOBase<T, I> &operator*=(const T rhs);
        GFOBase<T, I> &operator%=(const T rhs);
        GFOBase<T, I> &operator>>=(const T rhs);

        template<typename I2>
        GFOBase<T, I> &operator+=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        GFOBase<T, I> &operator-=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        GFOBase<T, I> &operator*=(const DeviceData<T, I2> &rhs);        
        template<typename I2>
        GFOBase<T, I> &operator^=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        GFOBase<T, I> &operator&=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        GFOBase<T, I> &operator>>=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        GFOBase<T, I> &operator<<=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        GFOBase<T, I> &operator+=(const GFOBase<T, I2> &rhs);
        template<typename I2>
        GFOBase<T, I> &operator-=(const GFOBase<T, I2> &rhs);
        template<typename I2>
        GFOBase<T, I> &operator*=(const GFOBase<T, I2> &rhs);
        template<typename I2>
        GFOBase<T, I> &operator^=(const GFOBase<T, I2> &rhs);
        template<typename I2>
        GFOBase<T, I> &operator&=(const GFOBase<T, I2> &rhs);

    protected:
        
        DeviceData<T, I> *shareA;
};

template<typename T, typename I = BufferIterator<T> >
class GFO : public GFOBase<T, I> {

    public:

        GFO(DeviceData<T, I> *a);
};

template<typename T>
class GFO<T, BufferIterator<T> > : public GFOBase<T, BufferIterator<T> > {

    public:

        GFO(DeviceData<T> *a);
        GFO(size_t n);
        GFO(std::initializer_list<double> il, bool convertToFixedPoint = true);

        void resize(size_t n);

    private:

        DeviceData<T> _shareA;
};

// Functionality

template<typename T, typename I>
void dividePublic(GFO<T, I> &a, T denominator);

template<typename T, typename I, typename I2>
void dividePublic(GFO<T, I> &a, DeviceData<T, I2> &denominators);

template<typename T, typename U, typename I, typename I2>
void dividePublic_no_off1(GFO<T, I> &a, T denominator, GFO<U, I2> &result);

template<typename T, typename U, typename I, typename I2, typename I3>
void dividePublic_no_off1(GFO<T, I> &a, DeviceData<T, I2> &denominators, GFO<U, I3> &result);

template<typename T, typename U, typename I, typename I2>
void privateCompare(GFO<T, I> &input, GFO<U, I2> &result);

template<typename T, typename I, typename I2>
void reconstruct(GFO<T, I> &in, DeviceData<T, I2> &out);

template<typename T>
void matmul(const GFO<T> &a, const GFO<T> &b, GFO<T> &c,
        int M, int N, int K,
        bool transpose_a, bool transpose_b, bool transpose_c, T truncation);

template<typename T, typename U, typename I, typename I2, typename I3, typename I4>
void selectShare(const GFO<T, I> &x, const GFO<T, I2> &y, const GFO<U, I3> &b, GFO<T, I4> &z);

template<typename T, typename I, typename I2>
void sqrt(GFO<T, I> &in, GFO<T, I2> &out);

template<typename T, typename I, typename I2>
void inverse(GFO<T, I> &in, GFO<T, I2> &out);

template<typename T, typename I, typename I2>
void sigmoid(GFO<T, I> &in, GFO<T, I2> &out);

template<typename T>
void convolution(const GFO<T> &A, const GFO<T> &B, GFO<T> &C,
        cutlass::conv::Operator op,
        int batchSize, int imageHeight, int imageWidth, int filterSize,
        int Din, int Dout, int stride, int padding, int truncation);

// TODO change into 2 arguments with subtraction, pointer NULL indicates compare w/ 0
template<typename T, typename U, typename I, typename I2>
void dReLU(const GFO<T, I> &input, GFO<U, I2> &result);
 
template<typename T, typename U, typename I, typename I2, typename I3>
void ReLU(const GFO<T, I> &input, GFO<T, I2> &result, GFO<U, I3> &dresult);

template<typename T, typename U, typename I, typename I2, typename I3>
void maxpool(GFO<T, I> &input, GFO<T, I2> &result, GFO<U, I3> &dresult, int k);

static int p = 257;
// 23 bit.
// static int q = 7340033;
// 60 bit.
static uint64_t q = 1081244548441771579;

#include "GForce.inl"

