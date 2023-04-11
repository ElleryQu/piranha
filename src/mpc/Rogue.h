/*
 * ROG.h
 */

#pragma once

#include <cstddef>
#include <initializer_list>

#include <cutlass/conv/convolution.h>

#include "../gpu/DeviceData.h"
#include "../globals.h"

#include "../mpc/MTPC.h"
#include "../mpc/TPC.h"
// #include "../mpc/GForce.h"

template <typename T, typename I>
class ROGBase {

    protected:
        
        ROGBase(DeviceData<T, I> *a, bool offline_known=false);

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

        static const std::string& getProt();

        ROGBase<T, I> &operator+=(const T rhs);
        ROGBase<T, I> &operator-=(const T rhs);
        ROGBase<T, I> &operator*=(const T rhs);
        ROGBase<T, I> &operator%=(const T rhs);
        ROGBase<T, I> &operator>>=(const T rhs);
        ROGBase<T, I> &operator^=(const T rhs);
        ROGBase<T, I> &operator&=(const T rhs);

        template<typename I2>
        ROGBase<T, I> &operator+=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        ROGBase<T, I> &operator-=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        ROGBase<T, I> &operator*=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        ROGBase<T, I> &operator^=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        ROGBase<T, I> &operator&=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        ROGBase<T, I> &operator>>=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        ROGBase<T, I> &operator<<=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        ROGBase<T, I> &operator+=(const ROGBase<T, I2> &rhs);
        template<typename I2>
        ROGBase<T, I> &operator-=(const ROGBase<T, I2> &rhs);
        template<typename I2>
        ROGBase<T, I> &operator*=(const ROGBase<T, I2> &rhs);
        template<typename I2>
        ROGBase<T, I> &operator^=(const ROGBase<T, I2> &rhs);
        template<typename I2>
        ROGBase<T, I> &operator&=(const ROGBase<T, I2> &rhs);

    protected:
        
        DeviceData<T, I> *shareA;
};

template<typename T, typename I = BufferIterator<T> >
class ROG : public ROGBase<T, I> {

    public:

        ROG(DeviceData<T, I> *a);
};

template<typename T>
class ROG<T, BufferIterator<T> > : public ROGBase<T, BufferIterator<T> > {

    public:

        ROG(DeviceData<T> *a);
        ROG(size_t n);
        ROG(std::initializer_list<double> il, bool convertToFixedPoint = true);

        void resize(size_t n);

    private:

        DeviceData<T> _shareA;
};

// Functionality

template<typename T, typename I>
void dividePublic(ROG<T, I> &a, T denominator);

template<typename T, typename I, typename I2>
void dividePublic(ROG<T, I> &a, DeviceData<T, I2> &denominators);

template<typename T, typename U, typename I, typename I2>
void dividePublic_no_off1(ROG<T, I> &a, T denominator, ROG<U, I2> &result);

template<typename T, typename U, typename I, typename I2, typename I3>
void dividePublic_no_off1(ROG<T, I> &a, DeviceData<T, I2> &denominators, ROG<U, I3> &result);

template<typename T, typename U, typename I, typename I2>
void privateCompare(ROG<T, I> &input, ROG<U, I2> &result);

template<typename T, typename U, typename I, typename I2, typename I3>
void FusionMux(MTPC<T, I> &x, ROG<U, I2> &b, ROG<T, I3> &result);

template<typename T, typename I, typename I2>
void reconstruct(ROG<T, I> &in, DeviceData<T, I2> &out);

template<typename T>
void matmul(const ROG<T> &a, const ROG<T> &b, ROG<T> &c,
        int M, int N, int K,
        bool transpose_a, bool transpose_b, bool transpose_c, T truncation);

template<typename T, typename U, typename I, typename I2, typename I3, typename I4>
void selectShare(const ROG<T, I> &x, const ROG<T, I2> &y, const ROG<U, I3> &b, ROG<T, I4> &z);

template<typename T, typename I, typename I2>
void sqrt(ROG<T, I> &in, ROG<T, I2> &out);

template<typename T, typename I, typename I2>
void inverse(ROG<T, I> &in, ROG<T, I2> &out);

template<typename T, typename I, typename I2>
void sigmoid(ROG<T, I> &in, ROG<T, I2> &out);

template<typename T>
void convolution(const ROG<T> &A, const ROG<T> &B, ROG<T> &C,
        cutlass::conv::Operator op,
        int batchSize, int imageHeight, int imageWidth, int filterSize,
        int Din, int Dout, int stride, int padding, int truncation);

// TODO change into 2 arguments with subtraction, pointer NULL indicates compare w/ 0
template<typename T, typename U, typename I, typename I2>
void dReLU(const ROG<T, I> &input, ROG<U, I2> &result);
 
template<typename T, typename U, typename I, typename I2, typename I3>
void ReLU(const ROG<T, I> &input, ROG<T, I2> &result, ROG<U, I3> &dresult);

template<typename T, typename U, typename I, typename I2, typename I3>
void maxpool(ROG<T, I> &input, ROG<T, I2> &result, ROG<U, I3> &dresult, int k);

template<typename T, typename I, typename I2>
void reshare(const ROG<T,I> &in, MTPC<T, I2> &out);

// static int p = 257;

#include "Rogue.inl"

