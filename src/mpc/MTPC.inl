/*
 * MTPC.inl
 */

#pragma once

#include "MTPC.h"

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


// MTPC class implementation 

template<typename T, typename I>
MTPCBase<T, I>::MTPCBase(DeviceData<T, I> *a, DeviceData<T, I> *b, bool offline_known) : 
                shareD(a), shared(b), offline_known(offline_known) {}

template<typename T, typename I>
void MTPCBase<T, I>::set(DeviceData<T, I> *a) {
    *shareD += a;
}

template<typename T, typename I>
size_t MTPCBase<T, I>::size() const {
    return shareD->size();
}

template<typename T, typename I>
void MTPCBase<T, I>::zero() {
    shareD->zero();
    shared->zero();
};

template<typename T, typename I>
void MTPCBase<T, I>::fill(T val) {
    shareD->zero();
    *shareD += *shared;
    *shareD += val;
}

/// @brief Given an array(double) in CPU, embedding it on GPU.
/// @tparam T 
/// @tparam I 
/// @param v 
template<typename T, typename I>
void MTPCBase<T, I>::setPublic(std::vector<double> &v) {
    std::vector<T> shifted_vals;
    for (double f : v) {
        shifted_vals.push_back((T) (f * (1 << FLOAT_PRECISION)));
    }

    switch (partyNum) {
        case SERVER:
            thrust::copy(shifted_vals.begin(), shifted_vals.end(), shareD->begin());
            *shareD += *shared;
            comm_profiler.start();
            shareD->transmit(MTPC<T>::otherParty(partyNum));
            shareD->join();
            comm_profiler.accumulate("comm-time");
            break;
        case CLIENT:
            comm_profiler.start();
            shareD->receive(MTPC<T>::otherParty(partyNum));
            shareD->join();
            comm_profiler.accumulate("comm-time");
            break;
    }
};

template<typename T, typename I>
DeviceData<T, I> *MTPCBase<T, I>::getShare(int i) {
    switch (i) {
        case 0:
            return shareD;
        case 1:
            return shared;
        default:
            return nullptr;
    }
}

template<typename T, typename I>
const DeviceData<T, I> *MTPCBase<T, I>::getShare(int i) const {
    switch (i) {
        case 0:
            return shareD;
        case 1:
            return shared;
        default:
            return nullptr;
    }
}

template<typename T, typename I>
const std::string& MTPCBase<T, I>::getProt() {
    const static std::string prot = "MTPC";
    return prot;
}

template<typename T, typename I>
MTPCBase<T, I> &MTPCBase<T, I>::operator+=(const T rhs) {
    *shareD += rhs;
    return *this;
}

template<typename T, typename I>
MTPCBase<T, I> &MTPCBase<T, I>::operator-=(const T rhs) {
    *shareD -= rhs;
    return *this;
}

template<typename T, typename I>
MTPCBase<T, I> &MTPCBase<T, I>::operator*=(const T rhs) {
    *shareD *= rhs;
    *shared *= rhs;
    return *this;
}

template<typename T, typename I>
MTPCBase<T, I> &MTPCBase<T, I>::operator>>=(const T rhs) {
    *shareD >>= rhs;
    *shared >>= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
MTPCBase<T, I> &MTPCBase<T, I>::operator+=(const DeviceData<T, I2> &rhs) {
    *shareD += rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
MTPCBase<T, I> &MTPCBase<T, I>::operator-=(const DeviceData<T, I2> &rhs) {
    *shareD -= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
MTPCBase<T, I> &MTPCBase<T, I>::operator*=(const DeviceData<T, I2> &rhs) {
    *shareD *= rhs;
    *shared *= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
MTPCBase<T, I> &MTPCBase<T, I>::operator^=(const DeviceData<T, I2> &rhs) {
    *shareD ^= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
MTPCBase<T, I> &MTPCBase<T, I>::operator&=(const DeviceData<T, I2> &rhs) {
    *shareD &= rhs;
    *shared &= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
MTPCBase<T, I> &MTPCBase<T, I>::operator>>=(const DeviceData<T, I2> &rhs) {
    *shareD >>= rhs;
    *shared >>= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
MTPCBase<T, I> &MTPCBase<T, I>::operator<<=(const DeviceData<T, I2> &rhs) {
    *shareD <<= rhs;
    *shared <<= rhs;
    return *this;
}

template<typename T, typename I>
template<typename I2>
MTPCBase<T, I> &MTPCBase<T, I>::operator+=(const MTPCBase<T, I2> &rhs) {
    *shareD += *rhs.getShare(0);
    *shared += *rhs.getShare(1);
    return *this;
}

template<typename T, typename I>
template<typename I2>
MTPCBase<T, I> &MTPCBase<T, I>::operator-=(const MTPCBase<T, I2> &rhs) {
    *shareD -= *rhs.getShare(0);
    *shared -= *rhs.getShare(1);
    return *this;
}

template<typename T, typename I>
template<typename I2>
MTPCBase<T, I> &MTPCBase<T, I>::operator*=(const MTPCBase<T, I2> &rhs) {

    size_t size = rhs.size();

    // Precomputation.
    TPC<T> dxy(size), dz(size);
    dxy.fill(0), dz.fill(0);
    // MTPCPrecomputeObject.getMT<T>(this, rhs, dxy);
    // MTPCPrecomputeObject.getNewd<T>(dz);

    DeviceData<T> Dy(size);
    Dy.zero();
    Dy += *rhs.getShare(0);
    Dy *= partyNum;
    Dy -= *rhs.getShare(1);
    *this->getShare(0) *= Dy;

    *this->getShare(1) *= static_cast<T>(-1);
    *this->getShare(1) *= *rhs.getShare(0);

    *this->getShare(0) += *this->getShare(1);
    *this->getShare(0) += *dxy.getShare(0);
    *this->getShare(0) += *dz.getShare(0);

    comm_profiler.start();
    this->getShare(0)->transmit(MTPC<T>::otherParty(partyNum));
    this->getShare(1)->receive(MTPC<T>::otherParty(partyNum));
    this->getShare(0)->join();
    this->getShare(1)->join();
    comm_profiler.accumulate("comm-time");

    *this->getShare(0) += *this->getShare(1);
    this->getShare(1)->zero();
    *this->getShare(1) += *dz.getShare(0);
     
    return *this;
}

template<typename T, typename I>
template<typename I2>
MTPCBase<T, I> &MTPCBase<T, I>::operator^=(const MTPCBase<T, I2> &rhs) {
    *shared ^= *rhs.getShare(0);
    return *this;
}

template<typename T, typename I>
template<typename I2>
MTPCBase<T, I> &MTPCBase<T, I>::operator&=(const MTPCBase<T, I2> &rhs) {

    size_t size = rhs.size();

    // Precomputation.
    TPC<T> dxy(size), dz(size);
    dxy.fill(0), dz.fill(0);
    // MTPCPrecomputeObject.getMT<T>(this, rhs, dxy);
    // MTPCPrecomputeObject.getNewd<T>(dz);

    DeviceData<T> Dy(size);
    Dy.zero();
    Dy ^= *rhs.getShare(0);
    Dy &= partyNum;
    Dy ^= *rhs.getShare(1);
    *this->getShare(0) &= Dy;

    *this->getShare(1) &= static_cast<T>(-1);
    *this->getShare(1) &= *rhs.getShare(0);

    *this->getShare(0) ^= *this->getShare(1);
    *this->getShare(0) ^= *dxy.getShare(0);
    *this->getShare(0) ^= *dz.getShare(0);

    this->getShare(0)->transmit(MTPC<T>::otherParty(partyNum));
    this->getShare(1)->receive(MTPC<T>::otherParty(partyNum));
    this->getShare(0)->join();
    this->getShare(1)->join();

    *this->getShare(0) ^= *this->getShare(1);
    this->getShare(1)->zero();
    *this->getShare(1) ^= *dz.getShare(0);
 
    return *this;
}

//TO_BE_DONE
template<typename T, typename I>
int MTPCBase<T, I>::otherParty(int party) {
	switch(party) {
        case SERVER:
            return CLIENT;
        default: // CLIENT
            return SERVER;
    }	
}

template<typename T, typename I>
int MTPCBase<T, I>::numShares() {
    return 2;
}

template<typename T, typename I>
MTPC<T, I>::MTPC(DeviceData<T, I> *a) : 
    _shared(a->size()), MTPCBase<T, I>(a, &_shared) {

    // TODO: offline
    TPC<T> d(&_shared);
    d.zero();
    // MTPCPrecomputeObject.getNewd(d);

    *a += _shared;
}

template<typename T>
MTPC<T, BufferIterator<T> >::MTPC(DeviceData<T> *a) :
    MTPCBase<T, BufferIterator<T> >(a, &_shared) {
    
    // TODO: offline
    TPC<T> d(&_shared);
    d.zero();
    // MTPCPrecomputeObject.getNewd(d);

    *this->getShare(0) += _shared;
}

template<typename T>
MTPC<T, BufferIterator<T> >::MTPC(size_t n) :
    _shared(n), _shareD(n),
    MTPCBase<T, BufferIterator<T> >(&_shareD, &_shared) {

    // TODO: offline
    TPC<T> d(&_shared);
    d.zero();
    // MTPCPrecomputeObject.getNewd(d);
}

template<typename T>
MTPC<T, BufferIterator<T> >::MTPC(std::initializer_list<double> il, bool convertToFixedPoint) :
    _shared(il.size()), _shareD(il.size()),
    MTPCBase<T, BufferIterator<T> >(&_shareD, &_shared) {
    // TODO: offline
    TPC<T> d(&_shared);
    d.zero();
    // MTPCPrecomputeObject.getNewd(d);

    std::vector<T> shifted_vals;
    for (double f : il) {
        if (convertToFixedPoint) {
            shifted_vals.push_back((T) (f * (1 << FLOAT_PRECISION)));
        } else {
            shifted_vals.push_back((T) f);
        }
    }

    thrust::copy(shifted_vals.begin(), shifted_vals.end(), _shareD.begin());
    _shareD += _shared;
}

template<typename T>
void MTPC<T, BufferIterator<T> >::resize(size_t n) {
    this->getShare(0)->resize(n);
    this->getShare(1)->resize(n);
}
