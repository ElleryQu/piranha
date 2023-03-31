/*
 * Mask sharing in 2PC.
 */

#pragma once

#include <cstddef>
#include <initializer_list>

#include <cutlass/conv/convolution.h>

#include "../gpu/DeviceData.h"
#include "../globals.h"

#include "../mpc/TPC.h"

template <typename T, typename I>
class MTPCBase {

    protected:
        
        MTPCBase(DeviceData<T, I> *a, DeviceData<T, I> *b, bool offline_known=false);

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

        MTPCBase<T, I> &operator+=(const T rhs);
        MTPCBase<T, I> &operator-=(const T rhs);
        MTPCBase<T, I> &operator*=(const T rhs);
        MTPCBase<T, I> &operator%=(const T rhs);
        MTPCBase<T, I> &operator>>=(const T rhs);

        template<typename I2>
        MTPCBase<T, I> &operator+=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        MTPCBase<T, I> &operator-=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        MTPCBase<T, I> &operator*=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        MTPCBase<T, I> &operator^=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        MTPCBase<T, I> &operator&=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        MTPCBase<T, I> &operator>>=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        MTPCBase<T, I> &operator<<=(const DeviceData<T, I2> &rhs);
        template<typename I2>
        MTPCBase<T, I> &operator+=(const MTPCBase<T, I2> &rhs);
        template<typename I2>
        MTPCBase<T, I> &operator-=(const MTPCBase<T, I2> &rhs);
        template<typename I2>
        MTPCBase<T, I> &operator*=(const MTPCBase<T, I2> &rhs);
        template<typename I2>
        MTPCBase<T, I> &operator^=(const MTPCBase<T, I2> &rhs);
        template<typename I2>
        MTPCBase<T, I> &operator&=(const MTPCBase<T, I2> &rhs);

    protected:
        
        DeviceData<T, I> *shareD;
        DeviceData<T, I> *shared;
};

template<typename T, typename I = BufferIterator<T> >
class MTPC : public MTPCBase<T, I> {

    public:

        MTPC(DeviceData<T, I> *a);
    
    private:

        DeviceData<T> _shared;
};

template<typename T>
class MTPC<T, BufferIterator<T> > : public MTPCBase<T, BufferIterator<T> > {

    public:

        MTPC(DeviceData<T> *a);
        MTPC(size_t n);
        MTPC(std::initializer_list<double> il, bool convertToFixedPoint = true);

        void resize(size_t n);

    private:

        DeviceData<T> _shareD;
        DeviceData<T> _shared;
};

#include "MTPC.inl"

