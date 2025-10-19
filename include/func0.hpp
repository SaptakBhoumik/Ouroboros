//Stuff from https://github.com/SaptakBhoumik/Ouroboros/blob/master/include/func.hpp
//Various types of norm
//Min,max,norm etc
#pragma once
#include "tensor.hpp"
#include <sys/cdefs.h>
#include <unordered_set>
#define __Ouroboros__Private__
#include "private_impl.hpp"//Has to be declared here
#undef __Ouroboros__Private__
namespace Ouroboros{
template<const auto func,
        typename T,
        typename ... Ts>
__always_inline auto transform(const T& t,const Ts& ... args){
    using R = decltype(func(t[0], args[0]...));
    Tensor<R> result(t.shape());

    for(std::size_t i=0;i<t.size();i++){
        result[i]=func(t[i],args[i]...);
    }
    return result;
}
template<const auto func,
        typename T,
        typename ... Ts>
__always_inline auto reduce(std::size_t axis,const T& t,const Ts& ... args){
    //Reduce along the given axis
    using R = decltype(func(t[0], args[0]...));
    Shape new_shape=t.shape();
    new_shape.set(axis,1);

    Tensor<R> result(new_shape);

    NDRange ndr(t.shape(),{axis});
    for(auto it0=ndr.begin();it0!=ndr.end();++it0){
        result[*(*it0).begin()]=func(it0, t, args...);
    }
    return result;
}
template<const auto func,
        typename T,
        typename ... Ts>
__always_inline auto reduce(std::set<size_t> axis,const T& t,const Ts& ... args){
    //Reduce along the given axis
    using R = decltype(func(t[0], args[0]...));
    Shape new_shape=t.shape();
    for(auto ax:axis){
        new_shape.set(ax,1);
    }

    Tensor<R> result(new_shape);

    NDRange ndr(t.shape(),axis);
    for(auto it0=ndr.begin();it0!=ndr.end();++it0){
        result[*(*it0).begin()]=func(it0, t, args...);
    }
    return result;
}

template<const auto func,
        typename T>
__always_inline auto accumulate(const T& t,std::size_t axis=0,double initial = 0){
    //Accumulate along the given axis
    using R = decltype(func(t[0], initial));
    Tensor<R> result(t.shape());

    NDRange ndr(t.shape(),{axis});
    for(auto it0=ndr.begin();it0!=ndr.end();++it0){
        R acc=initial;
        for(auto it1=(*it0).begin();it1!=(*it0).end();++it1){
            acc=func(acc,t[*it1]);
            result[*it1] = acc;
        }
    }
    return result;
}

//Outer,at,broadcast,concat,flip,transpose
}