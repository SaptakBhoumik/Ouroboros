#pragma once
#include "op.hpp"
#include <cmath>
#include <functional>
#include <omp.h>
#include <tuple>

#include "basic.hpp"
#include "err_func.hpp"
#include "sf_func.hpp"
#include "trig.hpp"
#include "h_trig.hpp"
#ifndef __MIN__COUNT__FOR__THREAD__
#define __MIN__COUNT__FOR__THREAD__ 200000
#endif
namespace Ouroboros{
namespace __Private__Impl__{
//Do not use this functions here
template<auto func,typename tuple, std::size_t ... Is>
__always_inline void __apply_impl(const tuple& t,double* res,size_t idx, std::index_sequence<Is...>){
    res[idx]=func(std::get<Is>(t)[idx]...);
}
template<std::size_t N,auto func, typename tuple,typename Indices = std::make_index_sequence<N>>
__always_inline void __apply(const tuple& t,double* res,size_t idx){
    __apply_impl<func>(t,res,idx,Indices{});
}
}
template<auto func,typename ... Ts>
Tensor transform(const Tensor& t,const Ts&... args){
    constexpr size_t n = sizeof...(Ts)+1;
    auto arg_data = std::make_tuple(t.data(),args.data()...);
    auto shape = t.shape();
    Tensor res(shape);
    size_t count=shape.count();
    double* res_data=res.data();
    if(count<=__MIN__COUNT__FOR__THREAD__){
        for(size_t i=0;i<count;i++){
            __Private__Impl__::__apply<n,func>(arg_data,res_data,i);
        }
    }
    else{
        #pragma omp parallel for
        for(size_t i=0;i<count;i++){
            __Private__Impl__::__apply<n,func>(arg_data,res_data,i);
        }
    }
    return res;
}
template<auto func,typename ... Ts>
Tensor transform(size_t min_count,const Tensor& t,const Ts&... args){
    constexpr size_t n = sizeof...(Ts)+1;
    auto arg_data = std::make_tuple(t.data(),args.data()...);
    auto shape = t.shape();
    Tensor res(shape);
    size_t count=shape.count();
    double* res_data=res.data();
    if(count<=min_count){
        for(size_t i=0;i<count;i++){
            __Private__Impl__::__apply<n,func>(arg_data,res_data,i);
        }
    }
    else{
        #pragma omp parallel for
        for(size_t i=0;i<count;i++){
            __Private__Impl__::__apply<n,func>(arg_data,res_data,i);
        }
    }
    return res;
}
/*
//TODO:
cumprod(...): Compute the cumulative product of the tensor x along axis.

cumsum(...): Compute the cumulative sum of the tensor x along axis.

cumulative_logsumexp(...): Compute the cumulative log-sum-exp of the tensor x along axis.

*/
}