#pragma once
#include "op.hpp"
#include <omp.h>
#include <tuple>
#include <cmath>

#include "basic.hpp"
#include "err_func.hpp"
#include "sf_func1.hpp"
#include "sf_func2.hpp"
#include "trig.hpp"
#include "h_trig.hpp"
#include "../macros.hpp"

namespace Ouroboros{
namespace __Private__Impl__{
//Do not use this functions here
template<typename T,typename tuple, std::size_t ... Is>
__always_inline void __apply_impl(const T& func,const tuple& t,double* res,size_t idx, std::index_sequence<Is...>){
    res[idx]=func(std::get<Is>(t)[idx]...);
}
template<std::size_t N,typename T, typename tuple,typename Indices = std::make_index_sequence<N>>
__always_inline void __apply(const T& func,const tuple& t,double* res,size_t idx){
    __apply_impl(func,t,res,idx,Indices{});
}
}
template<typename T,typename ... Ts>
__always_inline Tensor transform(const T& func,const Tensor& t,const Ts&... args){
    constexpr size_t n = sizeof...(Ts)+1;
    auto arg_data = std::make_tuple(t.data(),args.data()...);
    auto shape = t.shape();
    Tensor res(shape);
    size_t count=shape.count();
    double* res_data=res.data();
    if(count<=__MIN__COUNT__FOR__THREAD__){
        for(size_t i=0;i<count;i++){
            __Private__Impl__::__apply<n,T>(func,arg_data,res_data,i);
        }
    }
    else{
        #pragma omp parallel for
        for(size_t i=0;i<count;i++){
            __Private__Impl__::__apply<n,T>(func,arg_data,res_data,i);
        }
    }
    return res;
}
template<typename T,typename ... Ts>
__always_inline Tensor transform(const T& func,size_t min_count,const Tensor& t,const Ts&... args){
    constexpr size_t n = sizeof...(Ts)+1;
    auto arg_data = std::make_tuple(t.data(),args.data()...);
    auto shape = t.shape();
    Tensor res(shape);
    size_t count=shape.count();
    double* res_data=res.data();
    if(count<=min_count){
        for(size_t i=0;i<count;i++){
            __Private__Impl__::__apply<n,T>(func,arg_data,res_data,i);
        }
    }
    else{
        #pragma omp parallel for
        for(size_t i=0;i<count;i++){
            __Private__Impl__::__apply<n,T>(func,arg_data,res_data,i);
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