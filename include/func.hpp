#pragma once
#include "op.hpp"
#include <omp.h>
#include <tuple>
#include <cmath>
#include <type_traits>
#include "utils.hpp"
#include "macros.hpp"
#define __Ouroboros__ 
#include "private_impl.hpp"
#undef __Ouroboros__
#define PERMUTE(A,B,C,D) \
{\
    size_t j=0;\
    while (true) {\
        size_t offset=0;\
        for (size_t i = 0; i < B.size(); ++i) {\
            offset+=B[i]*C[i];\
        }\
        D[j++]=offset;\
        /*Find the rightmost index that can be incremented*/\
        int64_t k = (int64_t)A.size() - 1;\
        while (k >= 0 && B[k] == A[k] - 1){\
            k--;\
        }\
        /*If no such index exists, we are done*/\
        if (k < 0) {\
            break;\
        }\
        /*Increment the current index and reset all subsequent indices*/\
        B[k]++;\
        for (size_t i = k + 1; i < A.size(); ++i) {\
            B[i] = 0;\
        }\
    }\
}

namespace Ouroboros{
template<const auto func,
        typename T,
        size_t thread_c=8,
        size_t min_count=__MIN__COUNT__FOR__THREAD__,
        typename ... Ts>
__always_inline typename __Private__Impl__::Typer<std::is_same<__Private__Impl__::return_type_t<decltype(func)>, bool>{}>::Type
         transform(const T& t,const Ts&... args){
    static_assert(std::is_same<__Private__Impl__::return_type_t<decltype(func)>, bool>{} ||
                  std::is_same<__Private__Impl__::return_type_t<decltype(func)>, double>{}, 
                    "Function must return double or bool");
    if constexpr(std::is_same<__Private__Impl__::return_type_t<decltype(func)>, bool>{}){
        return __Private__Impl__::bool_transform<func,T,thread_c,min_count>(t,args...);
    }
    else if constexpr(std::is_same<__Private__Impl__::return_type_t<decltype(func)>, double>{}){
        return __Private__Impl__::double_transform<func,T,thread_c,min_count>(t,args...);
    }
}
template<double(*func)(Utils::Iterator<double>),size_t thread_c=8,size_t min_count=__MIN__COUNT__FOR__THREAD__>
__always_inline Tensor reduce(const Tensor& t,size_t axis=0){
    if(axis>=t.shape().dim()){
        throw std::invalid_argument("Invalid axis");
    }
    Shape input_shape=t.shape();
    Shape input_strides=t.strides();
    Shape output_shape=input_shape;
    output_shape[axis]=1;
    const double* data=t.data();
    if(input_shape.dim()==1){
        return Tensor({1},func(Utils::Iterator<double>(data,input_shape[0])));
    }
    std::vector<size_t> A;
    std::vector<size_t> B(input_shape.dim()-1, 0);
    std::vector<size_t> C;
    A.reserve(input_shape.dim()-1);
    C.reserve(input_shape.dim()-1);
    size_t count=1;
    for(size_t i=0;i<input_shape.dim();i++){
        if(i!=axis){
            count*=input_shape[i];
            A.emplace_back(input_shape[i]);
            C.emplace_back(input_strides[i]);
        }
    }
    size_t* offsets=new size_t[count];
    PERMUTE(A,B,C,offsets);
    Tensor res(output_shape);
    double* res_data=res.data();
    size_t sh=input_shape[axis];
    size_t step=input_strides[axis];
    if(count<=min_count){
        for(size_t i=0;i<count;i++){
            size_t off=offsets[i];
            res_data[i]=func(Utils::Iterator<double>(data+off,sh,step));
        }
    }
    else{
        #pragma omp parallel for num_threads(thread_c)
        for(size_t i=0;i<count;i++){
            size_t off=offsets[i];
            res_data[i]=func(Utils::Iterator<double>(data+off,sh,step));
        }
    }
    delete[] offsets;
    return res;
}
template<double(*func)(double,double),size_t thread_c=8,size_t min_count=__MIN__COUNT__FOR__THREAD__>
__always_inline Tensor accumulate(const Tensor& t,size_t axis=0,double initial = 0){
    if(axis>=t.shape().dim()){
        throw std::invalid_argument("Invalid axis");
    }
    Shape shape=t.shape();
    Shape strides=t.strides();
    const double* data=t.data();
    if(shape.dim()==1){
        Tensor res(shape);
        double* res_data=res.data();
        res_data[0]=func(initial,data[0]);
        for(size_t i=1;i<shape[0];i++){
            res_data[i]=func(res_data[i-1],data[i]);
        }
        return res;
    }
    std::vector<size_t> A;
    std::vector<size_t> B(shape.dim()-1, 0);
    std::vector<size_t> C;
    A.reserve(shape.dim()-1);
    C.reserve(shape.dim()-1);
    size_t count=1;
    for(size_t i=0;i<shape.dim();i++){
        if(i!=axis){
            count*=shape[i];
            A.emplace_back(shape[i]);
            C.emplace_back(strides[i]);
        }
    }
    size_t* offsets=new size_t[count];
    PERMUTE(A,B,C,offsets);
    Tensor res(shape);
    double* res_data=res.data();
    size_t sh=shape[axis];
    size_t step=strides[axis];
    if(count<=min_count){
        for(size_t i=0;i<count;i++){
            size_t off=offsets[i];
            res_data[off]=func(initial,data[off]);
            for(size_t j=1;j<sh;j++){
                res_data[off+j*step]=func(res_data[off+(j-1)*step],data[off+j*step]);
            }
        }
    }
    else{
        #pragma omp parallel for num_threads(thread_c)
        for(size_t i=0;i<count;i++){
            size_t off=offsets[i];
            res_data[off]=func(initial,data[off]);
            for(size_t j=1;j<sh;j++){
                res_data[off+j*step]=func(res_data[off+(j-1)*step],data[off+j*step]);
            }
        }
    }
    delete[] offsets;
    return res;
}
__always_inline Tensor normalize(const Tensor& t){
    return t/t.norm();
}
}
#undef PERMUTE