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
#define PERMUTE_OFFSET(A,B,C,D) \
{\
    size_t j=0;\
    while (true) {\
        size_t offset=0;\
        _Pragma("omp simd reduction(+:offset)")\
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

#define PERMUTE_IDX(A,B,D,count) \
{\
    /*D is a matrix of dim(count,B.size())*/\
    size_t j=0;\
    while (true) {\
        for (size_t i = 0; i < B.size(); ++i) {\
            D[j*count+i]=B[i];\
        }\
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
        j++;\
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
    const Shape input_shape=t.shape();
    const Shape input_strides=t.strides();
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
    PERMUTE_OFFSET(A,B,C,offsets);
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
    const Shape shape=t.shape();
    const Shape strides=t.strides();
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
    PERMUTE_OFFSET(A,B,C,offsets);
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
template<double(*func)(double,double),size_t thread_c=8,size_t min_count=__MIN__COUNT__FOR__THREAD__>
__always_inline Tensor outer(const Tensor& t1,const Tensor& t2){
    const auto t1_shape=t1.shape();
    const auto t2_shape=t2.shape();
    const auto t1_strides=t1.strides();
    const auto t2_strides=t2.strides();
    const size_t t1_count=t1_shape.count();
    const size_t t2_count=t2_shape.count();
    const size_t t1_dim=t1_shape.dim();
    const size_t t2_dim=t2_shape.dim();

    size_t* res_shape_ptr=new size_t[t1_dim+t2_dim];
    for(size_t i=0;i<t1_dim;i++){
        res_shape_ptr[i]=t1_shape[i];
    }
    for(size_t i=0;i<t2_dim;i++){
        res_shape_ptr[i+t1_dim]=t2_shape[i];
    }
    Shape res_shape(t1_dim+t2_dim,res_shape_ptr);
    delete[] res_shape_ptr;
    Tensor res(res_shape);
    Shape res_strides=res.strides();
    double* res_data=res.data();
    size_t* t1_idxs=new size_t[t1_dim*t1_count];
    {
        std::vector<size_t> A;
        std::vector<size_t> B(t1_dim, 0);
        A.reserve(t1_dim);
        for(size_t i=0;i<t1_dim;i++){
            A.emplace_back(t1_shape[i]);
        }
        PERMUTE_IDX(A,B,t1_idxs,t1_dim);
    }
    size_t* t2_idxs=new size_t[t2_dim*t2_count];
    {
        std::vector<size_t> A;
        std::vector<size_t> B(t2_dim, 0);
        A.reserve(t2_dim);
        for(size_t i=0;i<t2_dim;i++){
            A.emplace_back(t2_shape[i]);
        }
        PERMUTE_IDX(A,B,t2_idxs,t2_dim);
    }
    
    const double* t1_data=t1.data();
    const double* t2_data=t2.data();
    if(res_shape.count()<=min_count){
        for(size_t i=0;i<t1_count;i++){
            for(size_t j=0;j<t2_count;j++){
                size_t t1_off=0;
                size_t t2_off=0;
                size_t res_off=0;
                #pragma omp simd reduction(+:t1_off,t2_off,res_off)
                for(size_t k=0;k<t1_dim;k++){
                    t1_off+=t1_idxs[i*t1_dim+k]*t1_strides[k];
                    res_off+=t1_idxs[i*t1_dim+k]*res_strides[k];
                }
                for(size_t k=0;k<t2_dim;k++){
                    t2_off+=t2_idxs[j*t2_dim+k]*t2_strides[k];
                    res_off+=t2_idxs[j*t2_dim+k]*res_strides[k+t1_dim];
                }
                res_data[res_off]=func(t1_data[t1_off],t2_data[t2_off]);
            }
        }
    }
    else{
        if(t1_count>t2_count){
            #pragma omp parallel for num_threads(thread_c)
            for(size_t i=0;i<t1_count;i++){
                for(size_t j=0;j<t2_count;j++){
                    size_t t1_off=0;
                    size_t t2_off=0;
                    size_t res_off=0;
                    #pragma omp simd reduction(+:t1_off,t2_off,res_off)
                    for(size_t k=0;k<t1_dim;k++){
                        t1_off+=t1_idxs[i*t1_dim+k]*t1_strides[k];
                        res_off+=t1_idxs[i*t1_dim+k]*res_strides[k];
                    }
                    for(size_t k=0;k<t2_dim;k++){
                        t2_off+=t2_idxs[j*t2_dim+k]*t2_strides[k];
                        res_off+=t2_idxs[j*t2_dim+k]*res_strides[k+t1_dim];
                    }
                    res_data[res_off]=func(t1_data[t1_off],t2_data[t2_off]);
                }
            }
        }
        else{
            #pragma omp parallel for num_threads(thread_c)
            for(size_t j=0;j<t2_count;j++){
                for(size_t i=0;i<t1_count;i++){
                    size_t t1_off=0;
                    size_t t2_off=0;
                    size_t res_off=0;
                    #pragma omp simd reduction(+:t1_off,t2_off,res_off)
                    for(size_t k=0;k<t1_dim;k++){
                        t1_off+=t1_idxs[i*t1_dim+k]*t1_strides[k];
                        res_off+=t1_idxs[i*t1_dim+k]*res_strides[k];
                    }
                    for(size_t k=0;k<t2_dim;k++){
                        t2_off+=t2_idxs[j*t2_dim+k]*t2_strides[k];
                        res_off+=t2_idxs[j*t2_dim+k]*res_strides[k+t1_dim];
                    }
                    res_data[res_off]=func(t1_data[t1_off],t2_data[t2_off]);
                }
            }
        }
    }
    delete[] t1_idxs;
    delete[] t2_idxs;
    return res;
}
template<double(*func)(double,double),size_t thread_c=8,size_t min_count=__MIN__COUNT__FOR__THREAD__>
__always_inline Tensor at(const Tensor& t1,const Tensor& t2,const Shape& from,const Shape& to){
    const auto t1_shape=t1.shape();
    const auto t2_shape=t2.shape();
    const auto t1_strides=t1.strides();
    const auto t2_strides=t2.strides();
    const size_t t2_count=t2_shape.count();
    const size_t dim=from.dim();
    if(from.dim()!=to.dim() || from.dim()!=t2.shape().dim() || from.dim()!=t1.shape().dim()){
        throw std::invalid_argument("Invalid shape");
    }
    for(size_t i=0;i<dim;i++){
        if(to[i]>t1_shape[i] || to[i]<=from[i] || (to[i]-from[i])!=t2_shape[i]){
            throw std::invalid_argument("Invalid index");
        }
    }

    Tensor res=t1;
    double* res_data=res.data();
    size_t* t2_idxs=new size_t[dim*t2_count];
    {
        std::vector<size_t> A;
        std::vector<size_t> B(dim, 0);
        A.reserve(dim);
        for(size_t i=0;i<dim;i++){
            A.emplace_back(t2_shape[i]);
        }
        PERMUTE_IDX(A,B,t2_idxs,dim);
    }
    const double* t1_data=t1.data();
    const double* t2_data=t2.data();
    if(t2_count<=min_count){
        for(size_t i=0;i<t2_count;i++){
            size_t res_off=0;
            size_t t2_off=0;
            #pragma omp simd reduction(+:res_off,t2_off)
            for(size_t j=0;j<dim;j++){
                size_t temp=t2_idxs[i*dim+j];
                res_off+=t1_strides[j]*(temp+from[j]);
                t2_off+=t2_strides[j]*temp;
            }
            res_data[res_off]=func(t1_data[res_off],t2_data[t2_off]);
        }
    }
    else{
        #pragma omp parallel for num_threads(thread_c)
        for(size_t i=0;i<t2_count;i++){
            size_t res_off=0;
            size_t t2_off=0;
            #pragma omp simd reduction(+:res_off,t2_off)
            for(size_t j=0;j<dim;j++){
                size_t temp=t2_idxs[i*dim+j];
                res_off+=t1_strides[j]*(temp+from[j]);
                t2_off+=t2_strides[j]*temp;
            }
            res_data[res_off]=func(t1_data[res_off],t2_data[t2_off]);
        }
    }
    delete[] t2_idxs;
    return res;
}
__always_inline Tensor normalize(const Tensor& t){
    return t/t.norm();
}
}
#undef PERMUTE_OFFSET
#undef PERMUTE_IDX