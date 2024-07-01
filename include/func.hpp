#pragma once
#include "op.hpp"
#include <omp.h>
#include <tuple>
#include <cmath>
#include <type_traits>

#include "macros.hpp"

namespace Ouroboros{
namespace __Private__Impl__{
//Do not use this functions from here in your code
template<auto func,typename tuple, std::size_t ... Is>
__always_inline void __apply_impl(const tuple& t,double* res,size_t idx, std::index_sequence<Is...>){
    res[idx]=func(std::get<Is>(t)[idx]...);
}
template<std::size_t N,auto func, typename tuple,typename Indices = std::make_index_sequence<N>>
__always_inline void __apply(const tuple& t,double* res,size_t idx){
    __apply_impl<func>(t,res,idx,Indices{});
}

template<auto func,typename tuple, std::size_t ... Is>
__always_inline void __bool__apply_impl(const tuple& t,bool* res,size_t idx, std::index_sequence<Is...>){
    res[idx]=func(std::get<Is>(t)[idx]...);
}
template<std::size_t N,auto func, typename tuple,typename Indices = std::make_index_sequence<N>>
__always_inline void __bool__apply(const tuple& t,bool* res,size_t idx){
    __bool__apply_impl<func>(t,res,idx,Indices{});
}


template<auto func,typename T,size_t thread_c=4,size_t min_count=__MIN__COUNT__FOR__THREAD__,typename ... Ts>
__always_inline Tensor double_transform(const T& t,const Ts&... args){
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
        #pragma omp parallel for num_threads(thread_c)
        for(size_t i=0;i<count;i++){
            __Private__Impl__::__apply<n,func>(arg_data,res_data,i);
        }
    }
    return res;
}

template<auto func,typename T,size_t thread_c=4,size_t min_count=__MIN__COUNT__FOR__THREAD__,typename ... Ts>
__always_inline BoolTensor bool_transform(const T& t,const Ts&... args){
    constexpr size_t n = sizeof...(Ts)+1;
    auto arg_data = std::make_tuple(t.data(),args.data()...);
    auto shape = t.shape();
    BoolTensor res(shape);
    size_t count=shape.count();
    bool* res_data=res.data();
    if(count<=min_count){
        for(size_t i=0;i<count;i++){
            __Private__Impl__::__bool__apply<n,func>(arg_data,res_data,i);
        }
    }
    else{
        #pragma omp parallel for num_threads(thread_c)
        for(size_t i=0;i<count;i++){
            __Private__Impl__::__bool__apply<n,func>(arg_data,res_data,i);
        }
    }
    return res;
}
//Credit:-https://stackoverflow.com/questions/27822277/finding-out-the-return-type-of-a-function-lambda-or-function

template <typename F>
struct return_type_impl;

template <typename R, typename... Args>
struct return_type_impl<R(Args...)> { using type = R; };

template <typename R, typename... Args>
struct return_type_impl<R(Args..., ...)> { using type = R; };

template <typename R, typename... Args>
struct return_type_impl<R(*)(Args...)> { using type = R; };

template <typename R, typename... Args>
struct return_type_impl<R(*)(Args..., ...)> { using type = R; };

template <typename R, typename... Args>
struct return_type_impl<R(&)(Args...)> { using type = R; };

template <typename R, typename... Args>
struct return_type_impl<R(&)(Args..., ...)> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args...)> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args..., ...)> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args...) &> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args..., ...) &> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args...) &&> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args..., ...) &&> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args...) const> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args..., ...) const> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args...) const&> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args..., ...) const&> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args...) const&&> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args..., ...) const&&> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args...) volatile> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args..., ...) volatile> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args...) volatile&> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args..., ...) volatile&> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args...) volatile&&> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args..., ...) volatile&&> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args...) const volatile> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args..., ...) const volatile> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args...) const volatile&> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args..., ...) const volatile&> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args...) const volatile&&> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args..., ...) const volatile&&> { using type = R; };

template <typename T, typename = void>
struct return_type
    : return_type_impl<T> {};

template <typename T>
struct return_type<T, decltype(void(&T::operator()))>
    : return_type_impl<decltype(&T::operator())> {};

template <typename T>
using return_type_t = typename return_type<T>::type;


template<bool n>
struct Typer{};

template<>
struct Typer<0>{
    typedef Tensor Type;
};

template<>
struct Typer<1>{
    typedef BoolTensor Type;
};

}

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
template<double(*func)(const double*,const double*),size_t thread_c=8>
__always_inline Tensor reduce(const Tensor& t,size_t axis=0){
    if(axis>=t.shape().dim()){
        throw std::invalid_argument("Invalid axis");
    }
    Shape input_shape=t.shape();
    Shape input_strides=t.strides();
    Shape output_shape=input_shape;
    output_shape[axis]=1;
    const double* data=t.data();
    if(input_shape[axis]==1){
        return t;
    }
    else if(input_shape.dim()==1){
        return Tensor({1},func(data,data+input_shape[0]));
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
    size_t j=0;
    while (true) {
        // Print the current array B
        size_t offset=0;
        for (size_t i = 0; i < B.size(); ++i) {
            offset+=B[i]*C[i];
        }
        offsets[j++]=offset;

        // Find the rightmost index that can be incremented
        int64_t k = (int64_t)A.size() - 1;
        while (k >= 0 && B[k] == A[k] - 1) {
            k--;
        }

        // If no such index exists, we are done
        if (k < 0) {
            break;
        }
        // Increment the current index and reset all subsequent indices
        B[k]++;
        for (size_t i = k + 1; i < A.size(); ++i) {
            B[i] = 0;
        }
    }
    Tensor res(output_shape);
    double* res_data=res.data();
    size_t sh=input_shape[axis];
    size_t st=input_strides[axis];
    #pragma omp parallel for num_threads(thread_c)
    for(size_t i=0;i<count;i++){
        double* temp_data=new double[sh];
        size_t off=offsets[i];
        for(size_t j=0;j<sh;j++){
            temp_data[j]=data[off+j*st];
        }
        res_data[i]=func(temp_data,temp_data+sh);
    }
    delete[] offsets;
    return res;
}
__always_inline Tensor normalize(const Tensor& t){
    return t/t.norm();
}
}