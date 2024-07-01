#ifndef __Ouroboros__
#error "This file should not be included directly"
#endif
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
}
