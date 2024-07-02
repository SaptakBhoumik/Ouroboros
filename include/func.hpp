#pragma once
#include "op.hpp"
#include <omp.h>
#include <tuple>
#include <cmath>
#include <type_traits>
#include "utils.hpp"
#include "macros.hpp"

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
#define __Ouroboros__ 
#include "private_impl.hpp"//Has to be declared after PERMUTE_IDX and PERMUTE_OFFSET are defined
#undef __Ouroboros__
namespace Ouroboros{
template<typename T,
        size_t thread_c=8,
        size_t min_count=__MIN__COUNT__FOR__THREAD__,
        typename func_type,
        typename ... Ts>
__always_inline typename __Private__Impl__::Typer<std::is_same<__Private__Impl__::return_type_t<func_type>, bool>{}>::Type
         transform(const func_type& func,const T& t,const Ts&... args){
    static_assert(std::is_same<__Private__Impl__::return_type_t<func_type>, bool>{} ||
                  std::is_same<__Private__Impl__::return_type_t<func_type>, double>{}, 
                    "Function must return double or bool");
    constexpr size_t n = sizeof...(Ts)+1;
    auto arg_data = std::make_tuple(t.data(),args.data()...);
    auto shape = t.shape();
    if constexpr(n>1){
        std::vector<bool> check = {args.shape()==shape...};
        for(auto x:check){
            if(!x){
                throw std::invalid_argument("Shapes must be the same");
            }
        }
    }
    typename __Private__Impl__::Typer<std::is_same<__Private__Impl__::return_type_t<func_type>, bool>{}>::Type res(shape);
    size_t count=shape.count();
    auto res_data=res.data();
    if(count<=min_count){
        for(size_t i=0;i<count;i++){
            __Private__Impl__::__apply<n>(func,arg_data,res_data,i);
        }
    }
    else{
        #pragma omp parallel for num_threads(thread_c)
        for(size_t i=0;i<count;i++){
            __Private__Impl__::__apply<n>(func,arg_data,res_data,i);
        }
    }
    return res;
}
template<const auto func,
        typename T,
        size_t thread_c=8,
        size_t min_count=__MIN__COUNT__FOR__THREAD__,
        typename ... Ts>
__always_inline typename __Private__Impl__::Typer<std::is_same<__Private__Impl__::return_type_t<decltype(func)>, bool>{}>::Type
         transform(const T& t,const Ts&... args){
    return transform<T,thread_c,min_count>(func,t,args...);
}



template<size_t thread_c=8,size_t min_count=__MIN__COUNT__FOR__THREAD__>
__always_inline Tensor reduce(const std::function<double(Utils::Iterator<double>)>& func
                                ,const Tensor& t,size_t axis=0){
    if(axis>=t.shape().dim()){
        throw std::invalid_argument("Invalid axis");
    }
    const Shape input_shape=t.shape();
    const Shape input_strides=t.strides();
    Shape output_shape=input_shape;
    output_shape.set(axis,1);
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
template<double(*func)(Utils::Iterator<double>),size_t thread_c=8,size_t min_count=__MIN__COUNT__FOR__THREAD__>
__always_inline Tensor reduce(const Tensor& t,size_t axis=0){
    return reduce<thread_c,min_count>(func,t,axis);
}


template<size_t thread_c=8,size_t min_count=__MIN__COUNT__FOR__THREAD__>
__always_inline Tensor reduce(const std::function<double(Utils::Iterator<double>)>& func
                                ,const Tensor& t,std::vector<size_t> axes){
    if(axes.size()==0){
        return Tensor({1},func(Utils::Iterator<double>(t.data(),t.count())));
    }
    Tensor res=reduce<thread_c,min_count>(func,t,axes[0]);
    for(size_t i=1;i<axes.size();i++){
        res=reduce<thread_c,min_count>(func,res,axes[i]);
    }
    return res;
}
template<double(*func)(Utils::Iterator<double>),size_t thread_c=8,size_t min_count=__MIN__COUNT__FOR__THREAD__>
__always_inline Tensor reduce(const Tensor& t,std::vector<size_t> axes){
    return reduce<thread_c,min_count>(func,t,axes);
}


template<size_t thread_c=8,size_t min_count=__MIN__COUNT__FOR__THREAD__>
__always_inline Tensor accumulate(const std::function<double(double,double)>& func,
                                  const Tensor& t,size_t axis=0,double initial = 0){
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
__always_inline Tensor accumulate(const Tensor& t,size_t axis=0,double initial = 0){
    return accumulate<thread_c,min_count>(func,t,axis,initial);
}


template<size_t thread_c=8,size_t min_count=__MIN__COUNT__FOR__THREAD__>
__always_inline Tensor outer(const std::function<double(double,double)>& func,const Tensor& t1,const Tensor& t2){
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
__always_inline Tensor outer(const Tensor& t1,const Tensor& t2){
    return outer<thread_c,min_count>(func,t1,t2);
}


template<typename func_type,
        typename T,
        size_t thread_c=8,
        size_t min_count=__MIN__COUNT__FOR__THREAD__,
        typename ... Ts>
__always_inline T at(const func_type& func,const T& t1,const Shape& from,const Shape& to,const Ts&... t2){
    T res=t1;
    auto res_data=res.data();
    const size_t dim=from.dim();
    if(from.dim()!=to.dim()){
        throw std::invalid_argument("Shapes must have the same number of elements");
    }
    size_t* t2_shape_ptr=new size_t[dim];
    const Shape t1_shape=t1.shape();
    for(size_t i=0;i<dim;i++){
        if(from[i]>=to[i]||to[i]>t1_shape[i]){
            throw std::invalid_argument("Invalid shape");
        }
        t2_shape_ptr[i]=to[i]-from[i];
    }    

    const Shape t2_shape(dim,t2_shape_ptr);
    const Shape t2_strides=getStride(t2_shape);
    const Shape t1_strides=t1.strides();
    const size_t t2_count=t2_shape.count();

    delete[] t2_shape_ptr;

    {
        std::vector<bool> check = {t2.shape()==t2_shape...};
        for(auto x:check){
            if(!x){
                throw std::invalid_argument("Invalid shape ");
            }
        }
    }

    
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

    constexpr size_t n=sizeof...(Ts);
    auto tuple=std::make_tuple(t2.data()...);
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
            __Private__Impl__::__apply_self<n>(func,tuple,res_data,t2_off,res_off);
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
            __Private__Impl__::__apply_self<n>(func,tuple,res_data,t2_off,res_off);
        }
    }
    delete[] t2_idxs;
    return res;
}
template<const auto func,
        typename T,
        size_t thread_c=8,
        size_t min_count=__MIN__COUNT__FOR__THREAD__,
        typename ... Ts>
__always_inline T at(const T& t1,const Shape& from,const Shape& to,const Ts&... t2){
   return at<decltype(func),T,thread_c,min_count>(func,t1,from,to,t2...); 
}


template<size_t thread_c=8,size_t min_count=__MIN__COUNT__FOR__THREAD__>
__always_inline Tensor broadcast(const std::function<double(double,double)>& func,const Tensor& t1,const Tensor& t2){
    const auto t1_shape=t1.shape();
    const auto t2_shape=t2.shape();
    if(t1_shape.dim()!=t2_shape.dim()){
        throw std::invalid_argument("Shapes must have the same number of elements");
    }
    if(t1_shape.count()==1&&t2_shape.count()==1){
        return Tensor({1},func(t1[0],t2[0]));
    }
    else if(t1_shape.count()==1){
        double scalar=t1[0];
        auto new_func=[scalar,func](double a)->double{
            return func(scalar,a);
        };
        return transform<Tensor,thread_c,min_count>(new_func,t2);
    }
    else if(t2_shape.count()==1){
        double scalar=t2[0];
        auto new_func=[scalar,func](double a)->double{
            return func(a,scalar);
        };
        return transform<Tensor,thread_c,min_count>(new_func,t1);
    }
    for(size_t i=0;i<t1_shape.dim();i++){
        if(t1_shape[i]>t2_shape[i]){
            return __Private__Impl__::___broadcast<thread_c,min_count>(func,t1,t2);
        }
        else if(t1_shape[i]<t2_shape[i]){
            auto new_func=[func](double a,double b)->double{
                return func(b,a);
            };
            return __Private__Impl__::___broadcast<thread_c,min_count>(new_func,t2,t1);
        }
    }
    return transform<Tensor,thread_c,min_count>(func,t1,t2);
}
template<double(*func)(double,double),size_t thread_c=8,size_t min_count=__MIN__COUNT__FOR__THREAD__>
__always_inline Tensor broadcast(const Tensor& t1,const Tensor& t2){
    return broadcast<thread_c,min_count>(func,t1,t2);
}

template<typename T,
        size_t thread_c=8,
        size_t min_count=__MIN__COUNT__FOR__THREAD__,
        typename ... Ts>
__always_inline T concat(size_t axis,const T& t1,const T& t2,const Ts&... t3){
    const std::vector<T> tensors={t1,t2,t3...};
    const Shape shape=t1.shape();
    const size_t dim=shape.dim();
    if(axis>=dim){
        throw std::invalid_argument("Invalid axis");
    }
    if(dim==1){
        size_t count=0;
        for(const auto& t:tensors){
            count+=t.count();
        }
        T res({count});
        double* res_data=res.data();
        size_t offset=0;
        for(const auto& t:tensors){
            const double* data=t.data();
            const size_t t_count=t.count();
            for(size_t i=0;i<t_count;i++){
                res_data[offset+i]=data[i];
            }
            offset+=t_count;
        }
        return res;
    }
    Shape res_shape=shape;
    size_t count=shape[axis];
    for(size_t i=1;i<tensors.size();i++){
        const T& t=tensors[i];
        if(t.dim()!=dim){
            throw std::invalid_argument("Shapes must be of same dimension");
        }
        const auto s=t.shape();
        for(size_t i=0;i<dim;i++){
            if(i==axis){
                count+=s[i];
            }
            else if(s[i]!=shape[i]){
                throw std::invalid_argument("Shapes must be the same");
            }
        }
    }
    res_shape.set(axis,count);
    return __Private__Impl__::concat<T,thread_c,min_count>(axis,tensors,res_shape);
}

template<typename T,
        size_t thread_c=8,
        size_t min_count=__MIN__COUNT__FOR__THREAD__,
        typename ... Ts>
__always_inline T concat(const T& t1,const T& t2,const Ts&... t3){
    return concat<T,thread_c,min_count>(0,t1,t2,t3...);
}

__always_inline Tensor norm(const Tensor& t,std::vector<size_t> axes){
    auto func=[](Ouroboros::Utils::Iterator<double> a){
                    double sum=0;
                    for(auto x:a){
                        sum+=x*x;
                    }
                    return std::sqrt(sum);
                };
    Tensor res=reduce<func>(t,axes);
    return res;
}  
__always_inline Tensor norm2(const Tensor& t,std::vector<size_t> axes){
    auto func=[](Ouroboros::Utils::Iterator<double> a){
                    double sum=0;
                    for(auto x:a){
                        sum+=x*x;
                    }
                    return sum;
                };
    Tensor res=reduce<func>(t,axes);
    return res;
}  
__always_inline Tensor norm(const Tensor& t,size_t axis){
    return norm(t,std::vector<size_t>{axis});
}
__always_inline Tensor norm2(const Tensor& t,size_t axis){
    return norm2(t,std::vector<size_t>{axis});
}   
__always_inline Tensor normalize(const Tensor& t){
    return t/t.norm();
}
__always_inline Tensor normalize(const Tensor& t,std::vector<size_t> axes){
    auto norm_t=norm(t,axes);
    auto func=[](double a,double norm){
                    return a/norm;
                };
    return broadcast(func,t,norm_t);
}
__always_inline Tensor normalize(const Tensor& t,size_t axis){
    return normalize(t,std::vector<size_t>{axis});
}
}
#undef PERMUTE_OFFSET
#undef PERMUTE_IDX