//Stuff from https://github.com/SaptakBhoumik/Ouroboros/blob/master/include/func.hpp
//Various types of norm
//Min,max,norm etc
#pragma once
#include "tensor.hpp"
#include <algorithm>
#include <cstdint>
#include <sys/cdefs.h>
#include <sys/types.h>

#include <array>
namespace Ouroboros{
template<const auto func,
        typename T,
        typename ... Ts>
__always_inline auto transform(const T& t,const Ts& ... args){
    using R = decltype(func(t[0], args[0]...));
    Tensor<R> result(t.shape());

    for(std::uint64_t i=0;i<t.size();i++){
        result[i]=func(t[i],args[i]...);
    }
    return result;
}
template<const auto func,
        typename T,
        typename ... Ts>
__always_inline auto reduce(std::uint64_t axis,const T& t,const Ts& ... args){
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
__always_inline auto reduce(std::set<uint64_t> axis,const T& t,const Ts& ... args){
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
__always_inline auto accumulate(const T& t,std::uint64_t axis=0,double initial = 0){
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

template<const auto func,
        typename T0,
        typename T1>
__always_inline auto outer(const T0& a,const T1& b){
    using R = decltype(func(a[0], b[0]));
    std::vector<std::uint64_t> result_shape;
    auto a_shape = a.shape().to_vector();
    auto b_shape = b.shape().to_vector();
    result_shape.insert(result_shape.end(), a_shape.begin(), a_shape.end());
    result_shape.insert(result_shape.end(), b_shape.begin(), b_shape.end());
    Shape result_shape_obj(result_shape);
    Tensor<R> result(result_shape_obj);

    IdxIterator it_result(result_shape_obj);
    for(auto it=it_result.begin();it!=it_result.end();++it){
        std::vector<uint64_t> idx = it.get_index();
        result[*it]=func(a[idx.begin()], b[idx.begin() + a.dim()]);
    }
    return result;
}

template<typename T0,
        typename ... Ts>
__always_inline T0 concat(std::uint64_t axis,const T0& t0,const T0& t1,const Ts& ... ts){
    constexpr std::uint64_t num_tensors = 2 + sizeof...(ts);
    //Concatenate tensors along the given axis
    std::array<std::uint64_t, num_tensors> alpha = {t0.shape()[axis], t1.shape()[axis], ts.shape()[axis]...};
    std::array<const T0*,num_tensors> tensors = {&t0, &t1, &ts...};

    Shape new_shape=t0.shape();
    std::uint64_t total_size = 0;
    for(auto val:alpha){
        total_size += val;
    }
    new_shape.set(axis,total_size);

    T0 result(new_shape);
    IdxIterator it_result(new_shape);
    for(auto it=it_result.begin();it!=it_result.end();++it){
        std::vector<uint64_t> idx = it.get_index();
        uint64_t axis_i = idx[axis];
        //Do constexpr unrolling to find which tensor to pick from
        uint64_t offset = 0;
        for(std::uint64_t i=0;i<num_tensors;i++){
            if(axis_i<alpha[i]){
                idx[axis] = axis_i;
                result[*it] = tensors[i]->operator[](idx);
                break;
            }else{
                axis_i -= alpha[i];
            }
        }
    }
    return result;
}   
template<typename T>
__always_inline T transpose(const T& t,std::size_t ax1=0,std::size_t ax2=1) {
    Shape new_shape = t.shape();
    std::uint64_t a1 = new_shape[ax1];;
    std::uint64_t a2 = new_shape[ax2];
    new_shape.set(ax1,a2);
    new_shape.set(ax2,a1);
    T result(new_shape);
    IdxIterator it_result(new_shape);
    for(auto it=it_result.begin();it!=it_result.end();++it){
        std::vector<uint64_t> idx = it.get_index();
        std::uint64_t temp = idx[ax1];
        idx[ax1] = idx[ax2];
        idx[ax2] = temp;
        result[*it] = t[idx];
    }
    return result;
}

template<const auto func,
        typename T,
        typename ... Ts>
__always_inline auto at(std::vector<uint64_t> start,std::vector<uint64_t> end,std::vector<uint64_t> step,const T& t,const Ts& ... args){
    using R = decltype(func(t[0], args[0]...));
    Tensor<R> result(t.shape());

    IdxIterator2 it(t.shape(), start, end, step);
    for(auto it0=it.begin();it0!=it.end();++it0){
        result[*it0]=func(t[*it0],args[*it0]...);
    }
    return result;
}

template<typename T>
__always_inline T flip(const T& t,std::uint64_t axis) {
    T result(t.shape());
    
    IdxIterator it_result(t.shape());
    for(auto it=it_result.begin();it!=it_result.end();++it){
        uint64_t offset = *it;
        std::vector<uint64_t> idx = it.get_index();
        idx[axis] = t.shape()[axis] - 1 - idx[axis];
        result[offset] = t[idx];
    }
    return result;
}


template<const auto func,
        typename T0,
        typename T1,
        typename ... Ts>
__always_inline auto broadcast(const T0& t0,const T1& t1,const Ts& ... args){
    using R = decltype(func(t0[0], t1[0], args[0]...));
    //Determine the broadcasted shape
    std::vector<Shape> shapes = {t0.shape(), t1.shape(), args.shape()...};
    std::vector<std::uint64_t> result_shape_vec;
    result_shape_vec.assign(t0.dim(), 1);
    for(std::uint64_t i=0;i<t0.dim();i++){
        for(auto& shape:shapes){
            if(shape[i]>result_shape_vec[i]){
                result_shape_vec[i]=shape[i];
                break;
            }
        }
    }
    Shape result_shape(result_shape_vec);
    Tensor<R> result(result_shape);

    IdxIterator it_result(result_shape);

    auto get_broadcast_value = [](const auto& t, const std::vector<uint64_t>& idx) -> decltype(t[0]) {
        Shape t_shape = t.shape();
        uint64_t offset = 0;
        for(std::uint64_t i=0;i<t.dim();i++){
            if(t_shape[i]!=1){
                offset += idx[i]*t_shape.get_stride(i);
            }
        }
        return t[offset];
    };

    for(auto it=it_result.begin();it!=it_result.end();++it){
        std::vector<uint64_t> idx = it.get_index();
        result[*it]=func(get_broadcast_value(t0, idx), get_broadcast_value(t1, idx), get_broadcast_value(args, idx)...);
    }
    return result;
}
}