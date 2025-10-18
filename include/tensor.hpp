#pragma once
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <sys/cdefs.h>
#include <vector>
namespace Ouroboros{
class Shape{
    std::uint64_t* m_shape=nullptr;
    std::uint64_t* m_strides=nullptr;
    std::uint64_t m_count=1;//No of elements in the tensor i.e product of elms in m_shape
    std::uint64_t m_dim=0;//No of dimensions in the tensor i.e no of elms in m_shape
public:
    Shape(std::uint64_t dim,std::uint64_t val);
    Shape(std::uint64_t dim,std::uint64_t* shape);//Note we copy the shape
    Shape(std::initializer_list<std::uint64_t> shape);
    Shape(const Shape& shape);
    Shape(Shape&& shape);

    void operator=(const Shape& shape);
    void operator=(Shape&& shape);
    void operator=(std::initializer_list<std::uint64_t> shape);

    __always_inline const std::uint64_t operator[](std::uint64_t index) const{
        return m_shape[index];
    }
    __always_inline std::size_t offset(const std::vector<std::size_t>& indices) const{
        std::size_t off=0;
        for(std::size_t i=0;i<m_dim;i++){
            off+=indices[i]*m_strides[i];
        }
        return off;
    }
    const std::uint64_t* begin() const;
    const std::uint64_t* end() const;

    bool operator==(const Shape& shape) const;
    bool operator!=(const Shape& shape) const;

    std::uint64_t count() const;
    std::uint64_t dim() const;

    ~Shape();
}; 
std::ostream& operator<<(std::ostream& os,const Shape& shape);

template<typename T>
class Tensor{
    Shape m_shape;
    T* m_data=nullptr;
public:
    Tensor(const Shape& shape):m_shape(shape){
        m_data=new T[m_shape.count()];
    }
    Tensor(const Shape& shape,T val):m_shape(shape){
        m_data=new T[m_shape.count()];
        for(std::size_t i=0;i<m_shape.count();i++){
            m_data[i]=val;
        }
    }
    Tensor(const Shape& shape,const T* data):m_shape(shape){
        m_data=new T[m_shape.count()];
        for(std::size_t i=0;i<m_shape.count();i++){
            m_data[i]=data[i];
        }
    }
    Tensor(const Tensor<T>& tensor):m_shape(tensor.m_shape){
        m_data=new T[m_shape.count()];
        for(std::size_t i=0;i<m_shape.count();i++){
            m_data[i]=tensor.m_data[i];
        }
    }
    Tensor(Tensor<T>&& tensor):m_shape(std::move(tensor.m_shape)){
        m_data=tensor.m_data;
        tensor.m_data=nullptr;
    }
    ~Tensor(){
        if(m_data!=nullptr){
            delete[] m_data;
            m_data=nullptr;
        }
    }

    void operator=(const Tensor<T>& tensor){
        if(this==&tensor){
            return;
        }
        if(m_shape.count()!=tensor.m_shape.count()){
            if(m_data!=nullptr){
                delete[] m_data;
            }
            m_shape=tensor.m_shape;
            m_data=new T[m_shape.count()];
        }
        for(std::size_t i=0;i<m_shape.count();i++){
            m_data[i]=tensor.m_data[i];
        }
    }
    void operator=(Tensor<T>&& tensor){
        if(this==&tensor){
            return;
        }
        if(m_data!=nullptr){
            delete[] m_data;
        }
        m_shape=std::move(tensor.m_shape);
        m_data=tensor.m_data;
        tensor.m_data=nullptr;
    }

    __always_inline T& operator[](std::size_t index){
        return m_data[index];
    }
    __always_inline const T& operator[](std::size_t index) const{
        return m_data[index];
    }
    __always_inline T& operator[](const std::vector<std::size_t>& indices){
        std::size_t off=m_shape.offset(indices);
        return m_data[off];
    }
    __always_inline const T& operator[](const std::vector<std::size_t>& indices) const{
        std::size_t off=m_shape.offset(indices);
        return m_data[off];
    }

    const Shape& shape() const{
        return m_shape;
    }
    const T* data() const{
        return m_data;
    }
    T* data(){
        return m_data;
    }
    const std::size_t size() const{
        return m_shape.count();
    }
    const std::size_t dim() const{
        return m_shape.dim();
    }
};
}

#include "op0.hpp"
#include "op1.hpp"
#include "op2.hpp"
#include "op3.hpp"
#include "loss.hpp"