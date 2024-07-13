
#pragma once
#include "macros.hpp"
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
namespace Ouroboros{
class Shape{
    std::size_t* m_shape=nullptr;
    std::size_t m_count=1;//No of elements in the tensor i.e product of elms in m_shape
    std::size_t m_dim=0;//No of dimensions in the tensor i.e no of elms in m_shape
public:
    Shape(std::size_t dim,std::size_t val);
    Shape(std::size_t dim,std::size_t* shape);//Note we copy the shape
    Shape(std::initializer_list<std::size_t> shape);
    Shape(const Shape& shape);
    Shape(Shape&& shape);

    Shape& operator=(const Shape& shape);
    Shape& operator=(Shape&& shape);
    Shape& operator=(std::initializer_list<std::size_t> shape);

    __always_inline void set(std::size_t index,std::size_t val){
        #ifdef __OUROBOROS_CHECK__
        if(index>=m_dim){
            throw std::invalid_argument("Invalid index");
        }
        #endif
        m_shape[index]=val;
        m_count=1;//Cuz the prev value may be 0
        for(std::size_t i=0;i<m_dim;i++){
            m_count*=m_shape[i];
        }
    }   
    __always_inline const std::size_t operator[](std::size_t index) const{
        #ifdef __OUROBOROS_CHECK__
        if(index>=m_dim){
            throw std::invalid_argument("Invalid index");
        }
        #endif
        return m_shape[index];
    }
    const std::size_t* begin() const;
    const std::size_t* end() const;

    bool operator==(const Shape& shape) const;
    bool operator!=(const Shape& shape) const;

    std::size_t count() const;
    std::size_t dim() const;

    ~Shape();
}; 
std::ostream& operator<<(std::ostream& os,const Shape& shape);
}
