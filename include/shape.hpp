
#pragma once
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
namespace Ouroboros{
class Shape{
    size_t* m_shape=nullptr;
    size_t m_count=1;//No of elements in the tensor i.e product of elms in m_shape
    size_t m_dim=0;//No of dimensions in the tensor i.e no of elms in m_shape
public:
    Shape(size_t dim,size_t val);
    Shape(size_t dim,size_t* shape);//Note we copy the shape
    Shape(std::initializer_list<size_t> shape);
    Shape(const Shape& shape);
    Shape(Shape&& shape);

    Shape& operator=(const Shape& shape);
    Shape& operator=(Shape&& shape);
    
    Shape& operator=(std::initializer_list<size_t> shape);

    size_t get(size_t index) const;//This one checks for errors
    __always_inline size_t& operator[](size_t index){
        #ifdef __OUROBOROS_CHECK__
        if(index>=m_dim){
            throw std::invalid_argument("Invalid index");
        }
        #endif
        return m_shape[index];
    }
    __always_inline const size_t& operator[](size_t index) const{
        #ifdef __OUROBOROS_CHECK__
        if(index>=m_dim){
            throw std::invalid_argument("Invalid index");
        }
        #endif
        return m_shape[index];
    }
    const size_t* begin() const;
    const size_t* end() const;

    bool operator==(const Shape& shape) const;
    bool operator!=(const Shape& shape) const;

    size_t count() const;
    size_t dim() const;

    ~Shape();
}; 
std::ostream& operator<<(std::ostream& os,const Shape& shape);
}
