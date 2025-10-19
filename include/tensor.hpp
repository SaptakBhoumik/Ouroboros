#pragma once
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <sys/cdefs.h>
#include <set>
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

    __always_inline void set(std::uint64_t index,std::uint64_t val){
        m_shape[index]=val;
        //Recompute strides and count
        m_count=1;
        for(std::size_t i=m_dim;i>0;i--){
            if(i==m_dim){
                m_strides[i-1]=1;
            }else{
                m_strides[i-1]=m_strides[i]*m_shape[i];
            }
            m_count*=m_shape[i-1];
        }
    }
    __always_inline const std::uint64_t operator[](std::uint64_t index) const{
        return m_shape[index];
    }
    __always_inline const std::uint64_t get_stride(std::uint64_t index) const{
        return m_strides[index];
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

class NDRange {
    Shape shape;
    std::set<std::size_t> axis;
public:
    NDRange(std::initializer_list<size_t> s,std::set<size_t> axis);

    class Iterator0;
    class Range;
    class Iterator1 {
        const std::vector<size_t>* shape;
        const std::vector<size_t>* weight;
        std::vector<size_t> index;
        std::size_t offset = 0;
        bool end_flag = false;

        void reset(std::size_t off);
        public:
        Iterator1()=default;
        Iterator1(const std::vector<size_t>* shape_, const std::vector<size_t>* weight_, bool end=false,std::size_t off=0);

        __always_inline const std::size_t operator*() const { 
            return offset;
        }
        

        Iterator1& operator++();

        bool operator!=(const Iterator1& other) const;

        friend class Range;
    };

    class Range{
        Iterator1 it_start;
        Iterator1 it_end;
        std::size_t offset = 0;
        bool end_flag = false;
        void reset(std::size_t off);
        public:
        Range()=default;
        Range(const std::vector<size_t>* shape_, const std::vector<size_t>* weight_,std::size_t off=0);

        Iterator1 begin() const;
        Iterator1 end() const;
        friend class Iterator0;
    };
    class Iterator0 {
        //Info from the shape that is in axis i.e fixed first
        std::vector<std::size_t> shape0;
        std::vector<std::size_t> strides0;
        //Info from the shape that is not in axis i.e varying later
        std::vector<std::size_t> shape1;
        std::vector<std::size_t> strides1;
        bool end_flag = false;

        Iterator1 it0;
        Range it1;
        public:
        Iterator0(const Shape* shape_,const std::set<std::size_t>& axis, bool end=false);

        __always_inline const Range& operator*()const{ 
            return it1;
        };
        Iterator0& operator++();
        bool operator!=(const Iterator0& other) const;
    };

    Iterator0 begin() const;
    Iterator0 end()   const;
};
}

#include "op0.hpp"
#include "op1.hpp"
#include "op2.hpp"
#include "op3.hpp"
#include "func0.hpp"
#include "loss.hpp"