#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <sys/cdefs.h>
#include <sys/types.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
namespace Ouroboros{
class Shape{
    std::size_t* m_shape=nullptr;
    std::size_t* m_strides=nullptr;
    std::size_t m_count=1;//No of elements in the tensor i.e product of elms in m_shape
    std::size_t m_dim=0;//No of dimensions in the tensor i.e no of elms in m_shape
public:
    Shape(std::size_t dim,std::size_t val);
    Shape(std::size_t dim,std::size_t* shape);//Note we copy the shape
    Shape(std::initializer_list<std::size_t> shape);
    Shape(std::vector<std::size_t> shape);
    Shape(const Shape& shape);
    Shape(Shape&& shape);

    void operator=(const Shape& shape);
    void operator=(Shape&& shape);
    void operator=(std::initializer_list<std::size_t> shape);

    __always_inline void set(std::size_t index,std::size_t val){
        m_shape[index]=val;
        //Recompute strides and count
        m_strides[m_dim - 1] = 1;
        for (std::size_t i = m_dim - 1; i > 0; --i) {
            m_strides[i - 1] = m_strides[i] * m_shape[i];
        }
        m_count=m_strides[0]*m_shape[0];
    }
    __always_inline const std::size_t operator[](std::size_t index) const{
        return m_shape[index];
    }
    __always_inline const std::size_t get_stride(std::size_t index) const{
        return m_strides[index];
    }

    __always_inline std::size_t offset(const std::vector<std::size_t>& indices) const{
        std::size_t off=0;
        for(std::size_t i=0;i<m_dim;i++){
            off+=indices[i]*m_strides[i];
        }
        return off;
    }
    __always_inline std::size_t offset(const std::size_t* start) const{
        std::size_t off=0;
        for(std::size_t i=0;i<m_dim;i++){
            off+=start[i]*m_strides[i];
        }
        return off;
    }
    const std::size_t* begin() const;
    const std::size_t* end() const;

    bool operator==(const Shape& shape) const;
    bool operator!=(const Shape& shape) const;

    std::size_t count() const;
    std::size_t dim() const;
    std::vector<std::size_t> to_vector() const;

    ~Shape();
}; 
std::ostream& operator<<(std::ostream& os,const Shape& shape);

template<typename T>
class Tensor{
    Shape m_shape;
    T* m_data=nullptr;
    void printTensorRecursively(std::ostream& os, std::size_t dim, std::size_t offset) const{
        if (dim == m_shape.dim() - 1) {
            os << "[";
            for (std::size_t i = 0; i < m_shape[dim]; i++) {
                os << m_data[offset + i];
                if (i != m_shape[dim] - 1) {
                    os << ", ";
                }
            }
            os << "]";
        } else {
            os << "[";
            for (std::size_t i = 0; i < m_shape[dim]; i++) {
                printTensorRecursively(os, dim + 1, offset + i * m_shape.get_stride(dim));
                if (i != m_shape[dim] - 1) {
                    os << ", ";
                }
            }
            os << "]";
        }
    }
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
    Tensor(const Shape& shape,T* data):m_shape(shape){
        m_data=data;
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

    __always_inline std::size_t offset(const std::vector<std::size_t>& indices) const{
        return m_shape.offset(indices);
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
    __always_inline T& operator[](const std::size_t* indices){
        std::size_t off=m_shape.offset(indices);
        return m_data[off];
    }
    __always_inline const T& operator[](const std::size_t* indices) const{
        std::size_t off=m_shape.offset(indices);
        return m_data[off];
    }

    __always_inline void reshape(const Shape& shape){m_shape=shape;}

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
    template<T(*func)(T*,T*)>
    T reduce() const{
        return func(m_data,m_data + m_shape.count());
    }
    template<typename U>
    friend std::ostream& operator<<(std::ostream& os,const Tensor<U>& tensor);
};
template<typename U>
std::ostream& operator<<(std::ostream& os,const Tensor<U>& tensor){
    tensor.printTensorRecursively(os, 0, 0);
    os<<std::endl;
    return os;
}

class NDRange {
    Shape shape;
    std::unordered_set<std::size_t> axis;
public:
    NDRange(Shape s,std::unordered_set<size_t> axis);

    class Iterator0;
    class Range;
    class Iterator1 {
        const std::vector<size_t>* shape;
        const std::vector<size_t>* weight;
        std::vector<size_t> index;
        std::size_t offset = 0;
        bool end_flag = false;

        void reset(std::size_t off, bool end=false);
        public:
        Iterator1()=default;
        Iterator1(const std::vector<size_t>* shape_, const std::vector<size_t>* weight_, bool end=false,std::size_t off=0);

        __always_inline const std::size_t operator*() const { 
            return offset;
        }
        __always_inline const std::vector<size_t>& get_index() const {
            return index;
        }
        

        Iterator1& operator++();

        bool operator!=(const Iterator1& other) const;

        friend class Range;
    };

    class Range{
        Iterator1 it_start;
        Iterator1 it_end;
        std::vector<std::size_t> index;
        std::size_t offset = 0;
        bool end_flag = false;
        void reset(std::size_t off);
        public:
        Range()=default;
        Range(const std::vector<size_t>* shape_, const std::vector<size_t>* weight_,std::vector<std::size_t> index_,std::size_t off=0);

        __always_inline const std::size_t operator*() const { 
            return offset;
        }
        __always_inline const std::vector<size_t>& get_index() const {
            return index;
        }
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
        Iterator0(const Shape* shape_,const std::unordered_set<std::size_t>& axis, bool end=false);

        __always_inline const Range& operator*()const{ 
            return it1;
        };
        Iterator0& operator++();
        bool operator!=(const Iterator0& other) const;
    };

    Iterator0 begin() const;
    Iterator0 end()   const;
};
class IdxIterator {
    std::vector<size_t> shape;
    std::vector<int64_t> is_fixed;
    std::vector<size_t> weight;
    public:
    IdxIterator(Shape s,std::unordered_map<std::size_t,std::size_t> fixed_indices={});

    class Iterator {
        const std::vector<size_t>* shape;
        const std::vector<int64_t>* is_fixed;//Negative if not fixed else fixed value
        const std::vector<size_t>* weight;
        std::vector<size_t> index;
        std::size_t offset = 0;
        bool end_flag = false;

        public:
        Iterator()=default;
        Iterator(const std::vector<size_t>* shape_, const std::vector<int64_t>* is_fixed_, const std::vector<size_t>* weight_,bool end=false);

        __always_inline const std::size_t operator*() const { 
            return offset;
        }
        __always_inline const std::vector<size_t>& get_index() const {
            return index;
        }

        Iterator& operator++();

        bool operator!=(const Iterator& other) const;

        friend class IdxIterator;
    };
    Iterator begin() const;
    Iterator end()   const;

};

class IdxIterator2 {
    std::vector<size_t> start;
    std::vector<size_t> _end;
    std::vector<size_t> step;
    std::vector<size_t> weight;
    public:
    IdxIterator2(const Shape& shape_,const std::vector<size_t>& start_,const std::vector<size_t>& end_,const std::vector<size_t>& step_);

    class Iterator {
        const std::vector<size_t>* start;
        const std::vector<size_t>* end;
        const std::vector<size_t>* step;
        const std::vector<size_t>* weight;

        std::vector<size_t> index;
        std::size_t offset = 0;
        bool end_flag = false;
        public:
        Iterator()=default;
        Iterator(const std::vector<size_t>* start_,const std::vector<size_t>* end_, const std::vector<size_t>* step_,
                const std::vector<size_t>* weight_,bool end=false);

        __always_inline const std::size_t operator*() const { 
            return offset;
        }
        __always_inline const std::vector<size_t>& get_index() const {
            return index;
        }

        Iterator& operator++();

        bool operator!=(const Iterator& other) const;
    };

    Iterator begin() const;
    Iterator end()   const;
};
}
#include "op0.hpp"
#include "op1.hpp"
#include "op2.hpp"
#include "op3.hpp"
#include "func0.hpp"