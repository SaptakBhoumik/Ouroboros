#pragma once
#include "macros.hpp"
#include <fstream>
#include <cstdint>
#include <memory>
#include "tensor.hpp"
namespace Ouroboros{
namespace Utils{
enum class TensorType{
    //Done to reduce the size of the tensor in the file
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    FLOAT,
    DOUBLE,
};
void write_bin_float(std::fstream& file,float value);
void write_bin_double(std::fstream& file,double value);

void write_bin_uint8(std::fstream& file,std::uint8_t value);
void write_bin_uint16(std::fstream& file,std::uint16_t value);
void write_bin_uint32(std::fstream& file,std::uint32_t value);
void write_bin_uint64(std::fstream& file,std::uint64_t value);

void write_bin_int8(std::fstream& file,std::int8_t value);
void write_bin_int16(std::fstream& file,std::int16_t value);
void write_bin_int32(std::fstream& file,std::int32_t value);
void write_bin_int64(std::fstream& file,std::int64_t value);

void write_bin_str(std::fstream& file,std::string value);

void write_bin_shape(std::fstream& file,const Shape& shape);
void write_bin_tensor(std::fstream& file,const Tensor& tensor,TensorType type=TensorType::DOUBLE);
void write_bin_bool_tensor(std::fstream& file,const BoolTensor& tensor);

double read_bin_double(std::fstream& file);
float read_bin_float(std::fstream& file);

std::uint8_t read_bin_uint8(std::fstream& file);
std::uint16_t read_bin_uint16(std::fstream& file);
std::uint32_t read_bin_uint32(std::fstream& file);
std::uint64_t read_bin_uint64(std::fstream& file);

std::int8_t read_bin_int8(std::fstream& file);
std::int16_t read_bin_int16(std::fstream& file);
std::int32_t read_bin_int32(std::fstream& file);
std::int64_t read_bin_int64(std::fstream& file);

std::string read_bin_str(std::fstream& file);

Shape read_bin_shape(std::fstream& file);
Tensor read_bin_tensor(std::fstream& file);
BoolTensor read_bin_bool_tensor(std::fstream& file);

template<typename T>
class Iterator{
    T* m_data;
    std::size_t m_count=0;
    std::size_t m_step=0;
    public:
    struct It{
        It(T* data,std::size_t idx,std::size_t step):data(data),idx(idx),step(step){}
        T operator*(){
            return data[idx*step];
        }
        It& operator++(){
            idx++;
            return *this;
        }
        It& operator++(int){
            auto temp=*this;
            idx++;
            return temp;
        }
        It& operator--(){
            idx--;
            return *this;
        }
        It& operator--(int){
            auto temp=*this;
            idx--;
            return temp;
        }
        const T* operator->()const{
            return data+idx*step;
        }
        bool operator==(It& other){
            return idx==other.idx;
        }
        bool operator!=(It& other){
            return idx!=other.idx;
        }
        private:
        T* data;
        std::size_t idx=0;
        std::size_t step=0;
    };
    Iterator(T* data,std::size_t count,std::size_t step=1){
        this->m_data=data;
        this->m_count=count;
        this->m_step=step;
    }
    Iterator(const T* data,std::size_t count,std::size_t step=1){
        this->m_data=const_cast<T*>(data);//We never modify the data so it is safe to cast it
        this->m_count=count;
        this->m_step=step;
    }
    Iterator& operator=(const Iterator& other){
        this->m_data=other.m_data;
        this->m_count=other.m_count;
        this->m_step=other.m_step;
        return *this;
    }
    It begin(){
        return It(m_data,0,m_step);
    }
    It end(){
        return It(m_data,m_count,m_step);
    }
};
}
}
