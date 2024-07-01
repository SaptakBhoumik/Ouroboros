#pragma once
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

void write_bin_uint8(std::fstream& file,uint8_t value);
void write_bin_uint16(std::fstream& file,uint16_t value);
void write_bin_uint32(std::fstream& file,uint32_t value);
void write_bin_uint64(std::fstream& file,uint64_t value);

void write_bin_int8(std::fstream& file,int8_t value);
void write_bin_int16(std::fstream& file,int16_t value);
void write_bin_int32(std::fstream& file,int32_t value);
void write_bin_int64(std::fstream& file,int64_t value);

void write_bin_str(std::fstream& file,std::string value);

void write_bin_shape(std::fstream& file,const Shape& shape);
void write_bin_tensor(std::fstream& file,const Tensor& tensor,TensorType type=TensorType::DOUBLE);
void write_bin_bool_tensor(std::fstream& file,const BoolTensor& tensor);

double read_bin_double(std::fstream& file);
float read_bin_float(std::fstream& file);

uint8_t read_bin_uint8(std::fstream& file);
uint16_t read_bin_uint16(std::fstream& file);
uint32_t read_bin_uint32(std::fstream& file);
uint64_t read_bin_uint64(std::fstream& file);

int8_t read_bin_int8(std::fstream& file);
int16_t read_bin_int16(std::fstream& file);
int32_t read_bin_int32(std::fstream& file);
int64_t read_bin_int64(std::fstream& file);

std::string read_bin_str(std::fstream& file);

Shape read_bin_shape(std::fstream& file);
Tensor read_bin_tensor(std::fstream& file);
BoolTensor read_bin_bool_tensor(std::fstream& file);
}
}
