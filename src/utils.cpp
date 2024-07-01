#include "utils.hpp"
#include <iostream>
namespace Ouroboros{
namespace Utils{
__always_inline double handle_endian_d(const double inFloat ){
    #if BYTE_ORDER == LITTLE_ENDIAN
    double retVal;
    char *floatToConvert = ( char*)&inFloat;
    char *returnFloat = (char*)&retVal;
    returnFloat[0] = floatToConvert[7];
    returnFloat[1] = floatToConvert[6];
    returnFloat[2] = floatToConvert[5];
    returnFloat[3] = floatToConvert[4];
    returnFloat[4] = floatToConvert[3];
    returnFloat[5] = floatToConvert[2];
    returnFloat[6] = floatToConvert[1];
    returnFloat[7] = floatToConvert[0];
    return retVal;
    #else
    return inFloat;
    #endif
}
__always_inline float handle_endian_f(const float inFloat ){
    #if BYTE_ORDER == LITTLE_ENDIAN
    float retVal;
    char *floatToConvert = (char*)&inFloat;
    char *returnFloat = (char*)&retVal;
    returnFloat[0] = floatToConvert[3];
    returnFloat[1] = floatToConvert[2];
    returnFloat[2] = floatToConvert[1];
    returnFloat[3] = floatToConvert[0];
    return retVal;
    #else
    return inFloat;
    #endif
}
void write_bin_float(std::fstream& file,float value){
    float val=handle_endian_f(value);
    file.write(reinterpret_cast<char*>(&val),sizeof(float));
}
void write_bin_double(std::fstream& file,double value){
    double val=handle_endian_d(value);
    file.write(reinterpret_cast<char*>(&val),sizeof(double));
}
void write_bin_uint8(std::fstream& file,uint8_t value){
    file.write(reinterpret_cast<char*>(&value),sizeof(uint8_t));
}
void write_bin_uint16(std::fstream& file,uint16_t value){
    uint16_t val=htobe16(value);
    file.write(reinterpret_cast<char*>(&val),sizeof(uint16_t));
}
void write_bin_uint32(std::fstream& file,uint32_t value){
    uint32_t val=htobe32(value);
    file.write(reinterpret_cast<char*>(&val),sizeof(uint32_t));
}
void write_bin_uint64(std::fstream& file,uint64_t value){
    uint64_t val=htobe64(value);
    file.write(reinterpret_cast<char*>(&val),sizeof(uint64_t));
}


void write_bin_int8(std::fstream& file,int8_t value){
    file.write(reinterpret_cast<char*>(&value),sizeof(int8_t));
}
void write_bin_int16(std::fstream& file,int16_t value){
    int16_t val=htobe16(value);
    file.write(reinterpret_cast<char*>(&val),sizeof(int16_t));
}
void write_bin_int32(std::fstream& file,int32_t value){
    int32_t val=htobe32(value);
    file.write(reinterpret_cast<char*>(&val),sizeof(int32_t));
}
void write_bin_int64(std::fstream& file,int64_t value){
    int64_t val=htobe64(value);
    file.write(reinterpret_cast<char*>(&val),sizeof(int64_t));
}

void write_bin_shape(std::fstream& file,const Shape& shape){
    write_bin_uint64(file,shape.dim());
    for(size_t i=0;i<shape.dim();i++){
        write_bin_uint64(file,shape[i]);
    }
}

void write_bin_tensor(std::fstream& file,const Tensor& tensor,TensorType type){
    write_bin_shape(file,tensor.shape());
    write_bin_uint8(file,static_cast<uint8_t>(type));
    const double* data=tensor.data();
    size_t count=tensor.count();
    switch(type){
        case TensorType::INT8:
            for(size_t i=0;i<count;i++){
                write_bin_int8(file,static_cast<int8_t>(data[i]));
            }
            break;
        case TensorType::INT16:
            for(size_t i=0;i<count;i++){
                write_bin_int16(file,static_cast<int16_t>(data[i]));
            }
            break;
        case TensorType::INT32:
            for(size_t i=0;i<count;i++){
                write_bin_int32(file,static_cast<int32_t>(data[i]));
            }
            break;
        case TensorType::INT64:
            for(size_t i=0;i<count;i++){
                write_bin_int64(file,static_cast<int64_t>(data[i]));
            }
            break;
        case TensorType::UINT8:
            for(size_t i=0;i<count;i++){
                write_bin_uint8(file,static_cast<uint8_t>(data[i]));
            }
            break;
        case TensorType::UINT16:
            for(size_t i=0;i<count;i++){
                write_bin_uint16(file,static_cast<uint16_t>(data[i]));
            }
            break;
        case TensorType::UINT32:
            for(size_t i=0;i<count;i++){
                write_bin_uint32(file,static_cast<uint32_t>(data[i]));
            }
            break;
        case TensorType::UINT64:
            for(size_t i=0;i<count;i++){
                write_bin_uint64(file,static_cast<uint64_t>(data[i]));
            }
            break;
        case TensorType::FLOAT:
            for(size_t i=0;i<count;i++){
                write_bin_float(file,static_cast<float>(data[i]));
            }
            break;
        case TensorType::DOUBLE:
            for(size_t i=0;i<count;i++){
                write_bin_double(file,data[i]);
            }
            break;
    }
}
void write_bin_bool_tensor(std::fstream& file,const BoolTensor& tensor){
    write_bin_shape(file,tensor.shape());
    size_t count=tensor.count();
    const bool* data=tensor.data();       
    size_t byte_count=(count+7)/8;
    for(size_t i=0;i<byte_count;i++){
        uint8_t byte=0;
        for(size_t j=0;j<8;j++){
            if(i*8+j<count){
                byte|=(data[i*8+j]?1:0)<<j;
            }
        }
        write_bin_uint8(file,byte);
    }
}
void write_bin_str(std::fstream& file,std::string value){
    write_bin_uint64(file,value.size());
    file.write(value.c_str(),value.size());
}

double read_bin_double(std::fstream& file){
    double val;
    file.read(reinterpret_cast<char*>(&val),sizeof(double));
    return handle_endian_d(val);
}
float read_bin_float(std::fstream& file){
    float val;
    file.read(reinterpret_cast<char*>(&val),sizeof(float));
    return handle_endian_f(val);
}


uint8_t read_bin_uint8(std::fstream& file){
    uint8_t val;
    file.read(reinterpret_cast<char*>(&val),sizeof(uint8_t));
    return val;
}
uint16_t read_bin_uint16(std::fstream& file){
    uint16_t val;
    file.read(reinterpret_cast<char*>(&val),sizeof(uint16_t));
    return be16toh(val);
}
uint32_t read_bin_uint32(std::fstream& file){
    uint32_t val;
    file.read(reinterpret_cast<char*>(&val),sizeof(uint32_t));
    return be32toh(val);
}
uint64_t read_bin_uint64(std::fstream& file){
    uint64_t val;
    file.read(reinterpret_cast<char*>(&val),sizeof(uint64_t));
    return be64toh(val);
}


int8_t read_bin_int8(std::fstream& file){
    int8_t val;
    file.read(reinterpret_cast<char*>(&val),sizeof(int8_t));
    return val;
}
int16_t read_bin_int16(std::fstream& file){
    int16_t val;
    file.read(reinterpret_cast<char*>(&val),sizeof(int16_t));
    return be16toh(val);
}
int32_t read_bin_int32(std::fstream& file){
    int32_t val;
    file.read(reinterpret_cast<char*>(&val),sizeof(int32_t));
    return be32toh(val);
}
int64_t read_bin_int64(std::fstream& file){
    int64_t val;
    file.read(reinterpret_cast<char*>(&val),sizeof(int64_t));
    return be64toh(val);
}


std::string read_bin_str(std::fstream& file){
    uint64_t size=read_bin_uint64(file);
    char* buffer=new char[size];
    file.read(buffer,size);
    std::string val(buffer,size);
    delete[] buffer;
    return val;
}
Shape read_bin_shape(std::fstream& file){
    uint64_t dim=read_bin_uint64(file);
    size_t* shape=new size_t[dim];
    for(size_t i=0;i<dim;i++){
        shape[i]=read_bin_uint64(file);
    }
    return Shape(dim,shape);
}
Tensor read_bin_tensor(std::fstream& file){
    Shape shape=read_bin_shape(file);
    TensorType type=(TensorType)read_bin_uint8(file);
    size_t count=shape.count();
    Tensor tensor(shape);
    double* data=tensor.data();
    switch(type){
        case TensorType::INT8:
            for(size_t i=0;i<count;i++){
                data[i]=read_bin_int8(file);
            }
            break;
        case TensorType::INT16:
            for(size_t i=0;i<count;i++){
                data[i]=read_bin_int16(file);
            }
            break;
        case TensorType::INT32:
            for(size_t i=0;i<count;i++){
                data[i]=read_bin_int32(file);
            }
            break;
        case TensorType::INT64:
            for(size_t i=0;i<count;i++){
                data[i]=read_bin_int64(file);
            }
            break;
        case TensorType::UINT8:
            for(size_t i=0;i<count;i++){
                data[i]=read_bin_uint8(file);
            }
            break;
        case TensorType::UINT16:
            for(size_t i=0;i<count;i++){
                data[i]=read_bin_uint16(file);
            }
            break;
        case TensorType::UINT32:
            for(size_t i=0;i<count;i++){
                data[i]=read_bin_uint32(file);
            }
            break;
        case TensorType::UINT64:
            for(size_t i=0;i<count;i++){
                data[i]=read_bin_uint64(file);
            }
            break;
        case TensorType::FLOAT:
            for(size_t i=0;i<count;i++){
                data[i]=read_bin_float(file);
            }
            break;
        case TensorType::DOUBLE:
            for(size_t i=0;i<count;i++){
                data[i]=read_bin_double(file);
            }
            break;
    }
    return tensor;
}
BoolTensor read_bin_bool_tensor(std::fstream& file){
    Shape shape=read_bin_shape(file);
    size_t count=shape.count();
    size_t byte_count=(count+7)/8;
    BoolTensor tensor(shape);
    bool* data=tensor.data();
    for(size_t i=0;i<byte_count;i++){
        uint8_t byte=read_bin_uint8(file);
        for(size_t j=0;j<8;j++){
            if(i*8+j<count){
                data[i*8+j]=(byte&(1<<j))==1;
            }
        }
    }
    return tensor;
}
}
}
