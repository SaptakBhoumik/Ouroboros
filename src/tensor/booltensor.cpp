#include "tensor.hpp"
namespace Ouroboros{
BoolTensor::BoolTensor(const Shape& shape):m_shape(shape){
    if(m_shape.count()==0){
        throw std::invalid_argument("BoolTensor cannot have 0 elements");
    }
    m_data=new bool[m_shape.count()];
    m_strides = getStride(m_shape);
}
BoolTensor::BoolTensor(const Shape& shape,bool value):m_shape(shape){
    if(m_shape.count()==0){
        throw std::invalid_argument("BoolTensor cannot have 0 elements");
    }
    m_data=new bool[m_shape.count()];
    for(std::size_t i=0;i<m_shape.count();i++){
        m_data[i]=value;
    }
    m_strides = getStride(m_shape);
}
BoolTensor::BoolTensor(const Shape& shape,bool* data):m_shape(shape){
    if(m_shape.count()==0){
        throw std::invalid_argument("BoolTensor cannot have 0 elements");
    }
    m_data=data;
    m_strides = getStride(m_shape);
}
BoolTensor::BoolTensor(const BoolTensor& tensor):m_shape(tensor.m_shape){
    m_data=new bool[m_shape.count()];
    for(std::size_t i=0;i<m_shape.count();i++){
        m_data[i]=tensor.m_data[i];
    }
    m_strides = tensor.m_strides;
}
BoolTensor::BoolTensor(BoolTensor&& tensor):m_shape(tensor.m_shape){
    m_data=tensor.m_data;
    m_strides = tensor.m_strides;
    tensor.m_data=nullptr;
}

BoolTensor& BoolTensor::operator=(const BoolTensor& tensor){
    if(this==&tensor){
        return *this;
    }
    if(m_data!=nullptr){
        if(m_shape.count()!=tensor.m_shape.count()){
            delete[] m_data;
            m_data=new bool[tensor.m_shape.count()];
        }
        m_shape=tensor.m_shape;
        m_strides = tensor.m_strides;
    }
    else{
        m_shape=tensor.m_shape;
        m_data=new bool[m_shape.count()];
        m_strides = tensor.m_strides;
    }
    for(std::size_t i=0;i<m_shape.count();i++){
        m_data[i]=tensor.m_data[i];
    }
    return *this;
}
BoolTensor& BoolTensor::operator=(BoolTensor&& tensor){
    if(this==&tensor){
        return *this;
    }
    if(m_data!=nullptr){
        delete[] m_data;
    }
    m_shape=tensor.m_shape;
    m_strides = tensor.m_strides;
    m_data=tensor.m_data;
    tensor.m_data=nullptr;
    return *this;
}
void BoolTensor::reverse(){
    std::size_t n=m_shape.count();
    for(std::size_t i=0;i<n/2;i++){
        std::swap(m_data[i],m_data[n-i-1]);
    }
}
void BoolTensor::reshape(const Shape& shape){
    if(m_shape.count()!=shape.count()){
        throw std::invalid_argument("Invalid shape for reshape");
    }
    m_shape=shape;
    m_strides = getStride(m_shape);
}
void BoolTensor::flatten(){
    m_shape={m_shape.count()};
    m_strides = {1};
}

bool* BoolTensor::data(){
    return m_data;
}
const bool* BoolTensor::data() const{
    return m_data;
}
Shape BoolTensor::shape() const{
    return m_shape;
}
Shape BoolTensor::strides() const{
    return m_strides;
}
std::size_t BoolTensor::count() const{
    return m_shape.count();
}
std::size_t BoolTensor::dim() const{
    return m_shape.dim();
}
BoolTensor::~BoolTensor(){
    if(m_data!=nullptr){
        delete[] m_data;
        m_data=nullptr;
    }
}
void printTensorRecursively(std::ostream& os,const BoolTensor& tensor, 
                            const Shape& strides, std::size_t dim, std::size_t offset) {
    auto shape=tensor.shape();
    auto data=tensor.data();
    if (dim == shape.dim() - 1) {
        os << "[";
        for (std::size_t i = 0; i < shape[dim]; i++) {
            os << data[offset + i];
            if (i != shape[dim] - 1) {
                os << ", ";
            }
        }
        os << "]";
    } else {
        os << "[";
        for (std::size_t i = 0; i < shape[dim]; i++) {
            printTensorRecursively(os,tensor, strides, dim + 1, offset + i * strides[dim]);
            if (i != shape[dim] - 1) {
                os << ", ";
            }
        }
        os << "]";
    }
}

// Function to print the Tensor object as a multidimensional array
void printTensorAsArray(std::ostream& os , const BoolTensor& tensor) {
    auto shape=tensor.shape();
    
    auto strides = getStride(shape);
    printTensorRecursively(os,tensor, strides, 0, 0);
    os << std::endl;
}

std::ostream& operator<<(std::ostream& os,const BoolTensor& tensor){
    printTensorAsArray(os,tensor);
    return os;
}
}