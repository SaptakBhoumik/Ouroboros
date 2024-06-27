#include "tensor.hpp"
namespace Ouroboros{
Shape getStride(const Shape& shape){
    size_t* strides_data=new size_t[shape.dim()];
    size_t stride = 1;
    for (int i = shape.dim() - 1; i >= 0; --i) {
        strides_data[i] = stride;
        stride *= shape[i];
    }
    auto strides=Shape(shape.dim(), strides_data);
    delete[] strides_data;
    return strides;
}


Tensor::Tensor(const Shape& shape):m_shape(shape){
    if(m_shape.count()==0){
        throw std::invalid_argument("Tensor cannot have 0 elements");
    }
    m_data=new double[m_shape.count()];
    m_strides = getStride(m_shape);
}
Tensor::Tensor(const Shape& shape,double value):m_shape(shape){
    if(m_shape.count()==0){
        throw std::invalid_argument("Tensor cannot have 0 elements");
    }
    m_data=new double[m_shape.count()];
    for(size_t i=0;i<m_shape.count();++i){
        m_data[i]=value;
    }
    m_strides = getStride(m_shape);
}
Tensor::Tensor(const Shape& shape,double* data):m_shape(shape){
    if(m_shape.count()==0){
        throw std::invalid_argument("Tensor cannot have 0 elements");
    }
    m_data=data;
    m_strides = getStride(m_shape);
}
Tensor::Tensor(const Tensor& tensor):m_shape(tensor.m_shape){
    m_data=new double[m_shape.count()];
    for(size_t i=0;i<m_shape.count();++i){
        m_data[i]=tensor.m_data[i];
    }
    m_strides = getStride(m_shape);
}
Tensor::Tensor(Tensor&& tensor):m_shape(tensor.m_shape){
    m_data=tensor.m_data;
    m_strides = tensor.m_strides;
    tensor.m_data=nullptr;
}


Tensor& Tensor::operator=(const Tensor& tensor){
    if(this==&tensor){
        return *this;
    }
    if(m_data!=nullptr){
        if(m_shape.count()!=tensor.m_shape.count()){
            delete[] m_data;
            m_data=new double[tensor.m_shape.count()];
        }
        m_shape=tensor.m_shape;
    }
    else{
        m_shape=tensor.m_shape;
        m_data=new double[m_shape.count()];
    }
    for(size_t i=0;i<m_shape.count();++i){
        m_data[i]=tensor.m_data[i];
    }
    return *this;
}
Tensor& Tensor::operator=(Tensor&& tensor){
    if(this==&tensor){
        return *this;
    }
    if(m_data!=nullptr){
        delete[] m_data;
    }
    m_shape=tensor.m_shape;
    m_data=tensor.m_data;
    tensor.m_data=nullptr;
    return *this;
}

void Tensor::reshape(const Shape& shape){
    if(m_shape.count()!=shape.count()){
        throw std::invalid_argument("Invalid shape for reshape");
    }
    m_shape=shape;
    m_strides = getStride(m_shape);
}

double* Tensor::data(){
    return m_data;
}
const double* Tensor::data() const{
    return m_data;
}
Shape Tensor::shape() const{
    return m_shape;
}
Shape Tensor::strides() const{
    return m_strides;
}
size_t Tensor::count() const{
    return m_shape.count();
}
size_t Tensor::dim() const{
    return m_shape.dim();
}


Tensor::~Tensor(){
    if(m_data!=nullptr){
        delete[] m_data;
        m_data=nullptr;
    }
}
void printTensorRecursively(std::ostream& os,const Tensor& tensor, 
                            const Shape& strides, size_t dim, size_t offset) {
    auto shape=tensor.shape();
    auto data=tensor.data();
    if (dim == shape.dim() - 1) {
        os << "[";
        for (size_t i = 0; i < shape[dim]; ++i) {
            os << data[offset + i];
            if (i != shape[dim] - 1) {
                os << ", ";
            }
        }
        os << "]";
    } else {
        os << "[";
        for (size_t i = 0; i < shape[dim]; ++i) {
            printTensorRecursively(os,tensor, strides, dim + 1, offset + i * strides[dim]);
            if (i != shape[dim] - 1) {
                os << ", ";
            }
        }
        os << "]";
    }
}

// Function to print the Tensor object as a multidimensional array
void printTensorAsArray(std::ostream& os , const Tensor& tensor) {
    auto shape=tensor.shape();
    
    auto strides = getStride(shape);
    printTensorRecursively(os,tensor, strides, 0, 0);
    os << std::endl;
}

std::ostream& operator<<(std::ostream& os,const Tensor& tensor){
    printTensorAsArray(os,tensor);
    return os;
}
}
