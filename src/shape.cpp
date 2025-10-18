#include "tensor.hpp"

namespace Ouroboros{
Shape::Shape(std::size_t dim,std::size_t val){
    // if(dim==0){
    //     throw std::invalid_argument("Shape cannot be empty");
    // }
    m_dim=dim;
    m_shape=new std::size_t[m_dim];
    m_strides=new std::size_t[m_dim];
    m_strides[m_dim-1]=1;
    for(std::size_t i=0;i<m_dim;i++){
        m_shape[i]=val;
        m_count*=val;
        if(i<m_dim-1){
            m_strides[i]=m_strides[i+1]*m_shape[i+1];
        }
    }
}
Shape::Shape(std::size_t dim,std::size_t* shape){
    // if(dim==0){
    //     throw std::invalid_argument("Shape cannot be empty");
    // }
    m_dim=dim;
    m_shape=new std::size_t[m_dim];
    m_strides=new std::size_t[m_dim];
    m_strides[m_dim-1]=1;
    for(std::size_t i=0;i<m_dim;i++){
        m_shape[i]=shape[i];
        m_count*=shape[i];
        if(i<m_dim-1){
            m_strides[i]=m_strides[i+1]*m_shape[i+1];
        }
    }
}
Shape::Shape(std::initializer_list<std::size_t> shape){
    m_dim=shape.size();
    // if(m_dim==0){
    //     throw std::invalid_argument("Shape cannot be empty");
    // }
    m_shape=new std::size_t[m_dim];
    m_strides=new std::size_t[m_dim];
    m_strides[m_dim-1]=1;
    m_count=1;
    std::size_t i = 0;
    for(auto it=shape.begin();it!=shape.end();it++){
        m_shape[i]=*it;
        m_count*=m_shape[i];
        if(i<m_dim-1){
            m_strides[i]=m_strides[i+1]*m_shape[i+1];
        }
        i++;
    }
}
Shape::Shape(const Shape& shape){
    m_dim=shape.m_dim;
    m_count=shape.m_count;
    m_shape=new std::size_t[m_dim];
    m_strides=new std::size_t[m_dim];
    for(std::size_t i=0;i<m_dim;i++){
        m_shape[i]=shape.m_shape[i];
        m_strides[i]=shape.m_strides[i];
    }
}
Shape::Shape(Shape&& shape){
    m_dim=shape.m_dim;
    m_count=shape.m_count;
    m_shape=shape.m_shape;
    m_strides=shape.m_strides;
    shape.m_dim=0;
    shape.m_count=1;
    shape.m_shape=nullptr;
    shape.m_strides=nullptr;
}

void Shape::operator=(const Shape& shape){
    if(this==&shape){
        return;
    }
    if(m_dim!=shape.m_dim){
        if(m_shape!=nullptr){
            delete[] m_shape;
        }
        if(m_strides!=nullptr){
            delete[] m_strides;
        }
        m_shape=new std::size_t[m_dim];
        m_strides=new std::size_t[m_dim];
    }
    m_dim=shape.m_dim;
    m_count=shape.m_count;
    for(std::size_t i=0;i<m_dim;i++){
        m_shape[i]=shape.m_shape[i];
        m_strides[i]=shape.m_strides[i];
    }
}
void Shape::operator=(Shape&& shape){
    if(this==&shape){
        return;
    }
    if(m_shape!=nullptr){
        delete[] m_shape;
    }
    if(m_strides!=nullptr){
        delete[] m_strides;
    }
    m_dim=shape.m_dim;
    m_count=shape.m_count;
    m_shape=shape.m_shape;
    m_strides=shape.m_strides;
    shape.m_dim=0;
    shape.m_count=1;
    shape.m_shape=nullptr;
    shape.m_strides=nullptr;
}
void Shape::operator=(std::initializer_list<std::size_t> shape){
    if(m_dim!=shape.size()){
        if(m_shape!=nullptr){
            delete[] m_shape;
        }
        if(m_strides!=nullptr){
            delete[] m_strides;
        }
        m_strides=new std::size_t[shape.size()];
        m_shape=new std::size_t[shape.size()];
        m_strides[shape.size()-1]=1;
        
    }
    m_dim=shape.size();
    m_count=1;
    std::size_t i = 0;
    for(auto it=shape.begin();it!=shape.end();it++){
        m_shape[i]=*it;
        m_count*=m_shape[i];
        if(i<m_dim-1){
            m_strides[i]=m_strides[i+1]*m_shape[i+1];
        }
        i++;
    }
}
const std::size_t* Shape::begin() const{
    return m_shape;
}
const std::size_t* Shape::end() const{
    return m_shape+m_dim;
}

bool Shape::operator==(const Shape& shape) const{
    if(m_dim!=shape.m_dim||m_count!=shape.m_count){
        return false;
    }
    for(std::size_t i=0;i<m_dim;i++){
        if(m_shape[i]!=shape.m_shape[i]){
            return false;
        }
    }
    return true;
}
bool Shape::operator!=(const Shape& shape) const{
    return !(*this==shape);
}

std::size_t Shape::count() const{
    return m_count;
}
std::size_t Shape::dim() const{
    return m_dim;
}

Shape::~Shape(){
    if(m_shape!=nullptr){
        delete[] m_shape;
        m_shape=nullptr;
    }
    if(m_strides!=nullptr){
        delete[] m_strides;
        m_strides=nullptr;
    }
    this->m_dim=0;
    this->m_count=1;
}
std::ostream& operator<<(std::ostream& os,const Shape& shape){
    os<<"[";
    auto ptr=shape.begin();
    for(std::size_t i=0;i<shape.dim();i++){
        os<<*ptr;
        if(i!=shape.dim()-1){
            os<<",";
        }
        ++ptr;
    }
    os<<"]";
    return os;
}
}
