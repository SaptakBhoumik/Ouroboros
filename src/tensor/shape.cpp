
#include "shape.hpp"
namespace Ouroboros{
Shape::Shape(size_t dim,size_t val){
    if(dim==0){
        throw std::invalid_argument("Shape cannot be empty");
    }
    m_dim=dim;
    m_shape=new size_t[m_dim];
    for(size_t i=0;i<m_dim;i++){
        m_shape[i]=val;
        m_count*=val;
    }
}
Shape::Shape(size_t dim,size_t* shape){
    if(dim==0){
        throw std::invalid_argument("Shape cannot be empty");
    }
    m_dim=dim;
    m_shape=new size_t[m_dim];
    size_t* ptr=m_shape;
    for(size_t i=0;i<m_dim;i++){
        *ptr=shape[i];
        m_count*=shape[i];
        ++ptr;
    }
}
Shape::Shape(std::initializer_list<size_t> shape){
    m_dim=shape.size();
    if(m_dim==0){
        throw std::invalid_argument("Shape cannot be empty");
    }
    m_shape=new size_t[m_dim];
    m_count=1;
    size_t* ptr=m_shape;
    for(auto it=shape.begin();it!=shape.end();it++){
        *ptr=*it;
        m_count*=(*it);
        ++ptr;
    }
}
Shape::Shape(const Shape& shape){
    m_dim=shape.m_dim;
    m_count=shape.m_count;
    m_shape=new size_t[m_dim];
    size_t* ptr=m_shape;
    for(size_t i=0;i<m_dim;i++){
        *ptr=shape.m_shape[i];
        ++ptr;
    }
}
Shape::Shape(Shape&& shape){
    m_dim=shape.m_dim;
    m_count=shape.m_count;
    m_shape=shape.m_shape;
    shape.m_dim=0;
    shape.m_count=1;
    shape.m_shape=nullptr;
}

Shape& Shape::operator=(const Shape& shape){
    if(this==&shape){
        return *this;
    }
    if(m_shape!=nullptr){
        delete[] m_shape;
    }
    m_dim=shape.m_dim;
    m_count=shape.m_count;
    m_shape=new size_t[m_dim];
    size_t* ptr=m_shape;
    for(size_t i=0;i<m_dim;i++){
        *ptr=shape.m_shape[i];
        ++ptr;
    }
    return *this;
}
Shape& Shape::operator=(Shape&& shape){
    if(this==&shape){
        return *this;
    }
    if(m_shape!=nullptr){
        delete[] m_shape;
    }
    m_dim=shape.m_dim;
    m_count=shape.m_count;
    m_shape=shape.m_shape;
    shape.m_dim=0;
    shape.m_count=1;
    shape.m_shape=nullptr;
    return *this;
}
Shape& Shape::operator=(std::initializer_list<size_t> shape){
    if(m_shape!=nullptr){
        delete[] m_shape;
    }
    m_dim=shape.size();
    if(m_dim==0){
        throw std::invalid_argument("Shape cannot be empty");
    }
    m_shape=new size_t[m_dim];
    m_count=1;
    size_t* ptr=m_shape;
    for(auto it=shape.begin();it!=shape.end();it++){
        *ptr=*it;
        m_count*=(*it);
        ++ptr;
    }
    return *this;
}
const size_t* Shape::begin() const{
    return m_shape;
}
const size_t* Shape::end() const{
    return m_shape+m_dim;
}

bool Shape::operator==(const Shape& shape) const{
    if(m_dim!=shape.m_dim||m_count!=shape.m_count){
        return false;
    }
    for(size_t i=0;i<m_dim;i++){
        if(m_shape[i]!=shape.m_shape[i]){
            return false;
        }
    }
    return true;
}
bool Shape::operator!=(const Shape& shape) const{
    return !(*this==shape);
}

size_t Shape::count() const{
    return m_count;
}
size_t Shape::dim() const{
    return m_dim;
}

Shape::~Shape(){
    if(m_shape!=nullptr){
        delete[] m_shape;
        m_shape=nullptr;
    }
    this->m_dim=0;
    this->m_count=1;
}
std::ostream& operator<<(std::ostream& os,const Shape& shape){
    os<<"[";
    auto ptr=shape.begin();
    for(size_t i=0;i<shape.dim();i++){
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
