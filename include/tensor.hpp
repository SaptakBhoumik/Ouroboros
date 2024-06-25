#pragma once
#include "shape.hpp"
#include <memory>
#include <functional>
#include <vector>   
namespace Ouroboros{
//TODO:bool tensor
class Tensor{
    double* m_data=nullptr;
    Shape m_shape;
    Shape m_strides={1};
    public:
    Tensor(const Shape& shape);
    Tensor(const Shape& shape,double value);
    //We dont copy the data and instead this data is shared
    //So the user should not delete the data/use it
    Tensor(const Shape& shape,double* data);
    Tensor(const Tensor& tensor);
    Tensor(Tensor&& tensor);

    Tensor& operator=(const Tensor& tensor);
    Tensor& operator=(Tensor&& tensor);

    void reshape(const Shape& shape);
    //Modify the tensor data but not the shape
    void fill(double value);
    void fill(std::function<double()> func);
    void zeros();
    void ones();
    void rand(double start=0.0,double end=1.0);

    //Clean all values < a and set them to new_val
    void clean(double a=0.0,double new_val=0.0);
    //Clip all values to be in the range [a,b] i.e if a>x then x=a and if x>b then x=b else x=x
    void clamp(double a,double b);
    //Clip all values to be in the range {a,b,c} i.e if a>x then x=a and if x>b then x=b else x=c
    void clamp(double a,double b,double c);
    //Replace all values a with b
    void replace(double a,double b);

    void fill_nan(double value=0.0);
    void fill_inf(double value=0.0);
    void fill_neg_inf(double value=0.0);

    void fill_nan_inf(double value=0.0);
    void fill_nan_neg_inf(double value=0.0);
    void fill_inf_neg_inf(double value=0.0);

    void fill_nan_inf_neg_inf(double value=0.0);    

    bool is_zero();
    bool is_finite();
    bool has_nan();

    __always_inline double& operator[](size_t index){
        #ifdef __OUROBOROS_CHECK__
        if(index>=m_shape.count()){
            throw std::invalid_argument("Invalid index");
        }
        #endif
        return m_data[index];
    }
    __always_inline const double& operator[](size_t index) const{
        #ifdef __OUROBOROS_CHECK__
        if(index>=m_shape.count()){
            throw std::invalid_argument("Invalid index");
        }
        #endif
        return m_data[index];
    }
    __always_inline double& operator[](const Shape& index){
        #ifdef __OUROBOROS_CHECK__
        if(index.dim()!=m_shape.dim()){
            throw std::invalid_argument("Invalid index");
        }
        for(size_t i=0;i<index.dim();++i){
            if(index[i]>=m_shape[i]){
                throw std::invalid_argument("Invalid index");
            }
        }
        #endif
        size_t idx=0;
        for(size_t i=0;i<m_strides.dim();++i){
            idx+=m_strides[i]*index[i];
        }
        return m_data[idx];
    }
    __always_inline const double& operator[](const Shape& index) const{
        #ifdef __OUROBOROS_CHECK__
        if(index.dim()!=m_shape.dim()){
            throw std::invalid_argument("Invalid index");
        }
        for(size_t i=0;i<index.dim();++i){
            if(index[i]>=m_shape[i]){
                throw std::invalid_argument("Invalid index");
            }
        }
        #endif
        size_t idx=0;
        for(size_t i=0;i<m_shape.dim();++i){
            idx=idx*m_shape[i]+index[i];
        }
        return m_data[idx];
    }
    
    double* data();
    const double* data() const;
    const Shape& shape() const;
    const Shape& strides() const;
    size_t count() const;
    size_t dim() const;

    ~Tensor();
};
std::ostream& operator<<(std::ostream& os,const Tensor& tensor);
namespace CreateTensor{
Tensor zeros(const Shape& shape);
Tensor ones(const Shape& shape);
Tensor rand(const Shape& shape,double start=0.0,double end=1.0);
Tensor fill(const Shape& shape,double value);
Tensor fill(const Shape& shape,std::function<double()> func);
Tensor linspace(const Shape& shape,double start,double end);	 
//base^start,base^(start+step),base^(start+2*step),...,base^end
//step=(end-start)/(count-1)
Tensor logspace(const Shape& shape,double start,double end,double base=10.0);	
Tensor scalar_matrix(size_t col_count,double value);
Tensor diagonal_matrix(std::vector<double> diag);
}
}