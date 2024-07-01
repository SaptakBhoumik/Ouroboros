#pragma once
#include "shape.hpp"
#include <memory>
#include <functional>
#include <vector>   
namespace Ouroboros{
class BoolTensor{
    bool* m_data=nullptr;
    Shape m_shape;
    Shape m_strides={1};
    void sliceRecursive(BoolTensor& output, const Shape& start, const Shape& step, Shape& indices, 
                        Shape& output_indices, size_t dimension);
    public:
    BoolTensor(const Shape& shape);
    BoolTensor(const Shape& shape,bool value);
    //We dont copy the data and instead this data is shared
    //So the user should not delete the data/use it
    BoolTensor(const Shape& shape,bool* data);
    BoolTensor(const BoolTensor& tensor);
    BoolTensor(BoolTensor&& tensor);

    BoolTensor& operator=(const BoolTensor& tensor);
    BoolTensor& operator=(BoolTensor&& tensor);

    void reshape(const Shape& shape);

    __always_inline bool& operator[](size_t index){
        #ifdef __OUROBOROS_CHECK__
        if(index>=m_shape.count()){
            throw std::invalid_argument("Invalid index");
        }
        #endif
        return m_data[index];
    }
    __always_inline const bool& operator[](size_t index) const{
        #ifdef __OUROBOROS_CHECK__
        if(index>=m_shape.count()){
            throw std::invalid_argument("Invalid index");
        }
        #endif
        return m_data[index];
    }
    __always_inline size_t offset(const Shape& index) const{
        size_t idx=0;
        for(size_t i=0;i<m_strides.dim();i++){
            idx+=m_strides[i]*index[i];
        }
        return idx;
    }
    __always_inline bool& operator[](const Shape& index){
        #ifdef __OUROBOROS_CHECK__
        if(index.dim()!=m_shape.dim()){
            throw std::invalid_argument("Invalid index");
        }
        size_t idx=0;
        for(size_t i=0;i<index.dim();i++){
            if(index[i]>=m_shape[i]){
                throw std::invalid_argument("Invalid index");
            }
            idx+=m_strides[i]*index[i];
        }
        return m_data[idx];
        #else
        return m_data[offset(index)];
        #endif
    }
    __always_inline const bool& operator[](const Shape& index) const{
        #ifdef __OUROBOROS_CHECK__
        if(index.dim()!=m_shape.dim()){
            throw std::invalid_argument("Invalid index");
        }
        size_t idx=0;
        for(size_t i=0;i<index.dim();i++){
            if(index[i]>=m_shape[i]){
                throw std::invalid_argument("Invalid index");
            }
            idx+=m_strides[i]*index[i];
        }
        return m_data[idx];
        #else
        return m_data[offset(index)];
        #endif
    }

    __always_inline BoolTensor slice(const Shape& start,const Shape& end,const Shape& step){
        if (start.dim() != m_shape.dim() || end.dim() != m_shape.dim() || step.dim() != m_shape.dim()) {
            throw std::invalid_argument("Start, end, and step vectors must have the same length as the number of dimensions in the tensor");
        }

        // Calculate the shape of the sliced tensor
        size_t* data=new size_t[m_shape.dim()];
        for (size_t i = 0; i < m_shape.dim(); ++i) {
            data[i] = (end[i] - start[i] + step[i] - 1) / step[i];
        }
        Shape output_shape(m_shape.dim(),data);

        BoolTensor output(output_shape);

        size_t i=0;
        Shape indices(m_shape.dim(), i);
        Shape output_indices(m_shape.dim(), i);
        sliceRecursive(output, start, step, indices, output_indices, 0);

        return output;
    }
    __always_inline BoolTensor slice(const Shape& start,const Shape& end,size_t step=1){
        return slice(start,end,Shape(m_shape.dim(),step));
    }

    bool* data();
    const bool* data() const;
    Shape shape() const;
    Shape strides() const;
    size_t count() const;
    size_t dim() const;

    ~BoolTensor();
};
std::ostream& operator<<(std::ostream& os,const BoolTensor& tensor);
class Tensor{
    double* m_data=nullptr;
    Shape m_shape;
    Shape m_strides={1};
    void sliceRecursive(Tensor& output, const Shape& start, const Shape& step, Shape& indices, 
                        Shape& output_indices, size_t dimension);
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
    void flatten();
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
    void threshold(double a,double new_val=0.0);
    //Replace all values a with b
    void replace(double a,double b);

    void fill_nan(double value=0.0);
    void fill_inf(double value=0.0);
    void fill_neg_inf(double value=0.0);

    void fill_nan_inf(double value=0.0);
    void fill_nan_neg_inf(double value=0.0);
    void fill_inf_neg_inf(double value=0.0);

    void fill_nan_inf_neg_inf(double value=0.0);    

    bool is_zero()const;
    bool is_finite()const;
    bool has_nan()const;

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
    __always_inline size_t offset(const Shape& index) const{
        size_t idx=0;
        for(size_t i=0;i<m_strides.dim();i++){
            idx+=m_strides[i]*index[i];
        }
        return idx;
    }
    __always_inline double& operator[](const Shape& index){
        #ifdef __OUROBOROS_CHECK__
        if(index.dim()!=m_shape.dim()){
            throw std::invalid_argument("Invalid index");
        }
        size_t idx=0;
        for(size_t i=0;i<index.dim();i++){
            if(index[i]>=m_shape[i]){
                throw std::invalid_argument("Invalid index");
            }
            idx+=m_strides[i]*index[i];
        }
        return m_data[idx];
        #else
        return m_data[offset(index)];
        #endif
    }
    __always_inline const double& operator[](const Shape& index) const{
        #ifdef __OUROBOROS_CHECK__
        if(index.dim()!=m_shape.dim()){
            throw std::invalid_argument("Invalid index");
        }
        size_t idx=0;
        for(size_t i=0;i<index.dim();i++){
            if(index[i]>=m_shape[i]){
                throw std::invalid_argument("Invalid index");
            }
            idx+=m_strides[i]*index[i];
        }
        return m_data[idx];
        #else
        return m_data[offset(index)];
        #endif
    }
    __always_inline Tensor slice(const Shape& start,const Shape& end,const Shape& step){
        if (start.dim() != m_shape.dim() || end.dim() != m_shape.dim() || step.dim() != m_shape.dim()) {
            throw std::invalid_argument("Start, end, and step must have the same length as the number of dimensions in the tensor");
        }

        // Calculate the shape of the sliced tensor
        size_t* data=new size_t[m_shape.dim()];
        for (size_t i = 0; i < m_shape.dim(); ++i) {
            data[i] = (end[i] - start[i] + step[i] - 1) / step[i];
        }
        Shape output_shape(m_shape.dim(),data);

        Tensor output(output_shape);

        size_t i=0;
        Shape indices(m_shape.dim(), i);
        Shape output_indices(m_shape.dim(), i);
        sliceRecursive(output, start, step, indices, output_indices, 0);

        return output;
    }
    __always_inline Tensor slice(const Shape& start,const Shape& end,size_t step=1){
        return slice(start,end,Shape(m_shape.dim(),step));
    }
    double* data();
    const double* data() const;
    Shape shape() const;
    Shape strides() const;
    size_t count() const;
    size_t dim() const;

    double norm()const;
    double norm2()const;
    double sum()const;
    double prod()const;
    void normalize();

    ~Tensor();
};
std::ostream& operator<<(std::ostream& os,const Tensor& tensor);
Shape getStride(const Shape& shape);
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
//If condition is true then x else y
Tensor where(const BoolTensor& condition,const Tensor& x,const Tensor& y);
Tensor where(const BoolTensor& condition,const Tensor& x,double y);
Tensor where(const BoolTensor& condition,double x,const Tensor& y);
Tensor where(const BoolTensor& condition,double x,double y);
}
}