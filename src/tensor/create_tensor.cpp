#include "tensor.hpp"
#include <random>
namespace Ouroboros{
namespace CreateTensor{
Tensor zeros(const Shape& shape){
    return fill(shape,0.0);
}
Tensor ones(const Shape& shape){
    return fill(shape,1.0);
}
Tensor rand(const Shape& shape,double start,double end){
    if(start>end){
        throw std::invalid_argument("Invalid range");
    }
    std::random_device __dev;
    std::mt19937 __rng(__dev());
    std::uniform_real_distribution<double> dist(start,end);
    std::size_t count=shape.count();
    if(count==0){
        throw std::invalid_argument("Invalid shape");
    }
    double* data=new double[count];
    for(std::size_t i=0;i<count;i++){
        data[i]=dist(__rng);
    }
    return Tensor(shape,data);
}
Tensor fill(const Shape& shape,double value){
    std::size_t count=shape.count();
    if(count==0){
        throw std::invalid_argument("Invalid shape");
    }
    double* data=new double[count];
    for(std::size_t i=0;i<count;i++){
        data[i]=value;
    }
    return Tensor(shape,data);
}
Tensor fill(const Shape& shape,std::function<double()> func){
    std::size_t count=shape.count();
    if(count==0){
        throw std::invalid_argument("Invalid shape");
    }
    double* data=new double[count];
    for(std::size_t i=0;i<count;i++){
        data[i]=func();
    }
    return Tensor(shape,data);
}
Tensor linspace(const Shape& shape,double start,double end){
    std::size_t count=shape.count();
    if(count==0){
        throw std::invalid_argument("Invalid shape");
    }
    else if(count==1){
        return Tensor(shape,start);
    }
    double* data=new double[count];
    double step=(end-start)/(count-1);
    for(std::size_t i=0;i<count;i++){
        data[i]=start+i*step;
    }
    return Tensor(shape,data);
}
Tensor logspace(const Shape& shape,double start,double end,double base){
    std::size_t count=shape.count();
    if(count==0){
        throw std::invalid_argument("Invalid shape");
    }
    else if(count==1){
        return Tensor(shape,start);
    }
    double* data=new double[count];
    double step=(end-start)/(count-1);
    for(std::size_t i=0;i<count;i++){
        double exponent=start+i*step;
        data[i]=std::pow(base,exponent);
    }
    return Tensor(shape,data);
}
Tensor scalar_matrix(std::size_t col_count,double value){
    if(col_count==0){
        throw std::invalid_argument("Invalid shape");
    }
    Shape shape={col_count,col_count};
    Tensor tensor(shape,0.0);
    auto d=tensor.data();
    for(std::size_t i=0;i<col_count;i++){
        d[i*col_count+i]=value;
    }
    return tensor;
}
Tensor diagonal_matrix(std::vector<double> diag){
    if(diag.size()==0){
        throw std::invalid_argument("Invalid shape");
    }
    std::size_t col_count=diag.size();
    Tensor tensor({col_count,col_count},0.0);
    auto d=tensor.data();
    for(std::size_t i=0;i<col_count;i++){
        d[i*col_count+i]=diag[i];
    }
    return tensor;
}
Tensor where(const BoolTensor& condition,const Tensor& x,const Tensor& y){
    auto shape=condition.shape();
    if(shape!=x.shape()||shape!=y.shape()){
        throw std::invalid_argument("Invalid shape");
    }
    auto result=Tensor(shape);
    std::size_t count=shape.count();
    double* result_data=result.data();
    const double* x_data=x.data();
    const double* y_data=y.data();
    const bool* condition_data=condition.data();
    for(std::size_t i=0;i<count;i++){
        result_data[i]=condition_data[i]?x_data[i]:y_data[i];
    }
    return result;
}
Tensor where(const BoolTensor& condition,const Tensor& x,double y){
    auto shape=condition.shape();
    if(shape!=x.shape()){
        throw std::invalid_argument("Invalid shape");
    }
    auto result=Tensor(shape);
    std::size_t count=shape.count();
    double* result_data=result.data();
    const double* x_data=x.data();
    const bool* condition_data=condition.data();
    for(std::size_t i=0;i<count;i++){
        result_data[i]=condition_data[i]?x_data[i]:y;
    }
    return result;
}
Tensor where(const BoolTensor& condition,double x,const Tensor& y){
    auto shape=condition.shape();
    if(shape!=y.shape()){
        throw std::invalid_argument("Invalid shape");
    }
    auto result=Tensor(shape);
    std::size_t count=shape.count();
    double* result_data=result.data();
    const double* y_data=y.data();
    const bool* condition_data=condition.data();
    for(std::size_t i=0;i<count;i++){
        result_data[i]=condition_data[i]?x:y_data[i];
    }
    return result;
}
Tensor where(const BoolTensor& condition,double x,double y){
    auto shape=condition.shape();
    auto result=Tensor(shape);
    std::size_t count=shape.count();
    double* result_data=result.data();
    const bool* condition_data=condition.data();
    for(std::size_t i=0;i<count;i++){
        result_data[i]=condition_data[i]?x:y;
    }
    return result;
}
}
}