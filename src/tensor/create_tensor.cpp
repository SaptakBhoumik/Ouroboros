#include "tensor.hpp"
#include <random>
namespace Ouroboros{
Tensor zeros(const Shape& shape){
    return fill(shape,0.0);
}
Tensor ones(const Shape& shape){
    return fill(shape,1.0);
}
Tensor rand(const Shape& shape,double start,double end){
    std::random_device __dev;
    std::mt19937 __rng(__dev());
    std::uniform_real_distribution<std::mt19937::result_type> dist(start,end);
    size_t count=shape.count();
    if(count==0){
        throw std::invalid_argument("Invalid shape");
    }
    double* data=new double[count];
    for(size_t i=0;i<count;++i){
        data[i]=dist(__rng);
    }
    return Tensor(shape,data);
}
Tensor fill(const Shape& shape,double value){
    size_t count=shape.count();
    if(count==0){
        throw std::invalid_argument("Invalid shape");
    }
    double* data=new double[count];
    for(size_t i=0;i<count;++i){
        data[i]=value;
    }
    return Tensor(shape,data);
}
Tensor fill(const Shape& shape,std::function<double()> func){
    size_t count=shape.count();
    if(count==0){
        throw std::invalid_argument("Invalid shape");
    }
    double* data=new double[count];
    for(size_t i=0;i<count;++i){
        data[i]=func();
    }
    return Tensor(shape,data);
}
Tensor linspace(const Shape& shape,double start,double end){
    size_t count=shape.count();
    if(count==0){
        throw std::invalid_argument("Invalid shape");
    }
    else if(count==1){
        return Tensor(shape,start);
    }
    double* data=new double[count];
    double step=(end-start)/(count-1);
    for(size_t i=0;i<count;++i){
        data[i]=start+i*step;
    }
    return Tensor(shape,data);
}
Tensor logspace(const Shape& shape,double start,double end,double base){
    size_t count=shape.count();
    if(count==0){
        throw std::invalid_argument("Invalid shape");
    }
    else if(count==1){
        return Tensor(shape,start);
    }
    double* data=new double[count];
    double step=(end-start)/(count-1);
    for(size_t i=0;i<count;++i){
        double exponent=start+i*step;
        data[i]=std::pow(base,exponent);
    }
    return Tensor(shape,data);
}
}