#include "func/func.hpp"
namespace Ouroboros{
Tensor abs(const Tensor& t,size_t min_count){
    return transform(Scalar::abs,min_count,t);
}

Tensor exp(const Tensor& t,size_t min_count){
    return transform(Scalar::exp,min_count,t);
}

Tensor ln(const Tensor& t,size_t min_count){
    return transform(Scalar::ln,min_count,t);
}
Tensor log10(const Tensor& t,size_t min_count){
    return transform(Scalar::log10,min_count,t);
}
Tensor log2(const Tensor& t,size_t min_count){
    return transform(Scalar::log2,min_count,t);
}
Tensor log(const Tensor& t,double base,size_t min_count){
    double b=Scalar::ln(base);
    const auto function=[b](double x){return Scalar::ln(x)/b;};
    return transform(function,min_count,t);
}
Tensor log(double x,const Tensor& base,size_t min_count){
    double a=Scalar::ln(x);
    const auto function=[a](double x){return a/Scalar::ln(x);};
    return transform(function,min_count,base);
}
Tensor log(const Tensor& t,const Tensor& base,size_t min_count){
    return transform(Scalar::log,min_count,t,base);
}

Tensor cbrt(const Tensor& t,size_t min_count){
    return transform(Scalar::cbrt,min_count,t);
}
Tensor sqrt(const Tensor& t,size_t min_count){
    return transform(Scalar::sqrt,min_count,t);
}
Tensor pow(const Tensor& t,double y,size_t min_count){
    auto function=[y](double x){return Scalar::pow(x,y);};
    return transform(function,min_count,t);
}
Tensor pow(double x,const Tensor& y,size_t min_count){
    auto function=[x](double y){return Scalar::pow(x,y);};
    return transform(function,min_count,y);
}
Tensor pow(const Tensor& t,const Tensor& y,size_t min_count){
    return transform(Scalar::pow,min_count,t,y);
}

Tensor hypot2(const Tensor& x,double y,size_t min_count){
    auto function=[y](double x){return Scalar::hypot2(x,y);};
    return transform(function,min_count,x);
}
Tensor hypot2(double x,const Tensor& y,size_t min_count){
    auto function=[x](double y){return Scalar::hypot2(x,y);};
    return transform(function,min_count,y);
}
Tensor hypot2(const Tensor& x,const Tensor& y,size_t min_count){
    return transform(Scalar::hypot2,min_count,x,y);
}

Tensor hypot3(const Tensor& x,double y,double z,size_t min_count){
    auto function=[y,z](double x){return Scalar::hypot3(x,y,z);};
    return transform(function,min_count,x);
}
Tensor hypot3(double x,const Tensor& y,double z,size_t min_count){
    auto function=[x,z](double y){return Scalar::hypot3(x,y,z);};
    return transform(function,min_count,y);
}
Tensor hypot3(double x,double y,const Tensor& z,size_t min_count){
    auto function=[x,y](double z){return Scalar::hypot3(x,y,z);};
    return transform(function,min_count,z);
}
Tensor hypot3(const Tensor& x,const Tensor& y,double z,size_t min_count){
    auto function=[z](double x,double y){return Scalar::hypot3(x,y,z);};
    return transform(function,min_count,x,y);
}
Tensor hypot3(const Tensor& x,double y,const Tensor& z,size_t min_count){
    auto function=[y](double x,double z){return Scalar::hypot3(x,y,z);};
    return transform(function,min_count,x,z);
}
Tensor hypot3(double x,const Tensor& y,const Tensor& z,size_t min_count){
    auto function=[x](double y,double z){return Scalar::hypot3(x,y,z);};
    return transform(function,min_count,y,z);
}
Tensor hypot3(const Tensor& x,const Tensor& y,const Tensor& z,size_t min_count){
    return transform(Scalar::hypot3,min_count,x,y,z);
}

Tensor ceil(const Tensor& t,size_t min_count){
    return transform(Scalar::ceil,min_count,t);
}
Tensor floor(const Tensor& t,size_t min_count){
    return transform(Scalar::floor,min_count,t);
}
Tensor trunc(const Tensor& t,size_t min_count){
    return transform(Scalar::trunc,min_count,t);
}
Tensor nearbyint(const Tensor& t,size_t min_count){
    return transform(Scalar::nearbyint,min_count,t);
}
Tensor rint(const Tensor& t,size_t min_count){
    return transform(Scalar::rint,min_count,t);
}
Tensor round(const Tensor& t,size_t min_count){
    return transform(Scalar::round,min_count,t);
}
Tensor fmod(const Tensor& x,double y,size_t min_count){
    auto function=[y](double x){return Scalar::fmod(x,y);};
    return transform(function,min_count,x);
}
Tensor fmod(double x,const Tensor& y,size_t min_count){
    auto function=[x](double y){return Scalar::fmod(x,y);};
    return transform(function,min_count,y);
}
Tensor fmod(const Tensor& x,const Tensor& y,size_t min_count){
    return transform(Scalar::fmod,min_count,x,y);
}

Tensor min(const Tensor& x,double y,size_t min_count){
    auto function=[y](double x){return Scalar::min(x,y);};
    return transform(function,min_count,x);
}
Tensor min(double x,const Tensor& y,size_t min_count){
    auto function=[x](double y){return Scalar::min(x,y);};
    return transform(function,min_count,y);
}
Tensor min(const Tensor& x,const Tensor& y,size_t min_count){
    return transform(Scalar::min,min_count,x,y);
}
Tensor max(const Tensor& x,double y,size_t min_count){
    auto function=[y](double x){return Scalar::max(x,y);};
    return transform(function,min_count,x);
}
Tensor max(double x,const Tensor& y,size_t min_count){
    auto function=[x](double y){return Scalar::max(x,y);};
    return transform(function,min_count,y);
}
Tensor max(const Tensor& x,const Tensor& y,size_t min_count){
    return transform(Scalar::max,min_count,x,y);
}

Tensor clamp(const Tensor& x,double min,double max,size_t min_count){
    auto function=[min,max](double x){return Scalar::clamp(x,min,max);};
    return transform(function,min_count,x);
}
Tensor clamp(double x,const Tensor& min,double max,size_t min_count){
    auto function=[x,max](double min){return Scalar::clamp(x,min,max);};
    return transform(function,min_count,min);
}
Tensor clamp(double x,double min,const Tensor& max,size_t min_count){
    auto function=[x,min](double max){return Scalar::clamp(x,min,max);};
    return transform(function,min_count,max);
}
Tensor clamp(const Tensor& x,const Tensor& min,double max,size_t min_count){
    auto function=[max](double x,double min){return Scalar::clamp(x,min,max);};
    return transform(function,min_count,x,min);
}
Tensor clamp(const Tensor& x,double min,const Tensor& max,size_t min_count){
    auto function=[min](double x,double max){return Scalar::clamp(x,min,max);};
    return transform(function,min_count,x,max);
}
Tensor clamp(double x,const Tensor& min,const Tensor& max,size_t min_count){
    auto function=[x](double min,double max){return Scalar::clamp(x,min,max);};
    return transform(function,min_count,min,max);
}
Tensor clamp(const Tensor& x,const Tensor& min,const Tensor& max,size_t min_count){
    auto function=[](double x,double min,double max){return Scalar::clamp(x,min,max);};//Overloaded function so we have to do this
    return transform(function,min_count,x,min,max);
}

Tensor clamp(const Tensor& x,double min,double max,double c,size_t min_count){
    auto function=[min,max,c](double x){return Scalar::clamp(x,min,max,c);};
    return transform(function,min_count,x);
}
Tensor clamp(double x,const Tensor& min,double max,double c,size_t min_count){
    auto function=[x,max,c](double min){return Scalar::clamp(x,min,max,c);};
    return transform(function,min_count,min);
}
Tensor clamp(double x,double min,const Tensor& max,double c,size_t min_count){
    auto function=[x,min,c](double max){return Scalar::clamp(x,min,max,c);};
    return transform(function,min_count,max);
}
Tensor clamp(double x,double min,double max,const Tensor& c,size_t min_count){
    auto function=[x,min,max](double c){return Scalar::clamp(x,min,max,c);};
    return transform(function,min_count,c);
}
Tensor clamp(const Tensor& x,const Tensor& min,double max,double c,size_t min_count){
    auto function=[max,c](double x,double min){return Scalar::clamp(x,min,max,c);};
    return transform(function,min_count,x,min);
}
Tensor clamp(const Tensor& x,double min,const Tensor& max,double c,size_t min_count){
    auto function=[min,c](double x,double max){return Scalar::clamp(x,min,max,c);};
    return transform(function,min_count,x,max);
}
Tensor clamp(const Tensor& x,double min,double max,const Tensor& c,size_t min_count){
    auto function=[min,max](double x,double c){return Scalar::clamp(x,min,max,c);};
    return transform(function,min_count,x,c);
}
Tensor clamp(double x,const Tensor& min,const Tensor& max,double c,size_t min_count){
    auto function=[x,c](double min,double max){return Scalar::clamp(x,min,max,c);};
    return transform(function,min_count,min,max);
}
Tensor clamp(double x,const Tensor& min,double max,const Tensor& c,size_t min_count){
    auto function=[x,max](double min,double c){return Scalar::clamp(x,min,max,c);};
    return transform(function,min_count,min,c);
}
Tensor clamp(double x,double min,const Tensor& max,const Tensor& c,size_t min_count){
    auto function=[x,min](double max,double c){return Scalar::clamp(x,min,max,c);};
    return transform(function,min_count,max,c);
}
Tensor clamp(const Tensor& x,const Tensor& min,const Tensor& max,double c,size_t min_count){
    auto function=[c](double x,double min,double max){return Scalar::clamp(x,min,max,c);};
    return transform(function,min_count,x,min,max);
}
Tensor clamp(const Tensor& x,const Tensor& min,double max,const Tensor& c,size_t min_count){
    auto function=[max](double x,double min,double c){return Scalar::clamp(x,min,max,c);};
    return transform(function,min_count,x,min,c);
}
Tensor clamp(const Tensor& x,double min,const Tensor& max,const Tensor& c,size_t min_count){
    auto function=[min](double x,double max,double c){return Scalar::clamp(x,min,max,c);};
    return transform(function,min_count,x,max,c);
}
Tensor clamp(double x,const Tensor& min,const Tensor& max,const Tensor& c,size_t min_count){
    auto function=[x](double min,double max,double c){return Scalar::clamp(x,min,max,c);};
    return transform(function,min_count,min,max,c);
}
Tensor clamp(const Tensor& x,const Tensor& min,const Tensor& max,const Tensor& c,size_t min_count){
    auto function=[](double x,double min,double max,double c){return Scalar::clamp(x,min,max,c);};//Overloaded function so we have to do this
    return transform(function,min_count,x,min,max,c);
}

Tensor sign(const Tensor& t,size_t min_count){
    return transform(Scalar::sign,min_count,t);
}
Tensor fdim(const Tensor& x,double y,size_t min_count){
    auto function=[y](double x){return Scalar::fdim(x,y);};
    return transform(function,min_count,x);
}
Tensor fdim(double x,const Tensor& y,size_t min_count){
    auto function=[x](double y){return Scalar::fdim(x,y);};
    return transform(function,min_count,y);
}
Tensor fdim(const Tensor& x,const Tensor& y,size_t min_count){
    return transform(Scalar::fdim,min_count,x,y);
}
}