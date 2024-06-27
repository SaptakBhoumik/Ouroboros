#include "op.hpp"
#include <omp.h>
#include <limits>
namespace Ouroboros{
namespace Scalar{
double abs(double a);//Defined in src/func/scalar/basic.cpp
}
namespace CPU{
double* neg_ptr(const double* a,size_t size){
    double* result=new double[size];
    for(size_t i=0;i<size;i++){
        result[i]=-a[i];
    }
    return result;
}



double* add_ptr(const double* a,const double* b,size_t size){
    double* result=new double[size];
    for(size_t i=0;i<size;i++){
        result[i]=a[i]+b[i];
    }
    return result;
}
double* sub_ptr(const double* a,const double* b,size_t size){
    double* result=new double[size];
    for(size_t i=0;i<size;i++){
        result[i]=a[i]-b[i];
    }
    return result;
}
double* mul_ptr(const double* a,const double* b,size_t size){
    double* result=new double[size];
    for(size_t i=0;i<size;i++){
        result[i]=a[i]*b[i];
    }
    return result;
}
double* div_ptr(const double* a,const double* b,size_t size){
    double* result=new double[size];
    for(size_t i=0;i<size;i++){
        result[i]=a[i]/b[i];
    }
    return result;
}


double* add_ptr(const double* a,double b,size_t size){
    double* result=new double[size];
    for(size_t i=0;i<size;i++){
        result[i]=a[i]+b;
    }
    return result;
}
double* sub_ptr(const double* a,double b,size_t size){
    double* result=new double[size];
    for(size_t i=0;i<size;i++){
        result[i]=a[i]-b;
    }
    return result;
}
double* mul_ptr(const double* a,double b,size_t size){
    double* result=new double[size];
    for(size_t i=0;i<size;i++){
        result[i]=a[i]*b;
    }
    return result;
}
double* div_ptr(const double* a,double b,size_t size){
    double* result=new double[size];
    for(size_t i=0;i<size;i++){
        result[i]=a[i]/b;
    }
    return result;
}



double* sub_ptr(double a,const double* b,size_t size){
    double* result=new double[size];
    for(size_t i=0;i<size;i++){
        result[i]=a-b[i];
    }
    return result;
}
double* div_ptr(double a,const double* b,size_t size){
    double* result=new double[size];
    for(size_t i=0;i<size;i++){
        result[i]=a/b[i];
    }
    return result;
}



void add_ptr_self(double* a,const double* b,size_t size){
    for(size_t i=0;i<size;i++){
        a[i]+=b[i];
    }
}
void sub_ptr_self(double* a,const double* b,size_t size){
    for(size_t i=0;i<size;i++){
        a[i]-=b[i];
    }
}
void mul_ptr_self(double* a,const double* b,size_t size){
    for(size_t i=0;i<size;i++){
        a[i]*=b[i];
    }
}
void div_ptr_self(double* a,const double* b,size_t size){
    for(size_t i=0;i<size;i++){
        a[i]/=b[i];
    }
}



void add_ptr_self(double* a,double b,size_t size){
    for(size_t i=0;i<size;i++){
        a[i]+=b;
    }
}
void sub_ptr_self(double* a,double b,size_t size){
    for(size_t i=0;i<size;i++){
        a[i]-=b;
    }
}
void mul_ptr_self(double* a,double b,size_t size){
    for(size_t i=0;i<size;i++){
        a[i]*=b;
    }
}
void div_ptr_self(double* a,double b,size_t size){
    for(size_t i=0;i<size;i++){
        a[i]/=b;
    }
}

bool* eq_ptr(const double* a,const double* b,size_t size){
    //Slow but accurate(considering floating point precision)
    double eps=std::numeric_limits<double>::epsilon();
    bool* result=new bool[size];
    for(size_t i=0;i<size;i++){
        result[i]=Scalar::abs(a[i]-b[i])<eps;
    }
    return result;
}
bool* neq_ptr(const double* a,const double* b,size_t size){
    //Slow but accurate(considering floating point precision)
    double eps=std::numeric_limits<double>::epsilon();
    bool* result=new bool[size];
    for(size_t i=0;i<size;i++){
        result[i]=Scalar::abs(a[i]-b[i])>=eps;
    }
    return result;
}
bool* lt_ptr(const double* a,const double* b,size_t size){
    bool* result=new bool[size];
    for(size_t i=0;i<size;i++){
        result[i]=a[i]<b[i];
    }
    return result;
}
bool* gt_ptr(const double* a,const double* b,size_t size){
    bool* result=new bool[size];
    for(size_t i=0;i<size;i++){
        result[i]=a[i]>b[i];
    }
    return result;

}
bool* lteq_ptr(const double* a,const double* b,size_t size){
    bool* result=new bool[size];
    for(size_t i=0;i<size;i++){
        result[i]=a[i]<=b[i];
    }
    return result;
}
bool* gteq_ptr(const double* a,const double* b,size_t size){
    bool* result=new bool[size];
    for(size_t i=0;i<size;i++){
        result[i]=a[i]>=b[i];
    }
    return result;
}

bool* eq_ptr(const double* a,double b,size_t size){
    //Slow but accurate(considering floating point precision)
    double eps=std::numeric_limits<double>::epsilon();
    bool* result=new bool[size];
    for(size_t i=0;i<size;i++){
        result[i]=Scalar::abs(a[i]-b)<eps;
    }
    return result;
}
bool* neq_ptr(const double* a,double b,size_t size){
    //Slow but accurate(considering floating point precision)
    double eps=std::numeric_limits<double>::epsilon();
    bool* result=new bool[size];
    for(size_t i=0;i<size;i++){
        result[i]=Scalar::abs(a[i]-b)>=eps;
    }
    return result;
}
bool* lt_ptr(const double* a,double b,size_t size){
    bool* result=new bool[size];
    for(size_t i=0;i<size;i++){
        result[i]=a[i]<b;
    }
    return result;
}
bool* gt_ptr(const double* a,double b,size_t size){
    bool* result=new bool[size];
    for(size_t i=0;i<size;i++){
        result[i]=a[i]>b;
    }
    return result;
}
bool* lteq_ptr(const double* a,double b,size_t size){
    bool* result=new bool[size];
    for(size_t i=0;i<size;i++){
        result[i]=a[i]<=b;
    }
    return result;
}
bool* gteq_ptr(const double* a,double b,size_t size){
    bool* result=new bool[size];
    for(size_t i=0;i<size;i++){
        result[i]=a[i]>=b;
    }
    return result;
}
}
}