#include <omp.h>
#include <limits>
#include <cmath>
#include "op.hpp"
//NOTE:-We use abs for checking equality of floating point numbers because of floating point errors
//Since it is a slow operation so we use threads only for equality operations to speed up the process
namespace Ouroboros{
namespace CPU{
double* neg_ptr(const double* a,std::size_t size){
    double* result=new double[size];
    for(std::size_t i=0;i<size;i++){
        result[i]=-a[i];
    }
    return result;
}



double* add_ptr(const double* a,const double* b,std::size_t size){
    double* result=new double[size];
    for(std::size_t i=0;i<size;i++){
        result[i]=a[i]+b[i];
    }
    return result;
}
double* sub_ptr(const double* a,const double* b,std::size_t size){
    double* result=new double[size];
    for(std::size_t i=0;i<size;i++){
        result[i]=a[i]-b[i];
    }
    return result;
}
double* mul_ptr(const double* a,const double* b,std::size_t size){
    double* result=new double[size];
    for(std::size_t i=0;i<size;i++){
        result[i]=a[i]*b[i];
    }
    return result;
}
double* div_ptr(const double* a,const double* b,std::size_t size){
    double* result=new double[size];
    for(std::size_t i=0;i<size;i++){
        result[i]=a[i]/b[i];
    }
    return result;
}


double* add_ptr(const double* a,double b,std::size_t size){
    double* result=new double[size];
    for(std::size_t i=0;i<size;i++){
        result[i]=a[i]+b;
    }
    return result;
}
double* sub_ptr(const double* a,double b,std::size_t size){
    double* result=new double[size];
    for(std::size_t i=0;i<size;i++){
        result[i]=a[i]-b;
    }
    return result;
}
double* mul_ptr(const double* a,double b,std::size_t size){
    double* result=new double[size];
    for(std::size_t i=0;i<size;i++){
        result[i]=a[i]*b;
    }
    return result;
}
double* div_ptr(const double* a,double b,std::size_t size){
    double* result=new double[size];
    for(std::size_t i=0;i<size;i++){
        result[i]=a[i]/b;
    }
    return result;
}



double* sub_ptr(double a,const double* b,std::size_t size){
    double* result=new double[size];
    for(std::size_t i=0;i<size;i++){
        result[i]=a-b[i];
    }
    return result;
}
double* div_ptr(double a,const double* b,std::size_t size){
    double* result=new double[size];
    for(std::size_t i=0;i<size;i++){
        result[i]=a/b[i];
    }
    return result;
}



void add_ptr_self(double* a,const double* b,std::size_t size){
    for(std::size_t i=0;i<size;i++){
        a[i]+=b[i];
    }
}
void sub_ptr_self(double* a,const double* b,std::size_t size){
    for(std::size_t i=0;i<size;i++){
        a[i]-=b[i];
    }
}
void mul_ptr_self(double* a,const double* b,std::size_t size){
    for(std::size_t i=0;i<size;i++){
        a[i]*=b[i];
    }
}
void div_ptr_self(double* a,const double* b,std::size_t size){
    for(std::size_t i=0;i<size;i++){
        a[i]/=b[i];
    }
}



void add_ptr_self(double* a,double b,std::size_t size){
    for(std::size_t i=0;i<size;i++){
        a[i]+=b;
    }
}
void sub_ptr_self(double* a,double b,std::size_t size){
    for(std::size_t i=0;i<size;i++){
        a[i]-=b;
    }
}
void mul_ptr_self(double* a,double b,std::size_t size){
    for(std::size_t i=0;i<size;i++){
        a[i]*=b;
    }
}
void div_ptr_self(double* a,double b,std::size_t size){
    for(std::size_t i=0;i<size;i++){
        a[i]/=b;
    }
}

bool* eq_ptr(const double* a,const double* b,std::size_t size){
    double eps=std::numeric_limits<double>::epsilon();
    bool* result=new bool[size];
    if(size<=__MIN__COUNT__FOR__THREAD__){
        for(std::size_t i=0;i<size;i++){
            result[i]=std::abs(a[i]-b[i])<eps;
        }
    }
    else{
        #pragma omp parallel for
        for(std::size_t i=0;i<size;i++){
            result[i]=std::abs(a[i]-b[i])<eps;
        }
    }
    return result;
}
bool* neq_ptr(const double* a,const double* b,std::size_t size){
    double eps=std::numeric_limits<double>::epsilon();
    bool* result=new bool[size];
    if(size<=__MIN__COUNT__FOR__THREAD__){
        for(std::size_t i=0;i<size;i++){
            result[i]=std::abs(a[i]-b[i])>=eps;
        }
    }
    else{
        #pragma omp parallel for
        for(std::size_t i=0;i<size;i++){
            result[i]=std::abs(a[i]-b[i])>=eps;
        }
    }
    return result;
}
bool* lt_ptr(const double* a,const double* b,std::size_t size){
    bool* result=new bool[size];
    for(std::size_t i=0;i<size;i++){
        result[i]=a[i]<b[i];
    }
    return result;
}
bool* gt_ptr(const double* a,const double* b,std::size_t size){
    bool* result=new bool[size];
    for(std::size_t i=0;i<size;i++){
        result[i]=a[i]>b[i];
    }
    return result;

}
bool* lteq_ptr(const double* a,const double* b,std::size_t size){
    double eps=std::numeric_limits<double>::epsilon();
    bool* result=new bool[size];
    if(size<=__MIN__COUNT__FOR__THREAD__){
        for(std::size_t i=0;i<size;i++){
            result[i]=(a[i]<b[i])||(std::abs(a[i]-b[i])<eps);
        }
    }
    else{
        #pragma omp parallel for
        for(std::size_t i=0;i<size;i++){
            result[i]=(a[i]<b[i])||(std::abs(a[i]-b[i])<eps);
        }
    }
    return result;
}
bool* gteq_ptr(const double* a,const double* b,std::size_t size){
    double eps=std::numeric_limits<double>::epsilon();
    bool* result=new bool[size];
    if(size<=__MIN__COUNT__FOR__THREAD__){
        for(std::size_t i=0;i<size;i++){
            result[i]=(a[i]>b[i])||(std::abs(a[i]-b[i])<eps);
        }
    }
    else{
        #pragma omp parallel for
        for(std::size_t i=0;i<size;i++){
            result[i]=(a[i]>b[i])||(std::abs(a[i]-b[i])<eps);
        }
    }
    return result;
}

bool* eq_ptr(const double* a,double b,std::size_t size){
    double eps=std::numeric_limits<double>::epsilon();
    bool* result=new bool[size];
    if(size<=__MIN__COUNT__FOR__THREAD__){
        for(std::size_t i=0;i<size;i++){
            result[i]=std::abs(a[i]-b)<eps;
        }
    }
    else{
        #pragma omp parallel for
        for(std::size_t i=0;i<size;i++){
            result[i]=std::abs(a[i]-b)<eps;
        }
    }
    return result;
}
bool* neq_ptr(const double* a,double b,std::size_t size){
    double eps=std::numeric_limits<double>::epsilon();
    bool* result=new bool[size];
    if(size<=__MIN__COUNT__FOR__THREAD__){
        for(std::size_t i=0;i<size;i++){
            result[i]=std::abs(a[i]-b)>=eps;
        }
    }
    else{
        #pragma omp parallel for
        for(std::size_t i=0;i<size;i++){
            result[i]=std::abs(a[i]-b)>=eps;
        }
    }
    return result;
}
bool* lt_ptr(const double* a,double b,std::size_t size){
    bool* result=new bool[size];
    for(std::size_t i=0;i<size;i++){
        result[i]=a[i]<b;
    }
    return result;
}
bool* gt_ptr(const double* a,double b,std::size_t size){
    bool* result=new bool[size];
    for(std::size_t i=0;i<size;i++){
        result[i]=a[i]>b;
    }
    return result;
}
bool* lteq_ptr(const double* a,double b,std::size_t size){
    double eps=std::numeric_limits<double>::epsilon();
    bool* result=new bool[size];
    if(size<=__MIN__COUNT__FOR__THREAD__){
        for(std::size_t i=0;i<size;i++){
            result[i]=(a[i]<b)||(std::abs(a[i]-b)<eps);
        }
    }
    else{
        #pragma omp parallel for
        for(std::size_t i=0;i<size;i++){
            result[i]=(a[i]<b)||(std::abs(a[i]-b)<eps);
        }
    }
    return result;
}
bool* gteq_ptr(const double* a,double b,std::size_t size){
    double eps=std::numeric_limits<double>::epsilon();
    bool* result=new bool[size];
    if(size<=__MIN__COUNT__FOR__THREAD__){
        for(std::size_t i=0;i<size;i++){
            result[i]=(a[i]>b)||(std::abs(a[i]-b)<eps);
        }
    }
    else{
        #pragma omp parallel for
        for(std::size_t i=0;i<size;i++){
            result[i]=(a[i]>b)||(std::abs(a[i]-b)<eps);
        }
    }
    return result;
}
}
}