#include "op.hpp"
#include <omp.h>
namespace Ouroboros{
namespace CPU{
double* neg_ptr(const double* a,size_t size){
    double* result=new double[size];
    for(size_t i=0;i<size;++i){
        result[i]=-a[i];
    }
    return result;
}



double* add_ptr(const double* a,const double* b,size_t size){
    double* result=new double[size];
    for(size_t i=0;i<size;++i){
        result[i]=a[i]+b[i];
    }
    return result;
}
double* sub_ptr(const double* a,const double* b,size_t size){
    double* result=new double[size];
    for(size_t i=0;i<size;++i){
        result[i]=a[i]-b[i];
    }
    return result;
}
double* mul_ptr(const double* a,const double* b,size_t size){
    double* result=new double[size];
    for(size_t i=0;i<size;++i){
        result[i]=a[i]*b[i];
    }
    return result;
}
double* div_ptr(const double* a,const double* b,size_t size){
    double* result=new double[size];
    for(size_t i=0;i<size;++i){
        result[i]=a[i]/b[i];
    }
    return result;
}


double* add_ptr(const double* a,double b,size_t size){
    double* result=new double[size];
    for(size_t i=0;i<size;++i){
        result[i]=a[i]+b;
    }
    return result;
}
double* sub_ptr(const double* a,double b,size_t size){
    double* result=new double[size];
    for(size_t i=0;i<size;++i){
        result[i]=a[i]-b;
    }
    return result;
}
double* mul_ptr(const double* a,double b,size_t size){
    double* result=new double[size];
    for(size_t i=0;i<size;++i){
        result[i]=a[i]*b;
    }
    return result;
}
double* div_ptr(const double* a,double b,size_t size){
    double* result=new double[size];
    for(size_t i=0;i<size;++i){
        result[i]=a[i]/b;
    }
    return result;
}



double* sub_ptr(double a,const double* b,size_t size){
    double* result=new double[size];
    for(size_t i=0;i<size;++i){
        result[i]=a-b[i];
    }
    return result;
}
double* div_ptr(double a,const double* b,size_t size){
    double* result=new double[size];
    for(size_t i=0;i<size;++i){
        result[i]=a/b[i];
    }
    return result;
}



void add_ptr_self(double* a,const double* b,size_t size){
    for(size_t i=0;i<size;++i){
        a[i]+=b[i];
    }
}
void sub_ptr_self(double* a,const double* b,size_t size){
    for(size_t i=0;i<size;++i){
        a[i]-=b[i];
    }
}
void mul_ptr_self(double* a,const double* b,size_t size){
    for(size_t i=0;i<size;++i){
        a[i]*=b[i];
    }
}
void div_ptr_self(double* a,const double* b,size_t size){
    for(size_t i=0;i<size;++i){
        a[i]/=b[i];
    }
}



void add_ptr_self(double* a,double b,size_t size){
    for(size_t i=0;i<size;++i){
        a[i]+=b;
    }
}
void sub_ptr_self(double* a,double b,size_t size){
    for(size_t i=0;i<size;++i){
        a[i]-=b;
    }
}
void mul_ptr_self(double* a,double b,size_t size){
    for(size_t i=0;i<size;++i){
        a[i]*=b;
    }
}
void div_ptr_self(double* a,double b,size_t size){
    for(size_t i=0;i<size;++i){
        a[i]/=b;
    }
}
}
}