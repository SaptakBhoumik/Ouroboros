#include "tensor.hpp"
namespace Ouroboros{
BoolTensor operator!(const BoolTensor& a){
    BoolTensor res(a.shape());
    bool* res_data=res.data();
    size_t count=a.count();
    for(size_t i=0;i<count;i++){
        res_data[i]=!a[i];
    }
    return res;
}

BoolTensor operator==(const BoolTensor& a,const BoolTensor& b){
    BoolTensor res(a.shape());
    bool* res_data=res.data();
    size_t count=a.count();
    for(size_t i=0;i<count;i++){
        res_data[i]=a[i]==b[i];
    }
    return res;
}
BoolTensor operator!=(const BoolTensor& a,const BoolTensor& b){
    BoolTensor res(a.shape());
    bool* res_data=res.data();
    size_t count=a.count();
    for(size_t i=0;i<count;i++){
        res_data[i]=a[i]!=b[i];
    }
    return res;
}
BoolTensor operator&&(const BoolTensor& a,const BoolTensor& b){
    BoolTensor res(a.shape());
    bool* res_data=res.data();
    size_t count=a.count();
    for(size_t i=0;i<count;i++){
        res_data[i]=a[i]&&b[i];
    }
    return res;
}
BoolTensor operator||(const BoolTensor& a,const BoolTensor& b){
    BoolTensor res(a.shape());
    bool* res_data=res.data();
    size_t count=a.count();
    for(size_t i=0;i<count;i++){
        res_data[i]=a[i]||b[i];
    }
    return res;
}

BoolTensor operator==(const BoolTensor& a,bool b){
    BoolTensor res(a.shape());
    bool* res_data=res.data();
    size_t count=a.count();
    for(size_t i=0;i<count;i++){
        res_data[i]=a[i]==b;
    }
    return res;
}
BoolTensor operator!=(const BoolTensor& a,bool b){
    return (a==(!b));
}
BoolTensor operator&&(const BoolTensor& a,bool b){
    BoolTensor res(a.shape());
    bool* res_data=res.data();
    size_t count=a.count();
    for(size_t i=0;i<count;i++){
        res_data[i]=a[i]&&b;
    }
    return res;
}
BoolTensor operator||(const BoolTensor& a,bool b){
    BoolTensor res(a.shape());
    bool* res_data=res.data();
    size_t count=a.count();
    for(size_t i=0;i<count;i++){
        res_data[i]=a[i]||b;
    }
    return res;
}

BoolTensor operator==(bool a,const BoolTensor& b){
    return (b==a);
}
BoolTensor operator!=(bool a,const BoolTensor& b){
    return (b!=a);
}
BoolTensor operator&&(bool a,const BoolTensor& b){
    return (b&&a);
}
BoolTensor operator||(bool a,const BoolTensor& b){
    return (b||a);
}
}