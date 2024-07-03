#ifndef __Ouroboros__
#error "This file should not be included directly"
#endif
#define PERMUTE_OFFSET(A,B,C,D) \
{\
    size_t j=0;\
    while (true) {\
        size_t offset=0;\
        _Pragma("omp simd reduction(+:offset)")\
        for (size_t i = 0; i < B.size(); ++i) {\
            offset+=B[i]*C[i];\
        }\
        D[j++]=offset;\
        /*Find the rightmost index that can be incremented*/\
        int64_t k = (int64_t)A.size() - 1;\
        while (k >= 0 && B[k] == A[k] - 1){\
            k--;\
        }\
        /*If no such index exists, we are done*/\
        if (k < 0) {\
            break;\
        }\
        /*Increment the current index and reset all subsequent indices*/\
        B[k]++;\
        for (size_t i = k + 1; i < A.size(); ++i) {\
            B[i] = 0;\
        }\
    }\
}

#define PERMUTE_IDX(A,B,D,count) \
{\
    /*D is a matrix of dim(count,B.size())*/\
    size_t j=0;\
    while (true) {\
        for (size_t i = 0; i < B.size(); ++i) {\
            D[j*count+i]=B[i];\
        }\
        /*Find the rightmost index that can be incremented*/\
        int64_t k = (int64_t)A.size() - 1;\
        while (k >= 0 && B[k] == A[k] - 1){\
            k--;\
        }\
        /*If no such index exists, we are done*/\
        if (k < 0) {\
            break;\
        }\
        /*Increment the current index and reset all subsequent indices*/\
        B[k]++;\
        for (size_t i = k + 1; i < A.size(); ++i) {\
            B[i] = 0;\
        }\
        j++;\
    }\
}
#define PERMUTE_2OFFSET(A,B,C1,C2,D1,D2) \
{\
    size_t j=0;\
    while (true) {\
        size_t offset1=0;\
        size_t offset2=0;\
        _Pragma("omp simd reduction(+:offset1,offset2)")\
        for (size_t i = 0; i < B.size(); ++i) {\
            offset1+=B[i]*C1[i];\
            offset2+=B[i]*C2[i];\
        }\
        D1[j]=offset1;\
        D2[j]=offset2;\
        j++;\
        /*Find the rightmost index that can be incremented*/\
        int64_t k = (int64_t)A.size() - 1;\
        while (k >= 0 && B[k] == A[k] - 1){\
            k--;\
        }\
        /*If no such index exists, we are done*/\
        if (k < 0) {\
            break;\
        }\
        /*Increment the current index and reset all subsequent indices*/\
        B[k]++;\
        for (size_t i = k + 1; i < A.size(); ++i) {\
            B[i] = 0;\
        }\
    }\
}
namespace Ouroboros{
namespace __Private__Impl__{
//Do not use this functions from here in your code
template<typename tuple,typename T,typename func_type, std::size_t ... Is>
__always_inline void __apply_impl(func_type func,const tuple& t,T* res,size_t idx, std::index_sequence<Is...>){
    res[idx]=func(std::get<Is>(t)[idx]...);
}
template<std::size_t N, typename tuple,typename T,typename func_type,typename Indices = std::make_index_sequence<N>>
__always_inline void __apply(func_type func,const tuple& t,T* res,size_t idx){
    __apply_impl(func,t,res,idx,Indices{});
}
template<typename tuple,typename T,typename func_type, std::size_t ... Is>
__always_inline void __apply_impl_self(func_type func,const tuple& t,T* self,size_t idx1,size_t idx2, std::index_sequence<Is...>){
    auto temp=self[idx2];
    self[idx2]=func(temp,std::get<Is>(t)[idx1]...);
}
template<std::size_t N, typename tuple,typename T,typename func_type,typename Indices = std::make_index_sequence<N>>
__always_inline void __apply_self(func_type func,const tuple& t,T* self,size_t idx1,size_t idx2){
    __apply_impl_self(func,t,self,idx1,idx2,Indices{});
}
//Credit:-https://stackoverflow.com/questions/27822277/finding-out-the-return-type-of-a-function-lambda-or-function

template <typename F>
struct return_type_impl;

template <typename R, typename... Args>
struct return_type_impl<R(Args...)> { using type = R; };

template <typename R, typename... Args>
struct return_type_impl<R(Args..., ...)> { using type = R; };

template <typename R, typename... Args>
struct return_type_impl<R(*)(Args...)> { using type = R; };

template <typename R, typename... Args>
struct return_type_impl<R(*)(Args..., ...)> { using type = R; };

template <typename R, typename... Args>
struct return_type_impl<R(&)(Args...)> { using type = R; };

template <typename R, typename... Args>
struct return_type_impl<R(&)(Args..., ...)> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args...)> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args..., ...)> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args...) &> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args..., ...) &> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args...) &&> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args..., ...) &&> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args...) const> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args..., ...) const> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args...) const&> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args..., ...) const&> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args...) const&&> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args..., ...) const&&> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args...) volatile> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args..., ...) volatile> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args...) volatile&> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args..., ...) volatile&> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args...) volatile&&> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args..., ...) volatile&&> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args...) const volatile> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args..., ...) const volatile> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args...) const volatile&> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args..., ...) const volatile&> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args...) const volatile&&> { using type = R; };

template <typename R, typename C, typename... Args>
struct return_type_impl<R(C::*)(Args..., ...) const volatile&&> { using type = R; };

template <typename T, typename = void>
struct return_type
    : return_type_impl<T> {};

template <typename T>
struct return_type<T, decltype(void(&T::operator()))>
    : return_type_impl<decltype(&T::operator())> {};

template <typename T>
using return_type_t = typename return_type<T>::type;


template<bool n>
struct Typer{};

template<>
struct Typer<0>{
    typedef Tensor Type;
};

template<>
struct Typer<1>{
    typedef BoolTensor Type;
};
template<size_t thread_c=8,size_t min_count=__MIN__COUNT__FOR__THREAD__>
__always_inline Tensor ___broadcast(const std::function<double(double,double)>& func,const Tensor& t1,const Tensor& t2){
    const Shape t1_shape=t1.shape();
    const Shape t2_shape=t2.shape();
    const Shape t1_strides=t1.strides();
    const Shape t2_strides=t2.strides();

    const size_t dim=t1_shape.dim();

    const double* t1_data=t1.data();
    const double* t2_data=t2.data();

    std::vector<size_t> axis;//Axis where t2_shape[i]==1 i.e where t2 can be broadcasted
    std::vector<size_t> non_axis;//Rest of the axis
    std::vector<size_t> X;//Contains t1_shape[i] for every i where t1_shape[i]==t2_shape[i]
    std::vector<size_t> X_stride_t1;//Contains t1_strides[i] for every i where t1_shape[i]==t2_shape[i]
    std::vector<size_t> X_stride_t2;//Contains t2_strides[i] for every i where t1_shape[i]==t2_shape[i]
    std::vector<size_t> Y;//Contains t1_shape[i] for every i where t2_shape[i]==1
    std::vector<size_t> Y_stride_t1;//Contains t1_strides[i] for every i where t2_shape[i]==1
    size_t x_count=1;
    size_t y_count=1;
    for(size_t i=0;i<dim;i++){
        if(t1_shape[i]==t2_shape[i]){
            non_axis.push_back(i);
            X.push_back(t1_shape[i]);
            X_stride_t1.push_back(t1_strides[i]);
            X_stride_t2.push_back(t2_strides[i]);
            x_count*=t1_shape[i];
        }
        else if(t2_shape[i]==1){
            axis.push_back(i);
            Y.push_back(t1_shape[i]);
            Y_stride_t1.push_back(t1_strides[i]);
            y_count*=t1_shape[i];
        }
        else{
            throw std::invalid_argument("Invalid shape for broadcast");
        }
    }
    size_t* x_offsets_t1=new size_t[x_count];
    size_t* x_offsets_t2=new size_t[x_count];
    {
        std::vector<size_t> B(X.size(), 0);
        PERMUTE_2OFFSET(X, B, X_stride_t1, X_stride_t2, x_offsets_t1, x_offsets_t2);
    }
    size_t* y_offsets_t1=new size_t[y_count];
    {
        std::vector<size_t> B(Y.size(), 0);
        PERMUTE_OFFSET(Y, B, Y_stride_t1,y_offsets_t1);
    }
    Tensor res(t1_shape);
    double* res_data=res.data();
    if(x_count<=min_count&&y_count<=min_count){
        for(size_t i=0;i<x_count;i++){
            for(size_t j=0;j<y_count;j++){
                size_t idx1=x_offsets_t1[i]+y_offsets_t1[j];
                size_t idx2=x_offsets_t2[i];
                res_data[idx1]=func(t1_data[idx1],t2_data[idx2]);
            }
        }
    }
    else{
        if(x_count>=y_count){
            #pragma omp parallel for num_threads(thread_c)
            for(size_t i=0;i<x_count;i++){
                for(size_t j=0;j<y_count;j++){
                    size_t idx1=x_offsets_t1[i]+y_offsets_t1[j];
                    size_t idx2=x_offsets_t2[i];
                    res_data[idx1]=func(t1_data[idx1],t2_data[idx2]);
                }
            }
        }
        else{
            #pragma omp parallel for num_threads(thread_c)
            for(size_t j=0;j<y_count;j++){
                for(size_t i=0;i<x_count;i++){
                    size_t idx1=x_offsets_t1[i]+y_offsets_t1[j];
                    size_t idx2=x_offsets_t2[i];
                    res_data[idx1]=func(t1_data[idx1],t2_data[idx2]);
                }
            }
        }
    }
    delete[] x_offsets_t1;
    delete[] x_offsets_t2;
    delete[] y_offsets_t1;
    return res;
}

template<typename T,
        size_t thread_c=8,
        size_t min_count=__MIN__COUNT__FOR__THREAD__>
__always_inline T concat(size_t axis,const std::vector<T>& tensors,const Shape& res_shape){
    const size_t perm_count=res_shape.count()/res_shape[axis];
    const size_t dim=res_shape.dim();
    size_t* perm_idxs=new size_t[perm_count*dim];
    {
        std::vector<size_t> A(dim);
        std::vector<size_t> B(dim, 0);
        for(size_t i=0;i<dim;i++){
            A[i]=res_shape[i];
        }
        A[axis]=1;//So after permutation it will be 0 at axis
        PERMUTE_IDX(A, B,perm_idxs,perm_count);
    }
    
    T res(res_shape);
    const Shape res_strides=res.strides();
    const size_t res_stride_at_axis=res_strides[axis];
    auto res_data=res.data();
    std::vector<size_t> cumm_shape_at_axis={0};
    for(size_t i=0;i<tensors.size()-1;i++){
        size_t temp=cumm_shape_at_axis.back()+tensors[i].shape()[axis];
        cumm_shape_at_axis.push_back(temp);
    }
    if(perm_count<=min_count){
        for(size_t i=0;i<tensors.size();i++){
            const auto data=tensors[i].data();
            const Shape strides=tensors[i].strides();
            const size_t shape_at_axis=tensors[i].shape()[axis];
            const size_t stride_at_axis=strides[axis];
            const size_t idx=cumm_shape_at_axis[i];
            for(size_t j=0;j<perm_count;j++){
                size_t idx1=0;//for data
                size_t idx2=0;//for res
                for(size_t k=0;k<dim;k++){
                    idx1+=perm_idxs[j*dim+k]*strides[k];
                    idx2+=perm_idxs[j*dim+k]*res_strides[k];
                }
                for(size_t k=0;k<shape_at_axis;k++){
                    res_data[idx2+res_stride_at_axis*(idx+k)]=data[idx1+stride_at_axis*k];
                }
            }
        }
    }
    else{
        #pragma omp parallel for num_threads(thread_c)//Copying the tensor and is not time consuming so we thread this part
        for(size_t i=0;i<tensors.size();i++){
            const auto data=tensors[i].data();
            const Shape strides=tensors[i].strides();
            const size_t shape_at_axis=tensors[i].shape()[axis];
            const size_t stride_at_axis=strides[axis];
            const size_t idx=cumm_shape_at_axis[i];
            for(size_t j=0;j<perm_count;j++){
                size_t idx1=0;//for data
                size_t idx2=0;//for res
                for(size_t k=0;k<dim;k++){
                    idx1+=perm_idxs[j*dim+k]*strides[k];
                    idx2+=perm_idxs[j*dim+k]*res_strides[k];
                }
                for(size_t k=0;k<shape_at_axis;k++){
                    res_data[idx2+res_stride_at_axis*(idx+k)]=data[idx1+stride_at_axis*k];
                }
            }
        }
    }
    delete []perm_idxs;
    return res;
}
}
}
