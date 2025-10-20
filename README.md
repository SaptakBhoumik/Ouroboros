# Ouroboros

Ouroboros is a C++ library to work with N dimentional tensors

## Why?

Well the main reason that this library exists is for personal use in my other projects. The only reason that it is public is because some people may find it useful

## Goals

- Make it easy to use
- Give the bare minimum features so that it can be easily extended(with minimal code) to meet that users need

## Example usage

```cpp
#include <ouroboros/tensor.hpp>
#include <cmath>
int main(){
    Ouroboros::Tensor<double> t1({2,3},1.0);
    std::cout<<"t1:\n"<<t1<<"\n";
    Ouroboros::Tensor<double> t2({3,2},2.0);
    std::cout<<"t2:\n"<<t2<<"\n";

    auto func=[](double x,double y)->double{
        return x*y;
    };
    Ouroboros::Tensor t3=Ouroboros::outer<func>(t1,t2);//Outer product of 2 tensors
    std::cout<<"t3:\n"<<t3<<"\n";

    auto sin=[](double x)->double{
        return std::sin(x);
    };
    t3=Ouroboros::transform<sin>(t3);//Apply sin to all elements of t3
    std::cout<<"sin(t3):\n"<<t3<<"\n";

    std::cout<<"t1+t2:\n"<<t1+t2<<"\n";//Element wise addition
    std::cout<<"matmul(t1,t2):\n"<<Ouroboros::matmul(t1,t2)<<"\n";//Matrix multiplication


    //There's a lot more you can do with tensors, check the documentation for more information
}
```

More examples can be found [HERE](https://github.com/SaptakBhoumik/Ouroboros/tree/master/example) 


## Future TODOs

- Support block wise opperations like conv,block reduction etc
- Better support for tensor slices and easier function to create function
- Support GPU
- Create a logo

## Have questions?

Cool, you can contact me via mail.
Email: saptakbhoumik.acad@gmail.com

## License
The Ouroboros library is licensed under the [Mozilla Public License](https://github.com/SaptakBhoumik/Ouroboros/blob/master/LICENSE), which is attached in this repository


