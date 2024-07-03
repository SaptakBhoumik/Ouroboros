# Ouroboros

Ouroboros is a C++ library to work with N dimentional tensors

## Why?

Well the main reason that this library exists is for personal use in my other projects. The only reason that it is public is because some people may find it useful

## Goals

- Make it easy to use
- Give the bare minimum features so that it can be easily extended(with minimal code) to meet that users need

## Example usage

```cpp
#include <ouroboros/ouroboros.hpp>

int main(){
    Ouroboros::Tensor t1=Ouroboros::CreateTensor::linspace({3,3},1.0,9.0);
    std::cout<<"t2:\n"<<t1<<"\n";
    Ouroboros::Tensor t2=Ouroboros::CreateTensor::linspace({3,3},9.0,1.0);
    std::cout<<"t2:\n"<<t2<<"\n";

    auto func=[](double x,double y)->double{
        return x*y;
    };
    Ouroboros::Tensor t3=Ouroboros::outer<func>(t1,t2);//Outer product of 2 tensors
    std::cout<<"t3:\n"<<t3<<"\n";

    auto sin=[](double x)->double{
        return std::sin(x);
    };
    t3=Ouroboros::transform<sin>(t3);//Apply sin to all elements of t1
    std::cout<<"sin(t3):\n"<<t3<<"\n";
    
    std::cout<<"t1+t2:\n"<<t1+t2<<"\n";//Element wise addition

    //There's a lot more you can do with tensors, check the documentation for more information
}
```

More examples can be found [HERE](https://github.com/SaptakBhoumik/Ouroboros/tree/master/example) 

## Documentation

You can find the documentation and installation guide 
[HERE](https://github.com/SaptakBhoumik/Ouroboros/wiki)

## Future TODOs

- Support block wise reduction
- Support GPU
- Improve threading(Current implimentation is not up to the mark) 
- Improve template
- Create a logo
- Reduce the amount of heap allocation
- Improve it based on the feedback I recieve

## Have questions?

Cool, you can contact me via mail.
Email: saptakbhoumik@gmail.com

## Want to contribute?

Great, go ahead and make the changes you want, then submit a new pull request

## License
The Ouroboros library is licensed under the [Mozilla Public License](https://github.com/SaptakBhoumik/Ouroboros/blob/master/LICENSE), which is attached in this repository


