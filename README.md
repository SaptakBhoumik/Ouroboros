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
## Installation Guide
Run the following commands in your terminal
```bash
git clone https://github.com/SaptakBhoumik/Ouroboros.git
cd Ouroboros
meson --buildtype=release dist
cd dist
ninja
ninja install #Use sudo if required
```

## Documentation

For comprehensive documentation of all public APIs, including detailed examples and NumPy equivalents, see [doc/DOC.MD](doc/DOC.MD).

The documentation includes:
- Complete API reference for Shape and Tensor classes
- Iterator classes (NDRange, IdxIterator, IdxIterator2)
- All operators (bitwise, arithmetic, comparison, logical)
- Utility functions (transform, reduce, accumulate, outer, concat, transpose, flip, broadcast)
- NumPy-style examples showing equivalent operations for users familiar with NumPy

```cpp
// Quick comparison with NumPy:
// NumPy: arr = np.ones((2, 3))
Ouroboros::Tensor<double> t({2, 3}, 1.0);

// NumPy: arr2 = arr + 5
auto t2 = t + 5.0;

// NumPy: result = np.matmul(a, b)
auto result = Ouroboros::matmul(t1, t2);

// NumPy: transformed = np.sin(arr)
auto sin_func = [](double x) { return std::sin(x); };
auto transformed = Ouroboros::transform<sin_func>(t)
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


