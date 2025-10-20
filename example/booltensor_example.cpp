#include <cstdint>
#include <ouroboros/tensor.hpp>
#include <iostream>
int main(){
    Ouroboros::Shape shape={2,3,4};
    //Initialize a Tensor<bool> of shape 2x3x4 with all values as true
    Ouroboros::Tensor<bool> t1(shape);//We dont specify the value so only the data is allocated but not initialized with any value
    Ouroboros::Tensor<bool> t2(shape,true);
    std::cout<<"t1="<<t1<<std::endl;
    std::cout<<"t2="<<t2<<std::endl;
    bool* data=new bool[shape.count()];//It has to be heap allocated
    for(std::size_t i=0;i<shape.count();i++){
        data[i]=i%2==0;
    }
    //This method is useful when you have a preallocated array and you want to use it as the data for the tensor
    //But in general u should avoid it
    Ouroboros::Tensor<bool> t3(shape,data);//Note that the data is shared and not copied so the user should not use the data afterwards cuz we take ownership of the data
    std::cout<<"t3="<<t3<<std::endl;
    //Copy constructor
    Ouroboros::Tensor<bool> t4(t3);
    std::cout<<"t4="<<t4<<std::endl;
    //Move constructor
    Ouroboros::Tensor<bool> t5(std::move(t4));
    std::cout<<"t5="<<t5<<std::endl;
    //Assignment operator
    t1=t2;
    std::cout<<"t1="<<t1<<std::endl;
    //Reshape
    t1.reshape({4,3,2});//The count should be the same for both the shapes
    std::cout<<"t1="<<t1<<std::endl;
    //Indexing
    //When we use std::size_t as index then we get tensor.data[index] as the return value    
    //When we use the [] operator we get a reference to the value so we can modify it
    t1[(size_t)0]=true;
    std::cout<<"t2[0]="<<t2[(size_t)0]<<std::endl;
    t1[1]=false;
    std::cout<<"t2[1]="<<t2[1]<<std::endl;
    //When we use Shape as index then we get tensor.data[offset] as the return value 
    //Where offset is calculated using the strides and index    
    std::vector<size_t> index={0,1,2};
    t1[index]=true;//This is equivalent to t1[0,1,2]=true
    std::cout<<"t2[0,1,2]="<<t2[index]<<std::endl;
    //Offset
    std::cout<<"Offset of 0,1,2 is "<<t1.shape().offset(index)<<std::endl;
    //Slicing
    std::vector<size_t> start={0,0,0};
    std::vector<size_t> step={1,1,1};
    std::vector<size_t> end={1,2,3};
    //Getting raw data
    bool* raw_data=t1.data();
    //Getting the shape
    Ouroboros::Shape s=t1.shape();
    std::cout<<"Shape of t1="<<s<<std::endl;
    //Getting the strides
    //Getting the no of dimensions
    std::size_t dim=t1.dim();
    std::cout<<"Dim of t1="<<dim<<std::endl;
    //Operators 
    Ouroboros::Tensor<bool> t7({2,3,4},true);
    Ouroboros::Tensor<bool> t8({2,3,4},false);
    Ouroboros::Tensor<bool> t9=t7==t8;//Does element wise comparison
    std::cout<<"t9="<<t9<<std::endl;
    t9=t7!=t8;//Does element wise comparison
    std::cout<<"t9="<<t9<<std::endl;
    t9=t7&&t8;//Does element wise and
    std::cout<<"t9="<<t9<<std::endl;
    t9=t7||t8;//Does element wise or
    std::cout<<"t9="<<t9<<std::endl;
    t9=!t7;//Does element wise not
    std::cout<<"t9="<<t9<<std::endl;
    t9=t7==true;//Does element wise comparison with a scalar
    std::cout<<"t9="<<t9<<std::endl;
    t9=t7!=true;//Does element wise comparison with a scalar
    std::cout<<"t9="<<t9<<std::endl;
    t9=t7&&true;//Does element wise and with a scalar
    std::cout<<"t9="<<t9<<std::endl;
    t9=t7||true;//Does element wise or with a scalar
    std::cout<<"t9="<<t9<<std::endl;
    /*
    You can also do both boolean && tensor and tensor && boolean
    Same goes for || and == and !=
    */
    return 0;
    
}