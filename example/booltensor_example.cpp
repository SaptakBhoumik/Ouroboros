#include <ouroboros/ouroboros.hpp>
#include <iostream>
int main(){
    Ouroboros::Shape shape={2,3,4};
    //Initialize a BoolTensor of shape 2x3x4 with all values as true
    Ouroboros::BoolTensor t1(shape);//We dont specify the value so only the data is allocated but not initialized with any value
    Ouroboros::BoolTensor t2(shape,true);
    std::cout<<"t1="<<t1<<std::endl;
    std::cout<<"t2="<<t2<<std::endl;
    bool* data=new bool[shape.count()];//It has to be heap allocated
    for(std::size_t i=0;i<shape.count();i++){
        data[i]=i%2==0;
    }
    //This method is useful when you have a preallocated array and you want to use it as the data for the tensor
    //But in general u should avoid it
    Ouroboros::BoolTensor t3(shape,data);//Note that the data is shared and not copied so the user should not use the data afterwards cuz we take ownership of the data
    std::cout<<"t3="<<t3<<std::endl;
    //Copy constructor
    Ouroboros::BoolTensor t4(t3);
    std::cout<<"t4="<<t4<<std::endl;
    //Move constructor
    Ouroboros::BoolTensor t5(std::move(t4));
    std::cout<<"t5="<<t5<<std::endl;
    //Assignment operator
    t1=t2;
    std::cout<<"t1="<<t1<<std::endl;
    //Reshape
    t1.reshape({4,3,2});//The count should be the same for both the shapes
    std::cout<<"t1="<<t1<<std::endl;
    t1.flatten();
    std::cout<<"t1="<<t1<<std::endl;
    //Indexing
    //When we use std::size_t as index then we get tensor.data[index] as the return value    
    //When we use the [] operator we get a reference to the value so we can modify it
    t1[0]=true;
    std::cout<<"t2[0]="<<t2[0]<<std::endl;
    t1[1]=false;
    std::cout<<"t2[1]="<<t2[1]<<std::endl;
    //When we use Shape as index then we get tensor.data[offset] as the return value 
    //Where offset is calculated using the strides and index    
    Ouroboros::Shape index={0,1,2};
    t1[index]=true;//This is equivalent to t1[0,1,2]=true
    std::cout<<"t2[0,1,2]="<<t2[index]<<std::endl;
    //Offset
    std::cout<<"Offset of 0,1,2 is "<<t1.offset(index)<<std::endl;
    //Slicing
    Ouroboros::Shape start={0,0,0};
    Ouroboros::Shape step={1,1,1};
    Ouroboros::Shape end={1,2,3};
    Ouroboros::BoolTensor t6=t2.slice(start,end,step);//This is equivalent to t1[0:1,0:2,0:3] in numpy
    std::cout<<"t6="<<t6<<std::endl;
    /*
    Ouroboros::BoolTensor t6=t1.slice(start,end); // If we do something like this the step={1,1...}
    Ouroboros::BoolTensor t6=t1.slice(start,end,2); // If we do something like this the step={2,2...}
    */  
    //Getting raw data
    bool* raw_data=t1.data();
    //Getting the shape
    Ouroboros::Shape s=t1.shape();
    std::cout<<"Shape of t1="<<s<<std::endl;
    //Getting the strides
    Ouroboros::Shape strides=t1.strides();
    std::cout<<"Strides of t1="<<strides<<std::endl;
    //Getting the count i.e. the number of elements in the tensor
    std::size_t count=t1.count();
    std::cout<<"Count of t1="<<count<<std::endl;
    //Getting the no of dimensions
    std::size_t dim=t1.dim();
    std::cout<<"Dim of t1="<<dim<<std::endl;
    //Operators 
    Ouroboros::BoolTensor t7({2,3,4},true);
    Ouroboros::BoolTensor t8({2,3,4},false);
    Ouroboros::BoolTensor t9=t7==t8;//Does element wise comparison
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