#include <ouroboros/ouroboros.hpp>
#include <iostream>

int main(){
    //Various ways to initialize the Shape class
    Ouroboros::Shape shape1(3,(size_t)2);//Creates a shape of 3 items each of value 2. 
    std::cout<<"shape1="<<shape1<<std::endl;

    Ouroboros::Shape shape2={1,2,3};
    std::cout<<"shape2="<<shape2<<std::endl;

    Ouroboros::Shape shape3(shape2);
    std::cout<<"shape3="<<shape3<<std::endl;

    Ouroboros::Shape shape4(std::move(shape3));
    std::cout<<"shape4="<<shape4<<std::endl;

    size_t array[]={1,2,3};
    Ouroboros::Shape shape5(3, array);//Takes in a pointer to an array and copies it
    std::cout<<"shape5="<<shape5<<std::endl;

    //Assignment operator
    shape1=shape2;
    std::cout<<"shape1="<<shape1<<std::endl;
    shape1={1,2,3,5,6};
    std::cout<<"shape1="<<shape1<<std::endl;

    //Indexing
    std::cout<<"Item at idx 0 is "<<shape1[0]<<std::endl;
    //Modifying a value
    shape1.set(0,5);//Set the value at index 0 to 5
    std::cout<<"shape1="<<shape1<<std::endl;
    //Note:- Something like shape1[0]=5 is not allowed

    //Iterating over the shape
    const size_t* start=shape1.begin();//Get the pointer to the start of the shape
    const size_t* end=shape1.end();//Get the pointer to the end of the shape
    for(const size_t* i=start;i!=end;i++){
        std::cout<<*i<<" ";
    }
    for(auto i:shape1){
        std::cout<<i<<" ";
    }
    std::cout<<std::endl;

    //Comparing shapes
    std::cout<<"shape1==shape2 is "<<(shape1==shape2)<<std::endl;
    std::cout<<"shape1!=shape2 is "<<(shape1!=shape2)<<std::endl;

    //Getting the count and dim of the shape
    std::cout<<"shape1.count()="<<shape1.count()<<std::endl;//the product of all the elements in the shape.Useful for calculating no of items in tensor
    std::cout<<"shape1.dim()="<<shape1.dim()<<std::endl;//No of elements in the shape 
}