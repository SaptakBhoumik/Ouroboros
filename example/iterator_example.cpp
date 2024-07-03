#include <ouroboros/ouroboros.hpp>
#include <iostream>
int main(){
    double data[10]={1,2,3,4,5,6,7,8,9,10};
    // Note the iterator does not copy the data
    // It just stores the pointer to the data and never modifies or deletes it
    Ouroboros::Utils::Iterator<double> it1(data,10);//10 is the number of times we can iterate.
    for(auto x:it1){//U cant use auto& x:it because it does not return a reference
        std::cout<<x<<" ";
    }
    std::cout<<"\n-------------------"<<std::endl;
    Ouroboros::Utils::Iterator<double> it2(data,5,2);//5 is the number of times we can iterate. 2 is the step size
    for(auto x:it2){
        std::cout<<x<<" ";
    }
    std::cout<<"\n-------------------"<<std::endl;
    it2=it1;
    for(auto x:it2){
        std::cout<<x<<" ";
    }
    return 0;
}