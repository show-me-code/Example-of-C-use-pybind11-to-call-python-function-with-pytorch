#include <pybind11/embed.h>
#include <iostream>
#include <pybind11/numpy.h>

#include <pybind11/stl.h> //used to convert
#include <chrono>   
namespace py = pybind11;
using namespace py::literals;

int main(){
    auto start = std::chrono::system_clock::now();
    py::scoped_interpreter guard{}; // call interperter
    float list[784000];
    const int length = 784000;
    for (int i = 0; i < length; i++)
    {
        list[i] = float(i);
    }
    std::vector<float> v(list, list+length);
    std::cout << v.size() << std::endl;
    py::array_t<float> args = py::cast(v); //create a copy
    
    py::module_ nd_to_tensor = py::module_::import("inference"); //import python module
    for (int i = 0; i < 10; i++)
    {
        py::object result = nd_to_tensor.attr("inference")(args); // call function
        const float* star = result.cast<py::array_t<float>>().data(); // use a pointer to point the result
        std::vector<float> afterReturn(star, star+ length); //a example of copying result to std vector
        auto end = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout <<  "cost"
        << double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den 
        << "second" << std::endl; //test performance
    }
    
        
}
