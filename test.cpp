// #include <stdint.h>
#include <iostream>

int main(int argc, char** argv){
    std::cout << "Im here" << std::endl;
    uint64_t a = 0xdddddddddddddddd;
    uint64_t b = 0xeeeeeeeeeeeeeeee;
    uint64_t p = 0xffffffffffffffff;

    std::cout << "a= " << a << "\tb=" << b << "\tp=" << p << std::endl;

    uint64_t c = a*b;
    uint64_t cp = (a*b)%p;

    std::cout << "c= " << c << "\tcp=" << cp << std::endl;
}

