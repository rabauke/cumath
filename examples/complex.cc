// compile with:
// nvcc -o complex -x cu -arch=compute_20 -I .. complex.cc
// g++ -o complex -I .. complex.cc

#include <cstdlib>
#include <iostream>
#include <cumath/math.hpp>
#include <cumath/complex.hpp>

int main() {
  cumath::complex<double> a(1, 2);
  cumath::complex<double> b(2, 3);
  std::cout << a << " + " << b << " = " << (a+b) << '\n';
  return EXIT_SUCCESS;
}
