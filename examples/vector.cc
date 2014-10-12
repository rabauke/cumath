#include <cstdlib>
#include <iostream>
#include <vector>
#include <cumath/allocator.hpp>
#include <cumath/vector.hpp>
#include <cumath/complex.hpp>

template<typename T, typename view>
__global__ void mult(T factor, view v) {
  typename view::size_type j=blockIdx.x*blockDim.x+threadIdx.x;
  if (j<v.size())
    v[j]*=factor;
}

int main() {
  typedef cumath::complex<double> complex;
  {
    std::size_t n(1234);
    std::vector<complex> v1(n);
    for (int i(0); i<n; ++i) 
      v1[i]=complex(i, -i);
    cumath::vector<complex> v2(n);
    cumath::copy(v1.begin(), v1.end(), v2.begin());
    int threads_per_block=512;
    int blocks=(n+threads_per_block-1)/threads_per_block;
    mult<<<blocks, threads_per_block>>>(complex(2, 0), v2.view());
    cumath::vector<complex> v3(v2);
    cumath::copy(v3.begin(), v3.end(), v1.begin());
    for (std::size_t i(0); i<n; ++i) 
      std::cout << v1[i] << '\n';
  }

  {
    std::size_t n(1234);
    std::vector<complex> v1(n);
    for (int i(0); i<n; ++i) 
      v1[i]=complex(i, -i);
    cumath::vector<complex, cumath::page_locked_allocator<complex> > v2(n);
    cumath::copy(v1.begin(), v1.end(), v2.begin());
    int threads_per_block=512;
    int blocks=(n+threads_per_block-1)/threads_per_block;
    mult<<<blocks, threads_per_block>>>(complex(3, 0), v2.view());
    cumath::vector<complex, cumath::page_locked_allocator<complex> > v3(v2);
    cumath::copy(v3.begin(), v3.end(), v1.begin());
    for (std::size_t i(0); i<n; ++i) 
      std::cout << v1[i] << '\n';
  }

  {
    std::size_t n(1234);
    cumath::vector<complex, cumath::managed_allocator<complex> > v1(n);
    for (int i(0); i<n; ++i)
      v1[i]=complex(i, -i);
    int threads_per_block=512;
    int blocks=(n+threads_per_block-1)/threads_per_block;
    mult<<<blocks, threads_per_block>>>(complex(4, 0), v1.view());
    cudaDeviceSynchronize();
    for (std::size_t i(0); i<n; ++i) 
      std::cout << v1[i] << '\n';
  }
  
  return EXIT_SUCCESS;
}
