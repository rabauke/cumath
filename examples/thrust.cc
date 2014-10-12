// compile with
// nvcc -o thrust -O3 -x cu -arch=compute_20 -I .. thrust.cc
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/inner_product.h>
#include <thrust/functional.h>
#include <iostream>
#include <cstdlib>
#include <cumath/math.hpp>
#include <cumath/complex.hpp>
#include <cublas_v2.h>
#include "timer.hpp"

typedef double real_t;
typedef cumath::complex<real_t> complex_t;


namespace thrust {

  template<typename T>
  struct conj_multiplies : public binary_function<T, T, T> {
    __host__ __device__ T operator()(const T &x, const T &y) const {
      return complex_t(x.real()*y.real()+x.imag()*y.imag(),
		       x.real()*y.imag()-x.imag()*y.real());
    }
  };
  
}


inline
complex_t inner_product(const thrust::device_vector<complex_t> &x,
			const thrust::device_vector<complex_t> &y) {
  return
    thrust::inner_product(x.begin(), x.end(), y.begin(), 
			  complex_t(0, 0), 
			  thrust::plus<complex_t>(), 
			  thrust::conj_multiplies<complex_t>());
}


int main() {
  int N=1024*1024*8;
  thrust::host_vector<complex_t> Xh(N);
  thrust::host_vector<complex_t> Yh(N);
  for (int i(0); i<N; ++i) {
    real_t re(static_cast<real_t>(std::rand())/RAND_MAX);
    real_t im(static_cast<real_t>(std::rand())/RAND_MAX);
    re=2*re-1;
    im=2*im-1;
    Xh[i]=complex_t(re, im);
    re=static_cast<real_t>(std::rand())/RAND_MAX;
    im=static_cast<real_t>(std::rand())/RAND_MAX;
    re=2*re-1;
    im=2*im-1;
    Yh[i]=complex_t(re, im);
  }
  thrust::device_vector<complex_t> Xd(N);
  thrust::device_vector<complex_t> Yd(N);
  thrust::copy(Xh.begin(), Xh.end(), Xd.begin());
  thrust::copy(Yh.begin(), Yh.end(), Yd.begin());
  {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    timer::timer T;
    complex_t dsum(inner_product(Xd, Yd));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << dsum << '\t' << T.time() << '\t' << elapsedTime/1000 << '\n';
  }
  {
    cublasHandle_t handle;
    cublasCreate(&handle);
    timer::timer T;
    complex_t hsum(0, 0);
    cublasZdotc(handle, N, 
		reinterpret_cast<cuDoubleComplex *>(thrust::raw_pointer_cast(Xd.data())), 1,
		reinterpret_cast<cuDoubleComplex *>(thrust::raw_pointer_cast(Yd.data())), 1,
		reinterpret_cast<cuDoubleComplex *>(&hsum));
    //cudaDeviceSynchronize();
    std::cout << hsum << '\t' << T.time() << '\n';
    cublasDestroy(handle);
  }
  {
    timer::timer T;
    complex_t hsum(0, 0);
    for (int i(0); i<N; ++i)
      hsum+=complex_t(Xh[i].real()*Yh[i].real()+Xh[i].imag()*Yh[i].imag(),
		      Xh[i].real()*Yh[i].imag()-Xh[i].imag()*Yh[i].real());
    std::cout << hsum << '\t' << T.time() << '\n';
  }
  return 0;
}
