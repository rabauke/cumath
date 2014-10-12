// Copyright (c) 2014, Heiko Bauke
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//   * Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.  
//
//   * Redistributions in binary form must reproduce the above
//     copyright notice, this list of conditions and the following
//     disclaimer in the documentation and/or other materials provided
//     with the distribution.  
//
//   * Neither the name of the copyright holder nor the names of its
//     contributors may be used to endorse or promote products derived
//     from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
// OF THE POSSIBILITY OF SUCH DAMAGE.

#if !(defined CUMATH_BLAS_HPP)

#define CUMATH_BLAS_HPP

#include <cumath/cuda.hpp>
#include <cumath/error.hpp>
#include <cumath/vector.hpp>
#include <cumath/complex.hpp>
#include <cublas_v2.h>

namespace cumath {

  class blas_error : public error {
  public:
    explicit blas_error(cublasStatus_t err_code) throw() {
      switch (err_code) {
        case CUBLAS_STATUS_NOT_INITIALIZED:
          err_str="cuBLAS library was not initialized";
          break;
        case CUBLAS_STATUS_ALLOC_FAILED:
          err_str="buffer allocation failed";
          break;
        case CUBLAS_STATUS_INVALID_VALUE:
          err_str="unsupported value or parameter";
          break;
        case CUBLAS_STATUS_ARCH_MISMATCH:
          err_str="required feature absent from device architecture";
          break;
        case CUBLAS_STATUS_MAPPING_ERROR:
          err_str="access to GPU memory space failed";
          break;
        case CUBLAS_STATUS_EXECUTION_FAILED:
          err_str="function failed to launch on the GPU";
          break;
        case CUBLAS_STATUS_INTERNAL_ERROR:
          err_str="internal cuBLAS operation failed";
          break;
        case CUBLAS_STATUS_NOT_SUPPORTED:
          err_str="functionnality not supported";
          break;
        default:
          err_str="unknown error";
      }
    };
    blas_error(const blas_error &other) throw() {
      err_str=other.err_str;
    }
    blas_error &operator=(const blas_error &other) throw() {
      err_str=other.err_str;
      return *this;
    }
  };

  //--------------------------------------------------------------------

  class blas_handle {
  private:
    cublasHandle_t handle;
  public:
    blas_handle() {
      cublasStatus_t err=cublasCreate(&handle);
      if (err!=CUBLAS_STATUS_SUCCESS)
	throw blas_error(err);
    }
    ~blas_handle() {
      cublasDestroy(handle);
    }
  private:
    blas_handle(const blas_handle &);
    blas_handle &operator=(const blas_handle &);
  public:
    cublasHandle_t operator()() const {
      return handle;
    }
  };

  //--------------------------------------------------------------------

  namespace detail {

    void copy(const blas_handle &handle, const float *x, float *y, int n) {
      cublasStatus_t err=cublasScopy(handle(), n, x, 1, y, 1);
      if (err!=CUBLAS_STATUS_SUCCESS)
	throw blas_error(err);
    }

    void copy(const blas_handle &handle, const double *x, double *y, int n) {
      cublasStatus_t err=cublasDcopy(handle(), n, x, 1, y, 1);
      if (err!=CUBLAS_STATUS_SUCCESS)
	throw blas_error(err);
    }

    void copy(const blas_handle &handle, const qwave::complex<float> *x, qwave::complex<float> *y, int n) {
      cublasStatus_t err=cublasCcopy(handle(), n, reinterpret_cast<const cuComplex *>(x), 1, reinterpret_cast<cuComplex *>(y), 1);
      if (err!=CUBLAS_STATUS_SUCCESS)
	throw blas_error(err);
    }

    void copy(const blas_handle &handle, const qwave::complex<double> *x, qwave::complex<double> *y, int n) {
      cublasStatus_t err=cublasZcopy(handle(), n, reinterpret_cast<const cuDoubleComplex *>(x), 1, reinterpret_cast<cuDoubleComplex *>(y), 1);
      if (err!=CUBLAS_STATUS_SUCCESS)
	throw blas_error(err);
    }
      
  }
    
  template<typename T, typename Alloc>
  void copy(const blas_handle &handle, const vector_view<T, Alloc> &x, vector_view<T, Alloc> &y) {
    detail::copy(handle, x.begin(), y.begin(), static_cast<int>(x.size()));
  }

  //--------------------------------------------------------------------

  namespace detail {

    void dot(const blas_handle &handle, const float *x, const float *y, int n, float *res) {
      cublasStatus_t err=cublasSdot(handle(), n, x, 1, y, 1, res);
      if (err!=CUBLAS_STATUS_SUCCESS)
	throw blas_error(err);
    }

    void dot(const blas_handle &handle, const double *x, const double *y, int n, double *res) {
      cublasStatus_t err=cublasDdot(handle(), n, x, 1, y, 1, res);
      if (err!=CUBLAS_STATUS_SUCCESS)
	throw blas_error(err);
    }

    void dot(const blas_handle &handle, const qwave::complex<float> *x, const qwave::complex<float> *y, int n, qwave::complex<float> *res) {
      cublasStatus_t err=cublasCdotu(handle(), n, reinterpret_cast<const cuComplex *>(x), 1, reinterpret_cast<const cuComplex *>(y), 1, reinterpret_cast<cuComplex *>(res));
      if (err!=CUBLAS_STATUS_SUCCESS)
	throw blas_error(err);
    }

    void dot(const blas_handle &handle, const qwave::complex<double> *x, const qwave::complex<double> *y, int n, qwave::complex<double> *res) {
      cublasStatus_t err=cublasZdotu(handle(), n, reinterpret_cast<const cuDoubleComplex *>(x), 1, reinterpret_cast<const cuDoubleComplex *>(y), 1, reinterpret_cast<cuDoubleComplex *>(res));
      if (err!=CUBLAS_STATUS_SUCCESS)
	throw blas_error(err);
    }

  }

  template<typename T, typename Alloc>
  T dot(const blas_handle &handle, const vector_view<T, Alloc> &x, const vector_view<T, Alloc> &y) {
    T res;
    detail::dot(handle, x.begin(), y.begin(), static_cast<int>(x.size()), &res);
    return res;
  }

  //--------------------------------------------------------------------

  namespace detail {

    void hdot(const blas_handle &handle, const qwave::complex<float> *x, const qwave::complex<float> *y, int n, qwave::complex<float> *res) {
      cublasStatus_t err=cublasCdotc(handle(), n, reinterpret_cast<const cuComplex *>(x), 1, reinterpret_cast<const cuComplex *>(y), 1, reinterpret_cast<cuComplex *>(res));
      if (err!=CUBLAS_STATUS_SUCCESS)
	throw blas_error(err);
    }

    void hdot(const blas_handle &handle, const qwave::complex<double> *x, const qwave::complex<double> *y, int n, qwave::complex<double> *res) {
      cublasStatus_t err=cublasZdotc(handle(), n, reinterpret_cast<const cuDoubleComplex *>(x), 1, reinterpret_cast<const cuDoubleComplex *>(y), 1, reinterpret_cast<cuDoubleComplex *>(res));
      if (err!=CUBLAS_STATUS_SUCCESS)
	throw blas_error(err);
    }

  }

  template<typename T, typename Alloc>
  qwave::complex<T> hdot(const blas_handle &handle, const vector_view<qwave::complex<T>, Alloc> &x, const vector_view<qwave::complex<T>, Alloc> &y) {
    typename vector_view<qwave::complex<T>, Alloc>::value_type res;
    detail::hdot(handle, x.begin(), y.begin(), static_cast<int>(x.size()), &res);
    return res;
  }

  //--------------------------------------------------------------------

  namespace detail {

    void norm(const blas_handle &handle, const float *x, int n, float *res) {
      cublasStatus_t err=cublasSnrm2(handle(), n, x, 1, res);
      if (err!=CUBLAS_STATUS_SUCCESS)
	throw blas_error(err);
    }

    void norm(const blas_handle &handle, const double *x, int n, double *res) {
      cublasStatus_t err=cublasDnrm2(handle(), n, x, 1, res);
      if (err!=CUBLAS_STATUS_SUCCESS)
	throw blas_error(err);
    }

    void norm(const blas_handle &handle, const qwave::complex<float> *x, int n, float *res) {
      cublasStatus_t err=cublasScnrm2(handle(), n, reinterpret_cast<const cuComplex *>(x), 1, res);
      if (err!=CUBLAS_STATUS_SUCCESS)
	throw blas_error(err);
    }

    void norm(const blas_handle &handle, const qwave::complex<double> *x, int n, double *res) {
      cublasStatus_t err=cublasDznrm2(handle(), n, reinterpret_cast<const cuDoubleComplex *>(x), 1, res);
      if (err!=CUBLAS_STATUS_SUCCESS)
	throw blas_error(err);
    }

  }

  template<typename T, typename Alloc>
  typename qwave::complex_traits<T>::real_type norm(const blas_handle &handle, const vector_view<T, Alloc> &x) {
    typename qwave::complex_traits<T>::real_type res;
    detail::norm(handle, x.begin(), static_cast<int>(x.size()), &res);
    return res;
  }

  //--------------------------------------------------------------------

  namespace detail {

    void scale(const blas_handle &handle, float *x, const float &a, int n) {
      cublasStatus_t err=cublasSscal(handle(), n, &a, x, 1);
      if (err!=CUBLAS_STATUS_SUCCESS)
	throw blas_error(err);
    }

    void scale(const blas_handle &handle, double *x, const double &a, int n) {
      cublasStatus_t err=cublasDscal(handle(), n, &a, x, 1);
      if (err!=CUBLAS_STATUS_SUCCESS)
	throw blas_error(err);
    }

    void scale(const blas_handle &handle, qwave::complex<float> *x, const qwave::complex<float> &a, int n) {
      cublasStatus_t err=cublasCscal(handle(), n, reinterpret_cast<const cuComplex *>(&a), reinterpret_cast<cuComplex *>(x), 1);
      if (err!=CUBLAS_STATUS_SUCCESS)
	throw blas_error(err);
    }

    void scale(const blas_handle &handle, qwave::complex<float> *x, const float &a, int n) {
      cublasStatus_t err=cublasCsscal(handle(), n, &a, reinterpret_cast<cuComplex *>(x), 1);
      if (err!=CUBLAS_STATUS_SUCCESS)
	throw blas_error(err);
    }

    void scale(const blas_handle &handle, qwave::complex<double> *x, const qwave::complex<double> &a, int n) {
      cublasStatus_t err=cublasZscal(handle(), n, reinterpret_cast<const cuDoubleComplex *>(&a), reinterpret_cast<cuDoubleComplex *>(x), 1);
      if (err!=CUBLAS_STATUS_SUCCESS)
	throw blas_error(err);
    }

    void scale(const blas_handle &handle, qwave::complex<double> *x, const double &a, int n) {
      cublasStatus_t err=cublasZdscal(handle(), n, &a, reinterpret_cast<cuDoubleComplex *>(x), 1);
      if (err!=CUBLAS_STATUS_SUCCESS)
	throw blas_error(err);
    }

  }

  template<typename T, typename Alloc>
  void scale(const blas_handle &handle, vector_view<T, Alloc> &x, const T &a) {
    detail::scale(handle, x.begin(), a, static_cast<int>(x.size()));
  }

  template<typename T, typename Alloc>
  void scale(const blas_handle &handle, vector_view<qwave::complex<T>, Alloc> &x, const T &a) {
    detail::scale(handle, x.begin(), a, static_cast<int>(x.size()));
  }

  //--------------------------------------------------------------------

  namespace detail {

    void axpy(const blas_handle &handle, const float &a, const float *x, float *y, int n) {
      cublasStatus_t err=cublasSaxpy(handle(), n, &a, x, 1, y, 1);
      if (err!=CUBLAS_STATUS_SUCCESS)
	throw blas_error(err);
    }

    void axpy(const blas_handle &handle, const double &a, const double *x, double *y, int n) {
      cublasStatus_t err=cublasDaxpy(handle(), n, &a, x, 1, y, 1);
      if (err!=CUBLAS_STATUS_SUCCESS)
	throw blas_error(err);
    }

    void axpy(const blas_handle &handle, const qwave::complex<float> &a, const qwave::complex<float> *x, qwave::complex<float> *y, int n) {
      cublasStatus_t err=cublasCaxpy(handle(), n, reinterpret_cast<const cuComplex *>(&a), reinterpret_cast<const cuComplex *>(x), 1, reinterpret_cast<cuComplex *>(y), 1);
      if (err!=CUBLAS_STATUS_SUCCESS)
	throw blas_error(err);
    }

    void axpy(const blas_handle &handle, const qwave::complex<double> &a, const qwave::complex<double> *x, qwave::complex<double> *y, int n) {
      cublasStatus_t err=cublasZaxpy(handle(), n, reinterpret_cast<const cuDoubleComplex *>(&a), reinterpret_cast<const cuDoubleComplex *>(x), 1, reinterpret_cast<cuDoubleComplex *>(y), 1);
      if (err!=CUBLAS_STATUS_SUCCESS)
	throw blas_error(err);
    }

  }

  template<typename T, typename Alloc>
  void axpy(const blas_handle &handle, const T &a, const vector_view<T, Alloc> &x, vector_view<T, Alloc> &y) {
    detail::axpy(handle, a, x.begin(), y.begin(), static_cast<int>(x.size()));
  }

}

#endif
