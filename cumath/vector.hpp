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

#if !(defined CUMATH_VECTOR_HPP)

#define CUMATH_VECTOR_HPP

#include <algorithm>
#include <cumath/cuda.hpp>
#include <cumath/allocator.hpp>
#include <cumath/static_assertion.hpp>

namespace cumath {

  template<typename T, typename Alloc=device_allocator<T> >
  class vector_view;

  template<typename T, typename Alloc=device_allocator<T> >
  class vector;

  template<typename T, typename Alloc>
  void swap(vector<T, Alloc> &x, vector<T, Alloc> &y); 

  template<typename in_iter, typename out_iter>
  out_iter copy(in_iter first, in_iter last, out_iter to);

  template<typename T, typename Alloc>
  class vector_view {
  public:
    typedef T value_type;
    typedef value_type & reference;
    typedef const value_type & const_reference;	
    typedef T * pointer;
    typedef const T * const_pointer;
    typedef pointer iterator;
    typedef const_pointer const_iterator;
    typedef typename std::iterator_traits<iterator>::difference_type difference_type;
    typedef std::size_t size_type;
  private:
    size_type n;
    pointer p;
  protected:
    vector_view() {
    }
  public:
    CUMATH_HOST_DEVICE
    iterator begin() { return p; }
    CUMATH_HOST_DEVICE
    iterator end() { return p+n; }
    CUMATH_HOST_DEVICE
    const_iterator begin() const { return p; }
    CUMATH_HOST_DEVICE
    const_iterator end() const { return p+n; }
    CUMATH_HOST_DEVICE
    const_iterator cbegin() const { return p; }
    CUMATH_HOST_DEVICE
    const_iterator cend() const { return p+n; }
    CUMATH_HOST_DEVICE
    size_type size() const { return n; }
    CUMATH_HOST_DEVICE
    reference operator[](size_type i) {
#if !(defined __CUDA_ARCH__)
      static_assertion<sametype<Alloc, managed_allocator<T> >::result>();
#endif
      return p[i]; 
    }
    CUMATH_HOST_DEVICE
    const_reference operator[](size_type i) const { 
#if !(defined __CUDA_ARCH__)
      static_assertion<sametype<Alloc, managed_allocator<T> >::result>();
#endif
      return p[i]; 
    }
    void swap(vector_view &other) {
      std::swap(n, other.n);
      std::swap(p, other.p);
    }
    friend class vector<T, Alloc>; 
  };

  template<typename T, typename Alloc>
  class vector : public vector_view<T, Alloc> {
    typedef vector_view<T, Alloc> base;
  public:
    typedef T value_type;
    typedef Alloc allocator_type;
    typedef value_type & reference;
    typedef const value_type & const_reference;	
    typedef typename allocator_type::pointer pointer;
    typedef typename allocator_type::const_pointer const_pointer;
    typedef pointer iterator;
    typedef const_pointer const_iterator;
    typedef typename std::iterator_traits<iterator>::difference_type difference_type;
    typedef typename allocator_type::size_type size_type;
    typedef base view_type;
  private:
    allocator_type alloc;
    using base::p;
    using base::n;
  public:
    explicit vector(size_type n_, const allocator_type &alloc_=allocator_type()) : 
      alloc(alloc_) {
      n=n_;
      p=alloc.allocate(n);
    }
    explicit vector(const vector &other) : 
      alloc(other.alloc) {
      n=other.n;
      p=alloc.allocate(n);
      copy(other.begin(), other.end(), begin());
    }
    ~vector() {
      alloc.deallocate(p, n);
    }
    vector & operator=(const vector &other) {
      if (this!=&other) {
	alloc.deallocate(p, n);
	alloc=other.alloc;
	n=other.n;
	p=alloc.allocate(n);
	copy(other.begin(), other.end(), begin());
      }
      return *this;
    }
    using base::begin;
    using base::end;
    using base::cbegin;
    using base::cend;
    using base::operator[];
    void swap(vector &other) {
      base::swap(other);
      std::swap(alloc, other.alloc);
    }
    view_type view() const {
      return static_cast<view_type>(*this);
    }
  };

  template<typename T, typename Alloc>
  void swap(vector<T, Alloc> &x, vector<T, Alloc> &y) {
    x.swap(y);
  }  

  template<typename in_iter, typename out_iter>
  out_iter copy(in_iter first, in_iter last, out_iter to) {
    static_assertion<sametype<typename std::iterator_traits<in_iter>::value_type,
			      typename std::iterator_traits<out_iter>::value_type>::result>();
    typename std::iterator_traits<in_iter>::difference_type n(last-first);
    cudaError_t err=cudaMemcpy(&*(to), &*(first),
			       n*sizeof(typename std::iterator_traits<in_iter>::value_type),
			       cudaMemcpyDefault);
    if (err!=cudaSuccess)
      throw error(err);
    return to+n;
  }
  
}

#endif
