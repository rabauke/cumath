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

#if !(defined CUMATH_ALLOCATOR_HPP)

#define CUMATH_ALLOCATOR_HPP

#include <limits>
#include <cstddef>
#include <iterator>
#include <memory>
#include <cumath/cuda.hpp>
#include <cumath/error.hpp>

namespace cumath {
  
  class device_memory_trait {
  public:
    static void * malloc(::std::size_t size) {
      void * p;
      cudaError_t err=cudaMalloc(&p, size);
      if (err!=cudaSuccess)
	throw error(err);
      return p;
    }
    static void free(void *p) {
      cudaError_t err=cudaFree(p);
      if (err!=cudaSuccess)
	throw error(err);
    }
  };
    
  class page_locked_memory_trait {
  public:
    static void * malloc(::std::size_t size) {
      void * p;
      cudaSetDeviceFlags(cudaDeviceMapHost);
      cudaError_t err=cudaMallocHost(&p, size);
      if (err!=cudaSuccess)
	throw error(err);
      return p;
    }
    static void free(void *p) {
      cudaError_t err=cudaFreeHost(p);
      if (err!=cudaSuccess)
	throw error(err);
    }
  };
    
  class managed_memory_trait {
  public:
    static void * malloc(::std::size_t size) {
      void * p;
      cudaError_t err=cudaMallocManaged(&p, size, cudaMemAttachGlobal);
      if (err!=cudaSuccess)
	throw error(err);
      return p;
    }
    static void free(void *p) {
      cudaError_t err=cudaFree(p);
      if (err!=cudaSuccess)
	throw error(err);
    }
  };
  
  template<typename T, typename trait>
  class base_allocator {
  public:
    // type definitions
    typedef T value_type;
    typedef T * pointer;
    typedef const T * const_pointer;
    typedef T & reference;
    typedef const T & const_reference;
    typedef ::std::size_t size_type;
    typedef ::std::ptrdiff_t difference_type;
        
    // return address of values
    pointer address(reference value) const {
      return &value;
    }
    const_pointer address(const_reference value) const {
      return &value;
    }
    
    /* constructors and destructor
     * - nothing to do because the allocator has no state
     */
    base_allocator() throw() {
    }
    base_allocator(const base_allocator &) throw() {
    }
    template<typename U>
    base_allocator(const base_allocator<U, trait> &) throw() {
    }
    ~base_allocator() throw() {
    }
    
    // return maximum number of elements that can be allocated
    size_type max_size() const throw() {
      return ::std::numeric_limits< ::std::size_t>::max()/sizeof(T);
    }
    
    // allocate but don't initialize num elements of type T
    pointer allocate(size_type num, const void *hint=0) {
      return reinterpret_cast<pointer>(trait::malloc(num*sizeof(T)));
    }
    
    // // initialize elements of allocated storage p with value value
    // void construct(pointer p, const T &value) {
    //   // initialize memory with placement new
    //   new (reinterpret_cast<void *>(p)) T(value);
    // }
    
    // // destroy elements of initialized storage p
    // void destroy(pointer p) {
    //   // destroy objects by calling their destructor
    //   p->~T();
    // }
    
    // deallocate storage p of deleted elements
    void deallocate(pointer p, size_type num) {
      trait::free(reinterpret_cast<void *>(p));
    }
  };

  // return that all specializations of this allocator are interchangeable
  template<typename T1, typename T2, typename trait>
  bool operator==(const base_allocator<T1, trait> &, const base_allocator<T2, trait> &) throw() {
    return true;
  }
  template<typename T1, typename T2, typename trait>
  bool operator!=(const base_allocator<T1, trait>&, const base_allocator<T2, trait> &) throw() {
    return false;
  }
  
  template<typename T>
  class device_allocator : public base_allocator<T, device_memory_trait> {
    typedef base_allocator<T, device_memory_trait> base;
  public:
    typedef typename base::value_type value_type;
    typedef typename base::pointer pointer;
    typedef typename base::const_pointer const_pointer;
    typedef typename base::reference reference;
    typedef typename base::const_reference const_reference;
    typedef typename base::size_type size_type;
    typedef typename base::difference_type difference_type;

    using base::allocate;
    using base::deallocate;

    // rebind allocator to type U
    template<typename U>
    struct rebind {
      typedef device_allocator<U> other;
    };

    device_allocator() throw() {
    }
    device_allocator(const device_allocator &) throw() {
    }
    template<typename U>
    device_allocator(const device_allocator<U> &) throw() {
    }
    ~device_allocator() throw() {
    }
  };

  template<typename T>
  class page_locked_allocator : public base_allocator<T, page_locked_memory_trait> {
    typedef base_allocator<T, page_locked_memory_trait> base;
  public:
    typedef typename base::value_type value_type;
    typedef typename base::pointer pointer;
    typedef typename base::const_pointer const_pointer;
    typedef typename base::reference reference;
    typedef typename base::const_reference const_reference;
    typedef typename base::size_type size_type;
    typedef typename base::difference_type difference_type;
    
    // rebind allocator to type U
    template<typename U>
    struct rebind {
      typedef page_locked_allocator<U> other;
    };

    page_locked_allocator() throw() {
    }
    page_locked_allocator(const page_locked_allocator &) throw() {
    }
    template<typename U>
    page_locked_allocator(const page_locked_allocator<U> &) throw() {
    }
    ~page_locked_allocator() throw() {
    }
    // initialize elements of allocated storage p with value value
    void construct(pointer p, const T &value) {
      // initialize memory with placement new
      new (reinterpret_cast<void *>(p)) T(value);
    }
    // destroy elements of initialized storage p
    void destroy(pointer p) {
      // destroy objects by calling their destructor
      p->~T();
    }
  };

  template<typename T>
  class managed_allocator : public base_allocator<T, managed_memory_trait> {
    typedef base_allocator<T, managed_memory_trait> base;
  public:
    typedef typename base::value_type value_type;
    typedef typename base::pointer pointer;
    typedef typename base::const_pointer const_pointer;
    typedef typename base::reference reference;
    typedef typename base::const_reference const_reference;
    typedef typename base::size_type size_type;
    typedef typename base::difference_type difference_type;
    
    // rebind allocator to type U
    template<typename U>
    struct rebind {
      typedef managed_allocator<U> other;
    };

    managed_allocator() throw() {
    }
    managed_allocator(const managed_allocator &) throw() {
    }
    template<typename U>
    managed_allocator(const managed_allocator<U> &) throw() {
    }
    ~managed_allocator() throw() {
    }
    // initialize elements of allocated storage p with value value
    void construct(pointer p, const T &value) {
      // initialize memory with placement new
      new (reinterpret_cast<void *>(p)) T(value);
    }
    // destroy elements of initialized storage p
    void destroy(pointer p) {
      // destroy objects by calling their destructor
      p->~T();
    }
  };

}

#endif
