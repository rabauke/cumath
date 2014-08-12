#include <cstdlib>
#include <iostream>

// define necessary traits to use cumath::complex with boost ublas classes und functions

#include <limits>
#include <boost/numeric/ublas/traits.hpp>
#include <cumath/complex.hpp>
 
namespace cumath {

  template<class T>
  struct complex_traits {
    typedef complex_traits<T> self_type;
    typedef T value_type;
    typedef const T & const_reference;
    typedef T & reference;

    typedef typename T::value_type real_type;
    typedef real_type precision_type;

    static const unsigned plus_complexity = 2;
    static const unsigned multiplies_complexity = 6;

    static
    BOOST_UBLAS_INLINE
    real_type real(const_reference t) {
      return cumath::real(t);
    }
    static
    BOOST_UBLAS_INLINE
    real_type imag(const_reference t) {
      return cumath::imag(t);
    }
    static
    BOOST_UBLAS_INLINE
    value_type conj(const_reference t) {
      return cumath::conj(t);
    }

    static
    BOOST_UBLAS_INLINE
    real_type type_abs(const_reference t) {
      return cumath::abs(t);
    }
    static
    BOOST_UBLAS_INLINE
    value_type type_sqrt(const_reference t) {
      return cumath::sqrt(t);
    }

    static
    BOOST_UBLAS_INLINE
    real_type norm_1(const_reference t) {
      return self_type::type_abs(t);
    }
    static
    BOOST_UBLAS_INLINE
    real_type norm_2(const_reference t) {
      return self_type::type_abs(t);
    }
    static
    BOOST_UBLAS_INLINE
    real_type norm_inf(const_reference t) {
      return self_type::type_abs(t);
    }
    static
    BOOST_UBLAS_INLINE
    bool equals(const_reference t1, const_reference t2) {
      return self_type::norm_inf(t1-t2)<
	sqrt(std::numeric_limits<real_type>::epsilon())*std::max(std::max(self_type::norm_inf(t1), self_type::norm_inf(t2)), sqrt(std::numeric_limits<real_type>::min()));
    }
  };
  
}

namespace boost { 
  namespace numeric { 
    namespace ublas {
      
      template<typename T>
      struct type_traits< ::cumath::complex<T> > : ::cumath::complex_traits< ::cumath::complex<T> > {
	typedef type_traits< ::cumath::complex<T> > self_type;
	typedef ::cumath::complex<T> value_type;
	typedef const ::cumath::complex<T> & const_reference;
	typedef ::cumath::complex<T> & reference;
	typedef T real_type;
	typedef real_type precision_type;
      };

    }
  }
}


namespace ublas=boost::numeric::ublas;
typedef cumath::complex<float> complex_t;

int main() {
  complex_t a(1, 2);
  std::cout << ublas::type_traits<complex_t>::conj(a) << '\n';
  return EXIT_SUCCESS;
}
