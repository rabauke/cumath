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

#if !(defined CUMATH_COMPLEX_HPP)

#define CUMATH_COMPLEX_HPP

#include <ios>
#include <ostream>
#include <istream>
#include <sstream>
#include <cumath/math.hpp>

// see http://www.cplusplus.com/reference/cmath/

namespace cumath {

  template<typename T>
  class complex_base;
  
  template<typename T>
  class complex_base {
  protected:
    T x[2];
  public:
    typedef T value_type;
    CUMATH_HOST_DEVICE
    complex_base() {
#if !(defined __CUDACC__)
      x[0]=T(0);
      x[1]=T(0);
#endif
    }
    CUMATH_HOST_DEVICE
    complex_base(const T &r) {
      x[0]=r;
      x[1]=0;
    }
    CUMATH_HOST_DEVICE
    complex_base(const T &r, const T &i) {
      x[0]=r;
      x[1]=i;
    }
    CUMATH_HOST_DEVICE
    T real() const {
      return x[0];
    }
    void real(const T &r) {
      x[0]=r;
    }
    CUMATH_HOST_DEVICE
    T imag() const {
      return x[1];
    }
    void imag(const T &i) {
      x[1]=i;
    }
    CUMATH_HOST_DEVICE
    complex_base & operator=(const T &val) {
      x[0]=val;
      x[1]=0;
      return *this;
    }
    CUMATH_HOST_DEVICE
    complex_base & operator=(const complex_base &val) {
      x[0]=val.x[0];
      x[1]=val.x[1];
      return *this;
    }
    CUMATH_HOST_DEVICE
    complex_base & operator+=(const T &val) {
      x[0]+=val;
      return *this;
    }
    template<typename T2>
    CUMATH_HOST_DEVICE
    complex_base & operator+=(const complex_base<T2> &val) {
      x[0]+=val.real();
      x[1]+=val.imag();
      return *this;
    }
    CUMATH_HOST_DEVICE
    complex_base & operator-=(const T &val) {
      x[0]-=val;
      return *this;
    }
    template<typename T2>
    CUMATH_HOST_DEVICE
    complex_base & operator-=(const complex_base<T2> &val) {
      x[0]-=val.real();
      x[1]-=val.imag();
      return *this;
    }
    CUMATH_HOST_DEVICE
    complex_base & operator*=(const T &val) {
      x[0]*=val;
      x[1]*=val;
      return *this;
    }
    template<typename T2>
    CUMATH_HOST_DEVICE
    complex_base & operator*=(const complex_base<T2> &val) {
      *this=complex_base(real()*val.real()-imag()*val.imag(),
			 imag()*val.real()+real()*val.imag());
      return *this;
    }
    CUMATH_HOST_DEVICE
    complex_base & operator/=(const T &val) {
      x[0]/=val;
      x[1]/=val;
      return *this;
    }
    template<typename T2>
    CUMATH_HOST_DEVICE
    complex_base & operator/=(const complex_base<T2> &val) {
      if (abs(real())>abs(val.imag())) {
	T norm=val.real()+val.imag()*val.imag()/val.real();
	*this=complex_base((real()+imag()*val.imag()/val.real())/norm,
			   (imag()-real()*val.imag()/val.real())/norm);
      } else {
	T norm=val.imag()+val.real()*val.real()/val.imag();
	*this=complex_base((imag()+real()*val.real()/val.imag())/norm,
			   (real()-imag()*val.real()/val.imag())/norm);
      }
      return *this;
    }

  };

  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> operator+(const complex_base<T> &lhs, const complex_base<T> &rhs) {
    return complex_base<T>(lhs.real()+rhs.real(),
			   lhs.imag()+rhs.imag());
  }
  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> operator+(const complex_base<T> &lhs, const T &val)  {
    return complex_base<T>(lhs.real()+val,
			   lhs.imag());
  }
  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> operator+(const T& val, const complex_base<T> &rhs) {
    return complex_base<T>(val+rhs.real(),
			   rhs.imag());
  }

  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> operator-(const complex_base<T> &lhs, const complex_base<T> &rhs) {
    return complex_base<T>(lhs.real()-rhs.real(),
			   lhs.imag()-rhs.imag());
  }
  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> operator-(const complex_base<T> &lhs, const T &val)  {
    return complex_base<T>(lhs.real()-val,
			   lhs.imag());
  }
  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> operator-(const T& val, const complex_base<T> &rhs) {
    return complex_base<T>(val-rhs.real(),
			   rhs.imag());
  }

  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> operator*(const complex_base<T> &lhs, const complex_base<T> &rhs) {
    return complex_base<T>(lhs.real()*rhs.real()-lhs.imag()*rhs.imag(),
			   lhs.imag()*rhs.real()+lhs.real()*rhs.imag());
  }
  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> operator*(const complex_base<T> &lhs, const T &val)  {
    return complex_base<T>(lhs.real()*val,
			   lhs.imag()*val);
  }
  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> operator*(const T& val, const complex_base<T> &rhs) {
    return complex_base<T>(val*rhs.real(),
			   val*rhs.imag());
  }

  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> operator/(const complex_base<T> &lhs, const complex_base<T> &rhs) {
    if (abs(rhs.real())>abs(rhs.imag())) {
      T norm=rhs.real()+rhs.imag()*rhs.imag()/rhs.real();
      return complex_base<T>((lhs.real()+lhs.imag()*rhs.imag()/rhs.real())/norm,
			     (lhs.imag()-lhs.real()*rhs.imag()/rhs.real())/norm);
    } else {
      T norm=rhs.imag()+rhs.real()*rhs.real()/rhs.imag();
      return complex_base<T>((lhs.imag()+lhs.real()*rhs.real()/rhs.imag())/norm,
			     (lhs.real()-lhs.imag()*rhs.real()/rhs.imag())/norm);
    }
  }
  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> operator/(const complex_base<T> &lhs, const T &val)  {
    return complex_base<T>(lhs.real()/val,
			   lhs.imag()/val);
  }
  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> operator/(const T& val, const complex_base<T> &rhs) {
    return complex_base<T>(val)/rhs;
  }

  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> operator+(const complex_base<T> &rhs) {
    return rhs;
  }
  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> operator-(const complex_base<T> &rhs) {
    return complex_base<T>(-rhs.real(), -rhs.imag());
  }

  template<typename T> 
  CUMATH_HOST_DEVICE
  bool operator==(const complex_base<T> &lhs, const complex_base<T> &rhs) {
    return lhs.real()==rhs.real() and lhs.imag()==rhs.imag();
  }
  template<typename T> 
  CUMATH_HOST_DEVICE
  bool operator==(const complex_base<T> &lhs, const T& val) {
    return lhs.real()==val and lhs.imag()==T(0);
  }
  template<typename T> 
  CUMATH_HOST_DEVICE
  bool operator==(const T& val, const complex_base<T> &rhs) {
    return val==rhs.real() and T(0)==rhs.imag();
  }

  template<typename T> 
  CUMATH_HOST_DEVICE
  bool operator!=(const complex_base<T> &lhs, const complex_base<T> &rhs) {
    return lhs.real()!=rhs.real() or lhs.imag()!=rhs.imag();
  }
  template<typename T> 
  CUMATH_HOST_DEVICE
  bool operator!=(const complex_base<T> &lhs, const T& val) {
    return lhs.real()!=val or lhs.imag()!=T(0);
  }
  template<typename T> 
  CUMATH_HOST_DEVICE
  bool operator!=(const T& val, const complex_base<T> &rhs) {
    return val!=rhs.real() or T(0)!=rhs.imag();
  }

  template<typename T> 
  CUMATH_HOST_DEVICE
  T real(const complex_base<T> &val) {
    return val.real();
  }

  template<typename T> 
  CUMATH_HOST_DEVICE
  T imag(const complex_base<T> &val) {
    return val.imag();
  }

  template<typename T> 
  CUMATH_HOST_DEVICE
  T abs(const complex_base<T> &val) {
    return hypot(val.real(), val.imag());
  }
  
  template<typename T> 
  CUMATH_HOST_DEVICE
  T arg(const complex_base<T> &val) {
    return atan2(val.imag(), val.real());
  }
  
  template<typename T> 
  CUMATH_HOST_DEVICE
  T sgn(const complex_base<T> &val) {
    return 0<val.real() or (0==val.real() and 0<=val.imag()) ? T(1) : T(-1); 
  }
  
  template<typename T> 
  CUMATH_HOST_DEVICE
  T norm(const complex_base<T> &val) {
    return val.real()*val.real()+val.imag()*val.imag();
  }
  
  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> polar(const T &r, const T &phi) {
    return complex_base<T>(r*cos(phi), r*sin(phi));
  }

  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> conj(const complex_base<T> &x) {
    return complex_base<T>(x.real(), -x.imag());
  }

  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> proj(const complex_base<T> &x) {
    return (isinf(x.real()) or isinf(x.imag())) ? complex_base<T>(T(1)/T(0), copysign(T(0), x.imag)) : x;
  }
  
  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> sin(const complex_base<T> &x) {
    return complex_base<T>(sin(x.real())*cosh(x.imag()), +cos(x.real)*sinh(x.imag()));
  }
  
  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> sinh(const complex_base<T> &x) {
    return complex_base<T>(sinh(x.real())*cos(x.imag()), +cosh(x.real)*sin(x.imag()));
  }

  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> cos(const complex_base<T> &x) {
    return complex_base<T>(cos(x.real())*cosh(x.imag()), -sin(x.real)*sinh(x.imag()));
  }
  
  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> cosh(const complex_base<T> &x) {
    return complex_base<T>(cosh(x.real())*cos(x.imag()), -sinh(x.real)*sin(x.imag()));
  }
  
  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> tan(const complex_base<T> &x) {
    T sr(sin(x.real()));
    T cr(cos(x.real()));
    T shi(sinh(x.imag()));
    T chi(cosh(x.imag()));
    T norm(cr*cr + shi*shi);
    return complex_base<T>(sr*cr/norm, shi*chi/norm);
  }
  
  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> tanh(const complex_base<T> &x) {
    T shr(sinh(x.real()));
    T chr(cosh(x.real()));
    T si(sin(x.imag()));
    T ci(cos(x.imag()));
    T norm(shr*shr + ci*ci);
    return complex_base<T>(shr*chr/norm, si*ci/norm);
  }
  
  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> exp(const complex_base<T> &x) {
    T er(exp(x.real()));
    T ci(cos(x.imag()));
    T si(sin(x.imag()));
    return complex_base<T>(er*ci, er*si);
  }
  
  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> log(const complex_base<T> &x) {
    return complex_base<T>(log(x.real()*x.real())/T(2), atan2(x.imag(), x.real()));
  }
  
  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> log10(const complex_base<T> &x) {
    static const T c_log10(log(T(10)));
    return complex_base<T>(log(x.real()*x.real())/T(2)/c_log10, atan2(x.imag(), x.real())/c_log10);
  }
  
  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> pow(const complex_base<T> &x, const complex_base<T> &y) {
    return exp(y*log(x));
  }

  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> sqrt(const complex_base<T> &z) {
    T x(z.real()), y(z.imag());
    if (x==T(0)) {
      T t(sqrt(abs(y)/T(2)));
      return complex_base<T>(t, y<T(0) ? -t : t);
    } else { 
      T t(sqrt(2*(abs(z)+abs(x))));
      T u(t/2);
      return x>T(0) ? complex_base<T>(u, y/t) : complex_base<T>(abs(y)/t, y<T(0) ? -u : u);
    }
  }

  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> asin(const complex_base<T> &z) {
    T r2(z.real()*z.real());
    T i2(z.imag()*z.imag());
    T t1(z.real()+1);
    T t2(z.real()-1);
    t1*=t1;
    t1+=i2;
    t1=sqrt(t1);
    t2*=t2;
    t2+=i2;
    t2=sqrt(t2);
    return complex_base<T>(asin((t1-t2)/2), 
			   sgn(complex_base<T>(z.imag(), -z.real()))*log((t1+t2+sqrt(2*(r2+i2-1+t1*t2)))/2));
  }
  
  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> asinh(const complex_base<T> &z) {
    T r2(z.real()*z.real());
    T i2(z.imag()*z.imag());
    T t1(z.imag()+1);
    T t2(z.imag()-1);
    t1*=t1;
    t1+=r2;
    t1=sqrt(t1);
    t2*=t2;
    t2+=r2;
    t2=sqrt(t2);
    return complex_base<T>(sgn(z)*log((t1+t2+sqrt(2*(r2+i2-1+t1*t2)))/2),
			   asin((t1-t2)/2));
  }
  
  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> acos(const complex_base<T> &z) {
    T r2(z.real()*z.real());
    T i2(z.imag()*z.imag());
    T t1(z.real()+1);
    T t2(z.real()-1);
    t1*=t1;
    t1+=i2;
    t1=sqrt(t1);
    t2*=t2;
    t2+=i2;
    t2=sqrt(t2);
    return complex_base<T>(acos((t1-t2)/2), 
			   sgn(complex_base<T>(-z.imag(), z.real()))*log((t1+t2+sqrt(2*(r2+i2-1+t1*t2)))/2));
  }
  
  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> acosh(const complex_base<T> &z) {
    T r2(z.real()*z.real());
    T i2(z.imag()*z.imag());
    T t1(z.real()+1);
    T t2(z.real()-1);
    t1*=t1;
    t1+=i2;
    t1=sqrt(t1);
    t2*=t2;
    t2+=i2;
    t2=sqrt(t2);
    return sgn(complex_base<T>(z.imag(), -z.real()+1))*
      complex_base<T>(sgn(complex_base<T>(z.imag(), -z.real()))*log((t1+t2+sqrt(2*(r2+i2-1+t1*t2)))/2),
		      acos((t1-t2)/2));
  }

  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> atan(const complex_base<T> &z) {
    T r2(z.real()*z.real());
    T i2(z.imag()*z.imag());
    T t1(z.imag()+1);
    T t2(z.imag()-1);
    t1*=t1;
    t1+=r2;
    t2*=t2;
    t2+=r2;
    return complex_base<T>((atan2(z.real(), 1-z.imag())-atan2(-z.real(), 1+z.imag()))/2,
			   log(t1/t2)/4);
  }

  template<typename T> 
  CUMATH_HOST_DEVICE
  complex_base<T> atanh(const complex_base<T> &z) {
    T r2(z.real()*z.real());
    T i2(z.imag()*z.imag());
    T t1(z.real()+1);
    T t2(z.real()-1);
    t1*=t1;
    t1+=i2;
    t2*=t2;
    t2+=i2;
    return complex_base<T>(log(t1/t2)/4,
			   (atan2(z.imag(), 1+z.real())-atan2(-z.imag(), 1-z.real()))/2);
  }
  
  template<typename T, typename CharT, class Traits>
  ::std::basic_istream<CharT, Traits>&
  operator>>(::std::basic_istream<CharT, Traits>& in, complex_base<T>& z) {
    CharT ch;
    in >> ch;
    if (ch=='(') {
      T z_re, z_im;
      in >> z_re >> ch;
      if (ch==',') {
	in >> z_im >> ch;
	if (ch==')') 
	  z=complex_base<T>(z_re, z_im);
	else
	  in.setstate(::std::ios_base::failbit);
      } else if (ch==')') 
	z=z_re;
      else
	in.setstate(::std::ios_base::failbit);
    } else {
      T z_re;
      in.putback(ch);
      in >> z_re;
      z=z_re;
    }
    return in;
  }

  template<typename T, typename CharT, class Traits>
  ::std::basic_ostream<CharT, Traits>&
  operator<<(::std::basic_ostream<CharT, Traits>& out, const complex_base<T>& z) {
    ::std::basic_ostringstream<CharT, Traits> s;
    s.flags(out.flags());
    s.imbue(out.getloc());
    s.precision(out.precision());
    s << '(' << z.real() << ',' << z.imag() << ')';
    return out << s.str();
  }
    
  template<typename T>
  class complex : public complex_base<T> {
    typedef complex_base<T> base;
  public:
    CUMATH_HOST_DEVICE
    complex() : base() {
    }
    CUMATH_HOST_DEVICE
    complex(const T &r) : base(r) {
    }
    CUMATH_HOST_DEVICE
    complex(const T &r, const T &i) : base(r, i) {
    }
    CUMATH_HOST_DEVICE
    complex(const complex_base<T> &other) : base(other.real(), other.imag()) {
    }
    
  };

  template<>
  class complex<double> : public complex_base<double> {
    typedef complex_base<double> base;
  public:
    CUMATH_HOST_DEVICE
    complex() : base() {
    }
    CUMATH_HOST_DEVICE
    complex(const double &r) : base(r) {
    }
    CUMATH_HOST_DEVICE
    complex(const double &r, const double &i) : base(r, i) {
    }
    CUMATH_HOST_DEVICE
    complex(const complex_base<double> &other) : base(other.real(), other.imag()) {
    }
    CUMATH_HOST_DEVICE
    complex(const complex_base<float> &other) : base(static_cast<double>(other.real()), static_cast<double>(other.imag())) {
    }
  };
  
}

#endif
