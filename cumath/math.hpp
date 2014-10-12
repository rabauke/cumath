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

#if !(defined CUMATH_MATH_HPP)

#define CUMATH_MATH_HPP

#include <cumath/cuda.hpp>

#if !(defined __NVCC__)
#include <cmath>
#endif

// see http://www.cplusplus.com/reference/cmath/

namespace cumath {

  CUMATH_HOST_DEVICE
  float sin(float x) { return ::sinf(x); }

  CUMATH_HOST_DEVICE
  double sin(double x) { return ::sin(x); }
  
  CUMATH_HOST_DEVICE
  float cos(float x) { return ::cosf(x); }

  CUMATH_HOST_DEVICE
  double cos(double x) { return ::cos(x); }
  
  CUMATH_HOST_DEVICE
  float tan(float x) { return ::tanf(x); }

  CUMATH_HOST_DEVICE
  double tan(double x) { return ::tan(x); }

  CUMATH_HOST_DEVICE
  float asin(float x) { return ::asinf(x); }

  CUMATH_HOST_DEVICE
  double asin(double x) { return ::asin(x); }
  
  CUMATH_HOST_DEVICE
  float acos(float x) { return ::acosf(x); }

  CUMATH_HOST_DEVICE
  double acos(double x) { return ::acos(x); }
  
  CUMATH_HOST_DEVICE
  float atan(float x) { return ::atanf(x); }

  CUMATH_HOST_DEVICE
  double atan(double x) { return ::atan(x); }
  
  CUMATH_HOST_DEVICE
  float atan2(float y, float x) { return ::atan2f(y, x); }

  CUMATH_HOST_DEVICE
  double atan2(double y, double x) { return ::atan2(y, x); }

  CUMATH_HOST_DEVICE
  float sinh(float x) { return ::sinhf(x); }

  CUMATH_HOST_DEVICE
  double sinh(double x) { return ::sinh(x); }
  
  CUMATH_HOST_DEVICE
  float cosh(float x) { return ::coshf(x); }

  CUMATH_HOST_DEVICE
  double cosh(double x) { return ::cosh(x); }
  
  CUMATH_HOST_DEVICE
  float tanh(float x) { return ::tanhf(x); }

  CUMATH_HOST_DEVICE
  double tanh(double x) { return ::tanh(x); }
  
  CUMATH_HOST_DEVICE
  float asinh(float x) { return ::asinhf(x); }

  CUMATH_HOST_DEVICE
  double asinh(double x) { return ::asinh(x); }
  
  CUMATH_HOST_DEVICE
  float acosh(float x) { return ::acoshf(x); }

  CUMATH_HOST_DEVICE
  double acosh(double x) { return ::acosh(x); }
  
  CUMATH_HOST_DEVICE
  float atanh(float x) { return ::atanhf(x); }

  CUMATH_HOST_DEVICE
  double atanh(double x) { return ::atanh(x); }

  CUMATH_HOST_DEVICE
  float exp(float x) { return ::expf(x); }
  
  CUMATH_HOST_DEVICE
  double exp(double x) { return ::exp(x); }
  
  CUMATH_HOST_DEVICE
  float frexp(float x, int *exp) { return ::frexpf(x, exp); }

  CUMATH_HOST_DEVICE
  double frexp(double x, int *exp) { return ::frexp(x, exp); }

  CUMATH_HOST_DEVICE
  float ldexp(float x, int exp) { return ::ldexpf(x, exp); }

  CUMATH_HOST_DEVICE
  double ldexp(double x, int exp) { return ::ldexp(x, exp); }
  
  CUMATH_HOST_DEVICE
  float log(float x) { return ::logf(x); }

  CUMATH_HOST_DEVICE
  double log(double x) { return ::log(x); }
  
  CUMATH_HOST_DEVICE
  float log10(float x) { return ::log10f(x); }

  CUMATH_HOST_DEVICE
  double log10(double x) { return ::log10(x); }

  CUMATH_HOST_DEVICE
  float modf(float x, float *intpart) { return ::modff(x, intpart); }

  CUMATH_HOST_DEVICE
  double modf(double x, double *intpart) { return ::modf(x, intpart); }

  CUMATH_HOST_DEVICE
  float exp2(float x) { return ::exp2f(x); }
  
  CUMATH_HOST_DEVICE
  double exp2(double x) { return ::exp2(x); }
  
  CUMATH_HOST_DEVICE
  float expm1(float x) { return ::expm1f(x); }
  
  CUMATH_HOST_DEVICE
  double expm1(double x) { return ::expm1(x); }
  
  CUMATH_HOST_DEVICE
  float ilogb(float x) { return ::ilogbf(x); }
  
  CUMATH_HOST_DEVICE
  double ilogb(double x) { return ::ilogb(x); }
  
  CUMATH_HOST_DEVICE
  float log1p(float x) { return ::log1pf(x); }
  
  CUMATH_HOST_DEVICE
  double log1p(double x) { return ::log1p(x); }
  
  CUMATH_HOST_DEVICE
  float log2(float x) { return ::log2f(x); }
  
  CUMATH_HOST_DEVICE
  double log2(double x) { return ::log2(x); }
  
  CUMATH_HOST_DEVICE
  float logb(float x) { return ::logbf(x); }
  
  CUMATH_HOST_DEVICE
  double logb(double x) { return ::logb(x); }
  
  CUMATH_HOST_DEVICE
  float scalbn(float x, int exp) { return ::scalbnf(x, exp); }

  CUMATH_HOST_DEVICE
  double scalbn(double x, int exp) { return ::scalbn(x, exp); }
  
  CUMATH_HOST_DEVICE
  float scalbln(float x, long int exp) { return ::scalblnf(x, exp); }

  CUMATH_HOST_DEVICE
  double scalbln(double x, long int exp) { return ::scalbln(x, exp); }

  CUMATH_HOST_DEVICE
  float pow(float x, float y) { return ::powf(x, y); }

  CUMATH_HOST_DEVICE
  double pow(double x, double y) { return ::pow(x, y); }
  
  CUMATH_HOST_DEVICE
  float sqrt(float x) { return ::sqrtf(x); }

  CUMATH_HOST_DEVICE
  double sqrt(double x) { return ::sqrt(x); }
  
  CUMATH_HOST_DEVICE
  float cbrt(float x) { return ::cbrtf(x); }

  CUMATH_HOST_DEVICE
  double cbrt(double x) { return ::cbrt(x); }
  
  CUMATH_HOST_DEVICE
  float hypot(float x, float y) { return ::hypotf(x, y); }

  CUMATH_HOST_DEVICE
  double hypot(double x, double y) { return ::hypot(x, y); }

  CUMATH_HOST_DEVICE
  float erf(float x) { return ::erff(x); }

  CUMATH_HOST_DEVICE
  double erf(double x) { return ::erf(x); }
  
  CUMATH_HOST_DEVICE
  float erfc(float x) { return ::erfcf(x); }

  CUMATH_HOST_DEVICE
  double erfc(double x) { return ::erfc(x); }

  CUMATH_HOST_DEVICE
  float tgamma(float x) { return ::tgammaf(x); }

  CUMATH_HOST_DEVICE
  double tgamma(double x) { return ::tgamma(x); }
  
  CUMATH_HOST_DEVICE
  float lgamma(float x) { return ::lgammaf(x); }

  CUMATH_HOST_DEVICE
  double lgamma(double x) { return ::lgamma(x); }
  
  CUMATH_HOST_DEVICE
  float ceil(float x) { return ::ceilf(x); }

  CUMATH_HOST_DEVICE
  double ceil(double x) { return ::ceil(x); }
  
  CUMATH_HOST_DEVICE
  float floor(float x) { return ::floorf(x); }

  CUMATH_HOST_DEVICE
  double floor(double x) { return ::floor(x); }

  CUMATH_HOST_DEVICE
  float fmod(float x, float y) { return ::fmodf(x, y); }

  CUMATH_HOST_DEVICE
  double fmod(double x, double y) { return ::fmod(x, y); }
  
  CUMATH_HOST_DEVICE
  float trunc(float x) { return ::truncf(x); }

  CUMATH_HOST_DEVICE
  double trunc(double x) { return ::trunc(x); }

  CUMATH_HOST_DEVICE
  float round(float x) { return ::roundf(x); }

  CUMATH_HOST_DEVICE
  double round(double x) { return ::round(x); }

  CUMATH_HOST_DEVICE
  long int lround(float x) { return ::lroundf(x); }

  CUMATH_HOST_DEVICE
  long int lround(double x) { return ::lround(x); }
  
  CUMATH_HOST_DEVICE
  long long int llround(float x) { return ::llroundf(x); }

  CUMATH_HOST_DEVICE
  long long int llround(double x) { return ::llround(x); }
  
  CUMATH_HOST_DEVICE
  float nearbyint(float x) { return ::nearbyintf(x); }

  CUMATH_HOST_DEVICE
  double nearbyint(double x) { return ::nearbyint(x); }

  CUMATH_HOST_DEVICE
  float remainder(float x, float y) { return ::remainderf(x, y); }

  CUMATH_HOST_DEVICE
  double remainder(double x, double y) { return ::remainder(x, y); }

  CUMATH_HOST_DEVICE
  float remquo(float x, float y, int *quot) { return ::remquof(x, y, quot); }

  CUMATH_HOST_DEVICE
  double remquo(double x, double y, int *quot) { return ::remquo(x, y, quot); }

  CUMATH_HOST_DEVICE
  float copysign(float x, float y) { return ::copysignf(x, y); }

  CUMATH_HOST_DEVICE
  double copysign(double x, double y) { return ::copysign(x, y); }

  CUMATH_HOST_DEVICE
  double nan(const char *tagp) { return ::nan(tagp); }

  CUMATH_HOST_DEVICE
  float nextafter(float x, float y) { return ::nextafterf(x, y); }

  CUMATH_HOST_DEVICE
  double nextafter(double x, double y) { return ::nextafter(x, y); }

  CUMATH_HOST_DEVICE
  float fdim(float x, float y) { return ::fdimf(x, y); }

  CUMATH_HOST_DEVICE
  double fdim(double x, double y) { return ::fdim(x, y); }

  CUMATH_HOST_DEVICE
  float fmax(float x, float y) { return ::fmaxf(x, y); }

  CUMATH_HOST_DEVICE
  double fmax(double x, double y) { return ::fmax(x, y); }

  CUMATH_HOST_DEVICE
  float fmin(float x, float y) { return ::fminf(x, y); }

  CUMATH_HOST_DEVICE
  double fmin(double x, double y) { return ::fmin(x, y); }

  CUMATH_HOST_DEVICE
  float fabs(float x) { return ::fabsf(x); }

  CUMATH_HOST_DEVICE
  double fabs(double x) { return ::fabs(x); }

  CUMATH_HOST_DEVICE
  float abs(float x) { return ::fabsf(x); }

  CUMATH_HOST_DEVICE
  double abs(double x) { return ::fabs(x); }

  CUMATH_HOST_DEVICE
  float fma(float x, float y, float z) { return ::fmaf(x, y, z); }

  CUMATH_HOST_DEVICE
  double fma(double x, double y, double z) { return ::fma(x, y, z); }

#if defined(__GNUC__)

#undef signbit
#undef isfinite
#undef isnan
#undef isinf

#if defined(__APPLE__)

#define __signbit(x) \
        (sizeof(x) == sizeof(float) ? __signbitf(x) : sizeof(x) == sizeof(double) ? __signbitd(x) : __signbitl(x))
#define __isfinite(x) \
        (sizeof(x) == sizeof(float) ? __isfinitef(x) : sizeof(x) == sizeof(double) ? __isfinited(x) : __isfinite(x))
#define __isnan(x) \
        (sizeof(x) == sizeof(float) ? __isnanf(x) : sizeof(x) == sizeof(double) ? __isnand(x) : __isnan(x))
#define __isinf(x) \
        (sizeof(x) == sizeof(float) ? __isinff(x) : sizeof(x) == sizeof(double) ? __isinfd(x) : __isinf(x))

#else

#define __signbit(x) \
        (sizeof(x) == sizeof(float) ? __signbitf(x) : sizeof(x) == sizeof(double) ? __signbit(x) : __signbitl(x))
#define __isfinite(x) \
        (sizeof(x) == sizeof(float) ? __finitef(x) : sizeof(x) == sizeof(double) ? __finite(x) : __finitel(x))
#define __isnan(x) \
        (sizeof(x) == sizeof(float) ? __isnanf(x) : sizeof(x) == sizeof(double) ? __isnan(x) : __isnanl(x))
#define __isinf(x) \
        (sizeof(x) == sizeof(float) ? __isinff(x) : sizeof(x) == sizeof(double) ? __isinf(x) : __isinfl(x))

#endif

#endif

  CUMATH_HOST_DEVICE
  bool isfinite(float x) { return __isfinite(x); }

  CUMATH_HOST_DEVICE
  bool isfinite(double x) { return __isfinite(x); }

  CUMATH_HOST_DEVICE
  bool isinf(float x) { return __isinf(x); }

  CUMATH_HOST_DEVICE
  bool isinf(double x) { return __isinf(x); }

  CUMATH_HOST_DEVICE
  bool isnan(float x) { return __isnan(x); }

  CUMATH_HOST_DEVICE
  bool isnan(double x) { return __isnan(x); }

  CUMATH_HOST_DEVICE
  bool signbit(float x) { return __signbit(x); }

  CUMATH_HOST_DEVICE
  bool signbit(double x) { return __signbit(x); }

}

#endif
