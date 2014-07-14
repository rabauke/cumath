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

#include <cuda_runtime.h>

// see http://www.cplusplus.com/reference/cmath/

namespace cumath {

  __inline__ __host__ __device__
  float sin(float x) { return ::sinf(x); }

  __inline__ __host__ __device__
  double sin(double x) { return ::sin(x); }
  
  __inline__ __host__ __device__
  float cos(float x) { return ::cosf(x); }

  __inline__ __host__ __device__
  double cos(double x) { return ::cos(x); }
  
  __inline__ __host__ __device__
  float tan(float x) { return ::tanf(x); }

  __inline__ __host__ __device__
  double tan(double x) { return ::tan(x); }

  __inline__ __host__ __device__
  float asin(float x) { return ::asinf(x); }

  __inline__ __host__ __device__
  double asin(double x) { return ::asin(x); }
  
  __inline__ __host__ __device__
  float acos(float x) { return ::acosf(x); }

  __inline__ __host__ __device__
  double acos(double x) { return ::acos(x); }
  
  __inline__ __host__ __device__
  float atan(float x) { return ::atanf(x); }

  __inline__ __host__ __device__
  double atan(double x) { return ::atan(x); }
  
  __inline__ __host__ __device__
  float atan2(float y, float x) { return ::atan2f(y, x); }

  __inline__ __host__ __device__
  double atan2(double y, double x) { return ::atan2(y, x); }

  __inline__ __host__ __device__
  float sinh(float x) { return ::sinhf(x); }

  __inline__ __host__ __device__
  double sinh(double x) { return ::sinh(x); }
  
  __inline__ __host__ __device__
  float cosh(float x) { return ::coshf(x); }

  __inline__ __host__ __device__
  double cosh(double x) { return ::cosh(x); }
  
  __inline__ __host__ __device__
  float tanh(float x) { return ::tanhf(x); }

  __inline__ __host__ __device__
  double tanh(double x) { return ::tanh(x); }
  
  __inline__ __host__ __device__
  float asinh(float x) { return ::asinhf(x); }

  __inline__ __host__ __device__
  double asinh(double x) { return ::asinh(x); }
  
  __inline__ __host__ __device__
  float acosh(float x) { return ::acoshf(x); }

  __inline__ __host__ __device__
  double acosh(double x) { return ::acosh(x); }
  
  __inline__ __host__ __device__
  float atanh(float x) { return ::atanhf(x); }

  __inline__ __host__ __device__
  double atanh(double x) { return ::atanh(x); }

  __inline__ __host__ __device__
  float exp(float x) { return ::expf(x); }
  
  __inline__ __host__ __device__
  double exp(double x) { return ::exp(x); }
  
  __inline__ __host__ __device__
  float frexp(float x, int *exp) { return ::frexpf(x, exp); }

  __inline__ __host__ __device__
  double frexp(double x, int *exp) { return ::frexp(x, exp); }

  __inline__ __host__ __device__
  float ldexp(float x, int exp) { return ::ldexpf(x, exp); }

  __inline__ __host__ __device__
  double ldexp(double x, int exp) { return ::ldexp(x, exp); }
  
  __inline__ __host__ __device__
  float log(float x) { return ::logf(x); }

  __inline__ __host__ __device__
  double log(double x) { return ::log(x); }
  
  __inline__ __host__ __device__
  float log10(float x) { return ::log10f(x); }

  __inline__ __host__ __device__
  double log10(double x) { return ::log10(x); }

  __inline__ __host__ __device__
  float modf(float x, float *intpart) { return ::modff(x, intpart); }

  __inline__ __host__ __device__
  double modf(double x, double *intpart) { return ::modf(x, intpart); }

  __inline__ __host__ __device__
  float exp2(float x) { return ::exp2f(x); }
  
  __inline__ __host__ __device__
  double exp2(double x) { return ::exp2(x); }
  
  __inline__ __host__ __device__
  float expm1(float x) { return ::expm1f(x); }
  
  __inline__ __host__ __device__
  double expm1(double x) { return ::expm1(x); }
  
  __inline__ __host__ __device__
  float ilogb(float x) { return ::ilogbf(x); }
  
  __inline__ __host__ __device__
  double ilogb(double x) { return ::ilogb(x); }
  
  __inline__ __host__ __device__
  float log1p(float x) { return ::log1pf(x); }
  
  __inline__ __host__ __device__
  double log1p(double x) { return ::log1p(x); }
  
  __inline__ __host__ __device__
  float log2(float x) { return ::log2f(x); }
  
  __inline__ __host__ __device__
  double log2(double x) { return ::log2(x); }
  
  __inline__ __host__ __device__
  float logb(float x) { return ::logbf(x); }
  
  __inline__ __host__ __device__
  double logb(double x) { return ::logb(x); }
  
  __inline__ __host__ __device__
  float scalbn(float x, int exp) { return ::scalbnf(x, exp); }

  __inline__ __host__ __device__
  double scalbn(double x, int exp) { return ::scalbn(x, exp); }
  
  __inline__ __host__ __device__
  float scalbln(float x, long int exp) { return ::scalblnf(x, exp); }

  __inline__ __host__ __device__
  double scalbln(double x, long int exp) { return ::scalbln(x, exp); }

  __inline__ __host__ __device__
  float pow(float x, float y) { return ::powf(x, y); }

  __inline__ __host__ __device__
  double pow(double x, double y) { return ::pow(x, y); }
  
  __inline__ __host__ __device__
  float sqrt(float x) { return ::sqrtf(x); }

  __inline__ __host__ __device__
  double sqrt(double x) { return ::sqrt(x); }
  
  __inline__ __host__ __device__
  float cbrt(float x) { return ::cbrtf(x); }

  __inline__ __host__ __device__
  double cbrt(double x) { return ::cbrt(x); }
  
  __inline__ __host__ __device__
  float hypot(float x, float y) { return ::hypotf(x, y); }

  __inline__ __host__ __device__
  double hypot(double x, double y) { return ::hypot(x, y); }

  __inline__ __host__ __device__
  float erf(float x) { return ::erff(x); }

  __inline__ __host__ __device__
  double erf(double x) { return ::erf(x); }
  
  __inline__ __host__ __device__
  float erfc(float x) { return ::erfcf(x); }

  __inline__ __host__ __device__
  double erfc(double x) { return ::erfc(x); }

  __inline__ __host__ __device__
  float tgamma(float x) { return ::tgammaf(x); }

  __inline__ __host__ __device__
  double tgamma(double x) { return ::tgamma(x); }
  
  __inline__ __host__ __device__
  float lgamma(float x) { return ::lgammaf(x); }

  __inline__ __host__ __device__
  double lgamma(double x) { return ::lgamma(x); }
  
  __inline__ __host__ __device__
  float ceil(float x) { return ::ceilf(x); }

  __inline__ __host__ __device__
  double ceil(double x) { return ::ceil(x); }
  
  __inline__ __host__ __device__
  float floor(float x) { return ::floorf(x); }

  __inline__ __host__ __device__
  double floor(double x) { return ::floor(x); }

  __inline__ __host__ __device__
  float fmod(float x, float y) { return ::fmodf(x, y); }

  __inline__ __host__ __device__
  double fmod(double x, double y) { return ::fmod(x, y); }
  
  __inline__ __host__ __device__
  float trunc(float x) { return ::truncf(x); }

  __inline__ __host__ __device__
  double trunc(double x) { return ::trunc(x); }

  __inline__ __host__ __device__
  float round(float x) { return ::roundf(x); }

  __inline__ __host__ __device__
  double round(double x) { return ::round(x); }

  __inline__ __host__ __device__
  long int lround(float x) { return ::lroundf(x); }

  __inline__ __host__ __device__
  long int lround(double x) { return ::lround(x); }
  
  __inline__ __host__ __device__
  long long int llround(float x) { return ::llroundf(x); }

  __inline__ __host__ __device__
  long long int llround(double x) { return ::llround(x); }
  
  __inline__ __host__ __device__
  float nearbyint(float x) { return ::nearbyintf(x); }

  __inline__ __host__ __device__
  double nearbyint(double x) { return ::nearbyint(x); }

  __inline__ __host__ __device__
  float remainder(float x, float y) { return ::remainderf(x, y); }

  __inline__ __host__ __device__
  double remainder(double x, double y) { return ::remainder(x, y); }

  __inline__ __host__ __device__
  float remquo(float x, float y, int *quot) { return ::remquof(x, y, quot); }

  __inline__ __host__ __device__
  double remquo(double x, double y, int *quot) { return ::remquo(x, y, quot); }

  __inline__ __host__ __device__
  float copysign(float x, float y) { return ::copysignf(x, y); }

  __inline__ __host__ __device__
  double copysign(double x, double y) { return ::copysign(x, y); }

  __inline__ __host__ __device__
  double nan(const char *tagp) { return ::nan(tagp); }

  __inline__ __host__ __device__
  float nextafter(float x, float y) { return ::nextafterf(x, y); }

  __inline__ __host__ __device__
  double nextafter(double x, double y) { return ::nextafter(x, y); }

  __inline__ __host__ __device__
  float fdim(float x, float y) { return ::fdimf(x, y); }

  __inline__ __host__ __device__
  double fdim(double x, double y) { return ::fdim(x, y); }

  __inline__ __host__ __device__
  float fmax(float x, float y) { return ::fmaxf(x, y); }

  __inline__ __host__ __device__
  double fmax(double x, double y) { return ::fmax(x, y); }

  __inline__ __host__ __device__
  float fmin(float x, float y) { return ::fminf(x, y); }

  __inline__ __host__ __device__
  double fmin(double x, double y) { return ::fmin(x, y); }

  __inline__ __host__ __device__
  float fabs(float x) { return ::fabsf(x); }

  __inline__ __host__ __device__
  double fabs(double x) { return ::fabs(x); }

  __inline__ __host__ __device__
  float abs(float x) { return ::fabsf(x); }

  __inline__ __host__ __device__
  double abs(double x) { return ::fabs(x); }

  __inline__ __host__ __device__
  float fma(float x, float y, float z) { return ::fmaf(x, y, z); }

  __inline__ __host__ __device__
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

#else /* __APPLE__ */

#define __signbit(x) \
        (sizeof(x) == sizeof(float) ? __signbitf(x) : sizeof(x) == sizeof(double) ? __signbit(x) : __signbitl(x))
#define __isfinite(x) \
        (sizeof(x) == sizeof(float) ? __finitef(x) : sizeof(x) == sizeof(double) ? __finite(x) : __finitel(x))
#define __isnan(x) \
        (sizeof(x) == sizeof(float) ? __isnanf(x) : sizeof(x) == sizeof(double) ? __isnan(x) : __isnanl(x))
#define __isinf(x) \
        (sizeof(x) == sizeof(float) ? __isinff(x) : sizeof(x) == sizeof(double) ? __isinf(x) : __isinfl(x))

#endif /* __APPLE__ */

#endif

  __inline__ __host__ __device__
  bool isfinite(float x) { return __isfinite(x); }

  __inline__ __host__ __device__
  bool isfinite(double x) { return __isfinite(x); }

  __inline__ __host__ __device__
  bool isinf(float x) { return __isinf(x); }

  __inline__ __host__ __device__
  bool isinf(double x) { return __isinf(x); }

  __inline__ __host__ __device__
  bool isnan(float x) { return __isnan(x); }

  __inline__ __host__ __device__
  bool isnan(double x) { return __isnan(x); }

  __inline__ __host__ __device__
  bool signbit(float x) { return __signbit(x); }

  __inline__ __host__ __device__
  bool signbit(double x) { return __signbit(x); }

}

#endif
