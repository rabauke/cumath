#if !(defined TIMER_HPP)

#define TIMER_HPP

#if defined __unix__
  #include <unistd.h>
  #include <sys/time.h> 
  #include <sys/times.h>
#else
  #include <ctime>
#endif

namespace timer {

  class timer {
  private:
    const double _resolution;
    double _t, _t_start;
    bool isrunning;

    double get_time() const {
#if defined __unix__
      struct timeval tv;
      gettimeofday(&tv, 0);
      return static_cast<double>(tv.tv_sec)+static_cast<double>(tv.tv_usec)*1e-6;
#else
      return static_cast<double>(std::clock())*_resolution;
#endif
    }
  public:
    typedef enum { running, waiting } status_type;
    void reset() {
      _t=0.0;
    }
    void start() {
      _t_start=get_time();
      isrunning=true;
    }
    void stop() {
      if (isrunning) {
        _t+=get_time()-_t_start;
        isrunning=false;
      }
    }
    double time() const {
      return _t+( isrunning ? get_time()-_t_start : 0.0 );
    }
    double resolution() const {
      return _resolution;
    };
    timer(status_type s=running) :
#if defined __unix__
      _resolution(1e-6),
#else
      _resolution(1.0/CLOCKS_PER_SEC),
#endif
      _t(0), _t_start(get_time()),
      isrunning(true) {
      if (s==waiting) {
	stop();
	reset();
      }
    }
  };
  
}

#endif
