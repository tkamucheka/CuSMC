#ifndef __UTILITY_HPP
#define __UTILITY_HPP

#include <Rcpp.h>
#include <iostream>
#include <sys/time.h>

// Timer struct
typedef struct
{
  struct timeval startTime;
  struct timeval endTime;
} Timer;

// Methods
void print_m(float *A, unsigned sz_m, unsigned sz_n);

#ifdef __cplusplus
extern "C"
{
#endif
  void startTime(Timer *timer);
  void stopTime(Timer *timer);
  float elapsedTime(Timer timer);
#ifdef __cplusplus
}
#endif

#endif

#ifndef __UTILITY_CPP
#define __UTILITY_CPP

void print_m(float *A, unsigned sz_m, unsigned sz_n)
{
  for (unsigned i = 0; i < sz_m * sz_n; ++i)
  {
    Rcpp::Rcout << A[i] << "\t";
    if ((i + 1) % sz_m == 0)
      Rcpp::Rcout << std::endl;
  }

  Rcpp::Rcout << std::endl;
}

void startTime(Timer *timer)
{
  gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer *timer)
{
  gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer)
{
  return ((float)((timer.endTime.tv_sec - timer.startTime.tv_sec) +
                  (timer.endTime.tv_usec - timer.startTime.tv_usec) / 1.0e6));
}
#endif