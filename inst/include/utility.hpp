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