#ifndef __UTILITY_CPP
#define __UTILITY_CPP

#include <utility.hpp>

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

template <typename K, typename V>
void print_map(std::map<K, V> const &m)
{
  for (auto it = m.cbegin(); it != m.cend(); ++it)
    Rcpp::Rcout << "{ " << (*it).first << " }\n";
}

#endif