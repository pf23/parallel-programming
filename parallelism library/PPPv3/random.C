// Penn Parallel Primitives Library
// Author: Prof. Milo Martin
// University of Pennsylvania
// Spring 2010

#include "random.h"

namespace ppp {

  // "Multiply-With-Carry" generator of G. Marsaglia.  Code from
  // "Random_number_generation" Wikipedia's entry.
  // Note: not a good enough random generator for security or scientific
  // simulation, but simple, fast, and good enough for our purposes.
  namespace internal {
    __thread uint32_t m_w = 3421085238;    /* must not be zero */
    __thread uint32_t m_z = 1398836382;    /* must not be zero */
  }
  using namespace internal;
  
  uint32_t threadsafe_random()
  {
    m_z = 36969 * (m_z & 65535) + (m_z >> 16);
    m_w = 18000 * (m_w & 65535) + (m_w >> 16);
    return (m_z << 16) + m_w;  /* 32-bit result */
  }

  void reseed_random_per_thread() 
  {
    m_w = 3421085238;
    m_z = 1398836382;
    
    for (int i=0; i<s_thread_id; i++) {
      threadsafe_random();
    }
    uint32_t w = threadsafe_random();
    uint32_t z = threadsafe_random();
    m_w = w;
    m_z = z;
  }

  int64_t bigrand(int64_t max, int dist)
  {
    if (max < dist) {
      dist = 1;
    }
    int64_t range = max/dist;
    int64_t rand = 0;
    for (int i=0; i<dist; i++) {
      int64_t r = (ppp::threadsafe_random() << 31) ^ ppp::threadsafe_random();
      rand += (r % range);
    }
    assert(rand < max);
    return rand;
  }
}
