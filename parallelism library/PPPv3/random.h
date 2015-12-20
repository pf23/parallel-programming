// Penn Parallel Primitives Library
// Author: Prof. Milo Martin
// University of Pennsylvania
// Spring 2010

#ifndef PPP_RANDOM_H
#define PPP_RANDOM_H

#include "ppp.h"

namespace ppp {

  // Note: not a good enough random generator for security or scientific
  // simulation, but simple, fast, and good enough for our purposes.
  uint32_t threadsafe_random();
  void reseed_random_per_thread();
  int64_t bigrand(int64_t max, int dist=1);
}

#endif
