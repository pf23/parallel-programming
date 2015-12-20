// Penn Parallel Primitives Library
// Author: Prof. Milo Martin
// University of Pennsylvania
// Spring 2010

#ifndef PPP_BARRIER_H
#define PPP_BARRIER_H

#include "atomic.h"

// Global counter barrier.  Based upon the pseudo code from:
// http://www.cs.rochester.edu/research/synchronization/pseudocode/ss.html#centralized

//   shared count : integer := P
//   shared sense : Boolean := true
//   processor private local_sense : Boolean := true
  
//   procedure central_barrier
//       local_sense := not local_sense   // each processor toggles its own sense
//       if fetch_and_decrement (&count) = 1
//           count := P
//           sense := local_sense         // last processor toggles global sense
//       else
//           repeat until sense = local_sense

namespace ppp {

  extern void barrier();

  namespace internal {
    class Barrier {
    public:
      Barrier();
      void arrive();
    private:
      static bool s_init;
      atomic<int> m_counter;
      atomic<int> m_sense;
      
      // Private copy constructor and assignment operator
      Barrier(const Barrier& obj);
      Barrier& operator=(const Barrier& obj);
    };

    extern Barrier b;  // global variable
  }
  
  extern inline void barrier() {
    internal::b.arrive();
  }
}  

#endif


