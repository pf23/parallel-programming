// Penn Parallel Primitives Library
// Author: Prof. Milo Martin
// University of Pennsylvania
// Spring 2010

#include "barrier.h"

namespace ppp {
  namespace internal {

    __thread int t_local_sense = 0;
    bool Barrier::s_init = false;

    Barrier::Barrier() 
    {
      // only construct a barrier once and always before init() is called
      assert(get_thread_count() == 0);
      assert(s_init == false);
      s_init = true;
      
      m_counter.set(0); 
      m_sense.set(0);
    }
    
    void Barrier::arrive() 
    {
      t_local_sense = !t_local_sense;
      int val = m_counter.fetch_and_inc();
      if (val == get_thread_count()-1) {
        m_counter.set(0);
        m_sense.set(t_local_sense);
      } else {
        while (true) {
          if (m_sense.get() == t_local_sense) {
            break;
          }
          pause();
        }
      }
    }
    
    Barrier b;  // global variable
  }
}
