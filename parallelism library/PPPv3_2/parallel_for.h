// Penn Parallel Primitives Library
// Author: Prof. Milo Martin
// University of Pennsylvania
// Spring 2010

#ifndef PPP_FOR_H
#define PPP_FOR_H

#include "ppp.h"
#include "Task.h"
#include "TaskGroup.h"

namespace ppp {
  
  namespace internal {
    
    template <typename T>
    class SorterForTask: public ppp::Task {
    public:
      SorterForTask(T* functor, int64_t left, int64_t right, int64_t grainsize) {
        m_functor = functor;
        m_left = left;
        m_right = right;
        m_grainsize = grainsize;
      }
      
      void execute() {
        PPP_DEBUG_MSG("Execute: [" + to_string(m_left) + ", " + to_string(m_right) + "]");
        assert(m_left < m_right);

        if (m_right-m_left <= 1) {
          return;
        }
        
        if (m_right-m_left < m_grainsize) {
          PPP_DEBUG_MSG("std::sort_for: [" + to_string(m_left) + ", " + to_string(m_right) + "]");
          PPP_DEBUG_MSG("std::sort_for: [" + to_string(&m_functor[m_left]) + ", " + to_string(&m_functor[m_right]) + "]");
          m_functor->calculate(m_left, m_right);
          return;
        }
        
        int64_t pivot = (m_left + m_right)/2;
        PPP_DEBUG_MSG("Split: [" + to_string(m_left) + ", " + to_string(pivot) + "] [" +
                      to_string(pivot) + ", " + to_string(m_right) + "]");
        ppp::TaskGroup tg;
        SorterForTask t1(m_functor, m_left, pivot, m_grainsize);
        SorterForTask t2(m_functor, pivot, m_right, m_grainsize);
        tg.spawn(t1);
        tg.spawn(t2);
        tg.wait();
      }
    private:
      T* m_functor;
      int64_t m_left;
      int64_t m_right;
      int64_t m_grainsize;
    };
  }	

  template <typename T>
  extern inline
  void parallel_for(int64_t start, int64_t end, T* functor, int64_t grainsize=0)
  {
    // ASSIGNMENT: make this parallel via recursive divide and conquer
	if (grainsize == 0) {
      grainsize = (end-start+1) / (get_thread_count()*4);
    }
   
    PPP_DEBUG_MSG("parallel_for grainsize: " + to_string(grainsize));
    internal::SorterForTask<T> t(functor, start, end, grainsize);
    // the divider and conquer is implemented in execute()
    t.execute();
    PPP_DEBUG_MSG("parallel_for done");

  }
}

#endif
