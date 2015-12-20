// Penn Parallel Primitives Library
// Author: Prof. Milo Martin
// University of Pennsylvania
// Spring 2010

#ifndef PPP_REDUCE_H
#define PPP_REDUCE_H

#include "ppp.h"
#include "Task.h"
#include "TaskGroup.h"

namespace ppp {
  
  /*
  template <typename T>
  extern inline
  T parallel_reduce(T* array, int64_t start, int64_t end, int64_t grainsize=0)
  {
    // ASSIGNMENT: make this parallel via recursive divide and conquer

    // Sequential code
    T sum;
    sum = T(0);
    for (int i=start; i<end; i++) {
      sum = sum + array[i];
    }
    return sum;
  }
  */
   namespace internal {

    class SumTask: public ppp::Task {
    public:
        SumTask(int64_t* array, int64_t left, int64_t right, int64_t grainsize)
        {
            m_array = array;
            m_left = left;
            m_right = right;
            m_grainsize = grainsize;
            setSum(0);
        }
      
        void execute()
        {
            PPP_DEBUG_MSG("Execute: [" + to_string(m_left) + ", " + to_string(m_right) + "]");
        
            assert(m_left < m_right);

            if (m_right-m_left <= 1)
                return;
        
            if (m_right-m_left < m_grainsize) {
                PPP_DEBUG_MSG("std::sum: [" + to_string(m_left) + ", " + to_string(m_right) + "]");
                PPP_DEBUG_MSG("std::sum: [" + to_string(&m_array[m_left]) + ", " + to_string(&m_array[m_right]) + "]");
                // sum up the numbers in subtasks
                int64_t sum=0;
                for (int64_t i=m_left;i<m_right;i++)
                    sum += m_array[i];
                setSum(sum);
                return;
            }
            
            int64_t pivot = (m_left+m_right)/2;
            
            PPP_DEBUG_MSG("Split: [" + to_string(m_left) + ", " + to_string(pivot) + "] [" +
                      to_string(pivot) + ", " + to_string(m_right) + "]");
            ppp::TaskGroup tg;
            SumTask t1(m_array, m_left, pivot, m_grainsize);
            SumTask t2(m_array, pivot, m_right, m_grainsize);
            tg.spawn(t1);
            tg.spawn(t2);
            tg.wait();
            setSum(t1.return_sum() + t2.return_sum());
        }
        
        void setSum(int64_t new_sum)
        {
            sum = new_sum;
        }
        int64_t return_sum()
        {
            return sum;
        }
    
    private:
        int64_t* m_array;
        int64_t m_left;
        int64_t m_right;
        int64_t m_grainsize;
        int64_t sum;
    };
  }
  
  int64_t parallel_reduce(int64_t* array, int64_t left, int64_t right, int64_t grainsize=0)
  {
    if (grainsize == 0) {
      grainsize = (right-left+1) / (get_thread_count()*4);
    }
    PPP_DEBUG_MSG("parallel_reduce grainsize: " + to_string(grainsize));
            
    internal::SumTask t(array, left, right, grainsize);
    t.execute();
    PPP_DEBUG_MSG("parallel_reduce done");
    return t.return_sum();
  }
}
#endif
