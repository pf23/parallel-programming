// Penn Parallel Primitives Library
// Author: Prof. Milo Martin
// University of Pennsylvania
// Spring 2010

#ifndef PPP_SORT_H
#define PPP_SORT_H

#include <algorithm> 
#include "ppp.h"
#include "Task.h"
#include "TaskGroup.h"

namespace ppp {

  namespace internal {
    // Quicksort code based on pseudo code from
    // http://www.algolist.net/Algorithms/Sorting/Quicksort

    template <typename T>
    extern inline
    int64_t partition(T* array, int64_t left, int64_t right) 
    {
      assert(left < right);
      right--;
      T pivot = array[(left+right)/2];
      
      while (left <= right) {
        // Scan from left and right looking for an element to swap
        while (array[left] < pivot) {
          left++;
        }
        while (array[right] > pivot) {
          right--;
        }
        if (left <= right) {
          std::swap(array[left], array[right]);
          right--;
          left++;
        }
      }
      return left;
    }
    
    template <typename T>
    class SorterTask: public ppp::Task {
    public:
      SorterTask(T* array, int64_t left, int64_t right, int64_t grainsize) { 
        m_array = array;
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
          PPP_DEBUG_MSG("std::sort: [" + to_string(m_left) + ", " + to_string(m_right) + "]");
          PPP_DEBUG_MSG("std::sort: [" + to_string(&m_array[m_left]) + ", " + to_string(&m_array[m_right]) + "]");
          std::sort(&m_array[m_left], &m_array[m_right]); 
          return;
        }
        
        int64_t pivot = partition(m_array, m_left, m_right);
        assert(pivot < m_right);
        
        PPP_DEBUG_MSG("Split: [" + to_string(m_left) + ", " + to_string(pivot) + "] [" +
                      to_string(pivot) + ", " + to_string(m_right) + "]");
        ppp::TaskGroup tg;
        SorterTask t1(m_array, m_left, pivot, m_grainsize);
        SorterTask t2(m_array, pivot, m_right, m_grainsize);
        tg.spawn(t1);
        tg.spawn(t2);
        tg.wait();
      }
    private:
      T* m_array;
      int64_t m_left;
      int64_t m_right;
      int64_t m_grainsize;
    };
  }
    
  template <typename T>
  extern inline
  void parallel_sort(T* array, int64_t left, int64_t right, int64_t grainsize=0) 
  {
    if (grainsize == 0) {
      grainsize = (right-left+1) / (get_thread_count()*4);
    }
    PPP_DEBUG_MSG("parallel_sort grainsize: " + to_string(grainsize));
            
    internal::SorterTask<T> t(array, left, right, grainsize);
    t.execute();
    PPP_DEBUG_MSG("parallel_sort done");
  }
}

#endif
