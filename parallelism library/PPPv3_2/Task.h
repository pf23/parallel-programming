// Penn Parallel Primitives Library
// Author: Prof. Milo Martin
// University of Pennsylvania
// Spring 2010

#ifndef PPP_TASK_H
#define PPP_TASK_H

#include "ppp.h"
#include "atomic.h"

namespace ppp {
  
  class Task {
  public:
    Task() { 
      m_parent_counter_ptr = NULL; 
      m_is_heap = false;
    }

    virtual void execute() = 0; // pure virtual function

    void setCounter(atomic<int>* counter_ptr) {
      assert(counter_ptr != NULL);
      assert(m_parent_counter_ptr == NULL); // only once
      m_parent_counter_ptr = counter_ptr;
    }

    void setHeap() { m_is_heap = true; }

    void post_execute() { 
      if (m_parent_counter_ptr != NULL) {
        int cnt = m_parent_counter_ptr->fetch_and_dec();
        assert(cnt > 0);
      } 
      if (m_is_heap) {
        delete this;
      }
    }
  private:
    atomic<int>* m_parent_counter_ptr;
    bool m_is_heap;

    // Private copy constructor and assignment operator
    Task(const Task& obj);
    Task& operator=(const Task& obj);
  };
}
#endif



