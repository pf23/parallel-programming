// Penn Parallel Primitives Library
// Author: Prof. Milo Martin
// University of Pennsylvania
// Spring 2010

#ifndef PPP_TASK_GROUP_H
#define PPP_TASK_GROUP_H

#include "ppp.h"
#include "atomic.h"
#include "TaskQueue.h"
#include "random.h"

class TaskQueue;

namespace ppp {
  
  namespace internal {
    extern TaskQueue* g_queues_ptr;
  }

  using namespace internal;

  void process_tasks(const atomic<int>* counter);

  class TaskGroup {
  public:
    TaskGroup() { 
      m_wait_counter.set(0);
      assert(g_queues_ptr != NULL);
    }

    void spawn(Task *t) {
      t->setHeap(); // Make sure the task knows to delete itself
      spawn(*t);
    }

    void spawn(Task& t); // Defined in TaskGroup.C

    void wait() {
      process_tasks(&m_wait_counter);
    }

    ~TaskGroup() {
      assert(m_wait_counter.get() == 0); // Ensure join has been called
    }

  private:
    atomic<int> m_wait_counter;

    // Private copy constructor and assignment operator
    TaskGroup(const TaskGroup& obj);
    TaskGroup& operator=(const TaskGroup& obj);
  };
}
#endif




