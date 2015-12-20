// Penn Parallel Primitives Library
// Author: Prof. Milo Martin
// University of Pennsylvania
// Spring 2010

#include "ppp.h"
#include "TaskGroup.h"

namespace ppp {

  namespace internal {
    // Task-based scheduling variables
    TaskQueue* g_queues_ptr = NULL;
    atomic<int> g_stop_counter;
    
  }
  
  using namespace internal;

  void TaskGroup::spawn(Task& t) {
    assert(g_queues_ptr != NULL);
    //TaskQueue& queue = g_queues_ptr[0]; // centralized stack
    TaskQueue& queue = g_queues_ptr[get_thread_id()];  // ASSIGNMENT: use per-thread task queue with "get_thread_id()"
    m_wait_counter.fetch_and_inc();
    t.setCounter(&m_wait_counter);
    queue.enqueue(&t);
  }
  
  void process_tasks(const atomic<int>* counter)
  {
    //TaskQueue& queue = g_queues_ptr[0]; // centrailized stack
    TaskQueue& queue = g_queues_ptr[get_thread_id()];  // ASSIGNMENT: use per-thread task queue with "get_thread_id()"

    while (counter->get() != 0) {
      PPP_DEBUG_EXPR(queue.size());
      
      // Dequeue from local queue
      Task* task = queue.dequeue();
      
      // ASSIGNMENT: add task stealing
      // always steal from the one has the most tasks
      if(queue.size()==0)
      {
          int cnt = get_thread_count();
          int id  = 0;
          int tasks_cnt = 0;
          for (int i=0;i<cnt;i++)
          {
              if(g_queues_ptr[ (get_thread_id()+i+1)%cnt ].size()>tasks_cnt)
              {
                  tasks_cnt = g_queues_ptr[ (get_thread_id()+i+1)%cnt ].size();
                  id = (get_thread_id()+i+1)%cnt;
              }
          }
          Task* steal_task = g_queues_ptr [id].steal();
          queue.enqueue(steal_task);
      }
      
      if (task != NULL) {
        task->execute(); // overloaded method
        task->post_execute(); // cleanup, method of base class
      }
    }
  }
}

