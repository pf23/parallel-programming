// Penn Parallel Primitives Library
// Author: Prof. Milo Martin
// University of Pennsylvania
// Spring 2010

#ifndef PPP_TASKQUEUE_H
#define PPP_TASKQUEUE_H

#include "Task.h"
#include "TTSLock.h"
#include <deque>

namespace ppp {

  class TaskQueue {
  public:
    TaskQueue() { m_id = -1; };

    int size() const { return m_deque.size(); }

    void setThreadID(int id) { m_id = id; }

    void enqueue(Task *t) {
      m_lock.acquire();
      m_deque.push_back(t);
      m_lock.release();
    }

    Task* dequeue() {
      m_lock.acquire();
      Task* t = NULL;
      if (!m_deque.empty()) {
        t = m_deque.back();
        m_deque.pop_back();
      }
      m_lock.release();
      return t;
    }

    Task* steal() {
      m_lock.acquire();
      Task* t = NULL; // Return NULL if deque is empty
      if (!m_deque.empty()) {
        t = m_deque.front();
        m_deque.pop_front();  // Steal from front
      }
      if (t != NULL) {
        PPP_DEBUG_MSG("Thread " + to_string(get_thread_id()) + " stealing from thread " + to_string(m_id));
      }
      m_lock.release();
      return t;
    }

  private:
    int m_id;
    TTSLock m_lock;
    std::deque<Task*> m_deque;

    // Private copy constructor and assignment operator
    TaskQueue(const TaskQueue& obj);
    TaskQueue& operator=(const TaskQueue& obj);
  };
}
#endif



