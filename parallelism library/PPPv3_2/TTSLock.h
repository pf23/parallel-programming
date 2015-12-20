// Penn Parallel Primitives Library
// Author: Prof. Milo Martin
// University of Pennsylvania
// Spring 2010

#ifndef PPP_TTSLOCK_H
#define PPP_TTSLOCK_H

#include "atomic.h"

namespace ppp {
  
  namespace {
    enum LockVal { 
      LockVal_Unlocked = 0, 
      LockVal_Locked = 1 
    };
  }

  class TTSLock {
  public:
    TTSLock() { m_var.set(LockVal_Unlocked); }

    void acquire() {
      while(true) {
        LockVal val = m_var.get();
        if (val == LockVal_Locked) {
          pause();
        } else {
          val = m_var.compare_and_swap(LockVal_Unlocked, LockVal_Locked);
          if (val == LockVal_Unlocked) {
            break;  // Success!
          }
        }
      }
    }
    
    bool try_acquire() {
      LockVal val = m_var.get();
      if (val == LockVal_Unlocked) {
        val = m_var.compare_and_swap(LockVal_Unlocked, LockVal_Locked);
        if (val == LockVal_Unlocked) {
          return true; // Success!
        }
      }
      return false; // Failed
    }

    void release() {
      m_var.set(LockVal_Unlocked);
    }

  private:
    atomic<LockVal> m_var;

    // Private copy constructor and assignment operator
    TTSLock(const TTSLock& obj);
    TTSLock& operator=(const TTSLock& obj);
  };
}
#endif



