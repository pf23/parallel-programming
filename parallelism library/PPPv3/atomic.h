// Penn Parallel Primitives Library
// Author: Prof. Milo Martin
// University of Pennsylvania
// Spring 2010

#ifndef PPP_ATOMIC_H
#define PPP_ATOMIC_H

#include "ppp.h"
#include "machine.h"

namespace ppp {

  template<typename T, size_t size> class atomic_internal;

  template<typename T>
  class atomic {
  public:
    inline void set(T value) { 
      compiler_fence();
      m_internal.m_var = value; 
      compiler_fence();
    }
    
    inline T get() const { 
      compiler_fence();
      T val = m_internal.m_var;
      compiler_fence();
      return val; 
    }
    
    inline T compare_and_swap(T test_value, T new_value) { 
      return m_internal.compare_and_swap(test_value, new_value);
    }
    
    inline T fetch_and_add(int64_t delta) { 
      assert(this != NULL);
      while(true) {
        T original_value = m_internal.m_var;
        T new_value = original_value + delta;
        T old = compare_and_swap(original_value, new_value);
        if (old == original_value) {
          return old;    // swap succeeded
        }
        pause(); // Hint to multithreaded processors that we're spinning
      }
    }
    
    inline T fetch_and_inc() { return fetch_and_add(1); }
    inline T fetch_and_dec() { return fetch_and_add(-1); }
  private:
    atomic_internal<T, sizeof(T)> m_internal;
    char m_padding[CACHE_BLOCK_SIZE-sizeof(T)];
  };  

  
  /***********************************************************/
  // The below code is model loosely after the way TBB uses a
  // "atomic_impl" as an internal class.  The atomic internal allow for
  // casting the type T into the right four-byte or eight-byte type for
  // doing the compare-and-swap operation.  The use of the union below
  // allows us to do type casts at the lowest level representation level
  // (the actual bits), thus avoiding any C++ type conversions.
  
  // 4-byte objects
  template<typename T>
  class atomic_internal<T, 4> {
  public:
    union converter { T orig; uint32_t bits; };   // Used for "clean" bit-level type conversion
    T m_var __attribute__ ((aligned(4)));
    T compare_and_swap(T test_value, T new_value) {
      converter test_value_c;
      test_value_c.orig = test_value;

      converter new_value_c;
      new_value_c.orig = new_value;

      converter result;
      result.bits = cmpswp((uint32_t *) &m_var, test_value_c.bits, new_value_c.bits);
      return result.orig;
    }
  };

  // 8-byte objects
  template<typename T>
  class atomic_internal<T, 8> {
  public:
    union converter { T orig; uint64_t bits; };   // Used for "clean" bit-level type conversion
    T m_var __attribute__ ((aligned(8)));
    T compare_and_swap(T test_value, T new_value) {
      converter test_value_c;
      test_value_c.orig = test_value;

      converter new_value_c;
      new_value_c.orig = new_value;

      converter result;
      result.bits = cmpswp((uint64_t *) &m_var, test_value_c.bits, new_value_c.bits);
      return result.orig;
    }
  };

  // Any other size objects, trigger error
  template<typename T, size_t size> 
  class atomic_internal {
  public:
    static_assert((sizeof(T) == 4) || (sizeof(T) == 8), "PPP library: Atomic types must be 4 or 8 bytes");
    T m_var;
    T compare_and_swap(T test_value, T new_value) {
      assert(false);  // never executed
      return m_var;   // never executed
    }
  };
}  

#endif

