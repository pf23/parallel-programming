// Penn Parallel Primitives Library
// Author: Prof. Milo Martin
// University of Pennsylvania
// Spring 2010

#ifndef PPP_MACHINE_H
#define PPP_MACHINE_H

#include "ppp.h"

namespace ppp {

  // 4-byte and 8-byte atomic operations
  static inline uint32_t cmpswp(volatile uint32_t *ptr, uint32_t test_value, uint32_t new_value);
  static inline uint64_t cmpswp(volatile uint64_t *ptr, uint64_t test_value, uint64_t new_value);
  static inline void compiler_fence();
  static inline void pause();

  const int CACHE_BLOCK_SIZE = 64; // This could be specialized for different architectures

#ifdef __x86_64__  /**************** x86-64 ****************/

  const int MAX_THREADS = 8;
    
  static inline uint32_t cmpswp(volatile uint32_t *ptr, uint32_t test_value, uint32_t new_value)
  {
    uint32_t result;
    // The following inline assembly code taken from 
    // Threading Building Blocks's machine/linux_intel64.h (GNU LGPL v2)

    __asm__ __volatile__("lock cmpxchg %2,%1"
                         : "=a"(result), "=m"(*(uint32_t *)ptr)
                         : "q"(new_value), "0"(test_value), "m"(*(uint32_t *)ptr)
                         : "memory");
    return result;
  }
  
  static inline uint64_t cmpswp(volatile uint64_t *ptr, uint64_t test_value, uint64_t new_value) 
  {
    uint64_t result;
    // The following inline assembly code taken from 
    // Threading Building Blocks's machine/linux_intel64.h (GNU LGPL v2)

    __asm__ __volatile__("lock cmpxchgq %2,%1"
                         : "=a"(result), "=m"(*(uint64_t *)ptr)
                         : "q"(new_value), "0"(test_value), "m"(*(uint64_t *)ptr)
                         : "memory");
    return result;
  }

  static inline void pause()
  {
    __asm__ __volatile__("pause");
  }

#elif __sparc  /**************** sparc ****************/
  
  const int MAX_THREADS = 128;

  static inline uint32_t cmpswp(volatile uint32_t *ptr, uint32_t test_value, uint32_t new_value) 
  {
    uint32_t result;
    // The following inline assembly code taken from Penn's port of 
    // Threading Building Blocks to SPARC/Solaris 
    // by Michajlo Matijkiw and Milo Martin 

    __asm__ __volatile__("cas\t[%1],%2,%0"
                         : "=r"(result)
                         : "r"(ptr), "r"(test_value), "0"(new_value)
                         : "memory");
    
    return result;
  }
  
  static inline uint64_t cmpswp(volatile uint64_t *ptr, uint64_t test_value, uint64_t new_value) 
  {
    uint64_t result;
    // The following inline assembly code taken from Penn's port of 
    // Threading Building Blocks to SPARC/Solaris 
    // by Michajlo Matijkiw and Milo Martin 
    __asm__ __volatile__("casx\t[%1],%2,%0"
                         : "=r"(result)
                         : "r"(ptr), "r"(test_value), "0"(new_value)
                         : "memory");
    
    return result;
  }

  static inline void pause()
  {
    // no "pause" on SPARC, as far as I know
  }

  
#else
  #error "Machine architecture not recognized"
#endif

  // The following null inline assembly should prevent the compiler from
  // moving any code around, thus acting as a fence to prevent compiler
  // optimizations
  static inline void compiler_fence() { __asm__ __volatile__ ("": : :"memory"); };
}

#endif

  

