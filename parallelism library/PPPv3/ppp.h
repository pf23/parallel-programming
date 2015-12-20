// Penn Parallel Primitives Library
// Author: Prof. Milo Martin
// University of Pennsylvania
// Spring 2010

#ifndef PPP_H
#define PPP_H

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <iostream>
#include <string>
#include <sstream>
#include <assert.h>

//#define PPP_DEBUG 1

namespace ppp {
  namespace internal {
    extern __thread int s_thread_id;
    extern __thread int s_thread_count;
  }

  void init(void (*func) (void), int num_threads);
  void init_tasks(void (*func) (void), int num_threads);
  extern inline int get_thread_id() { return internal::s_thread_id; }
  extern inline int get_thread_count() { return internal::s_thread_count; }

#ifdef PPP_DEBUG
  const bool PPP_DEBUG_FLAG = true;
#else
  const bool PPP_DEBUG_FLAG = false;
#endif

  const char* gethost();
  const char* getuser();
  double get_seconds();
}


#define PPP_DEBUG_MSG(MESSAGE)\
{\
  if (ppp::PPP_DEBUG_FLAG) {\
    std::ostringstream sstr;\
    sstr << "Debug: thread " << ppp::get_thread_id()\
         << " in fn "\
         << __PRETTY_FUNCTION__\
         << " in " << __FILE__ << ":"\
         << __LINE__ << ": "\
         << (MESSAGE) << "\n";\
    std::cout << sstr.str() << std::flush;\
  }\
}

#define PPP_DEBUG_EXPR(EXPR)\
{\
  if (ppp::PPP_DEBUG_FLAG) {\
    std::ostringstream sstr;\
    sstr << "Debug: thread " << ppp::get_thread_id()\
         << " in fn "\
         << __PRETTY_FUNCTION__\
         << " in " << __FILE__ << ":"\
         << __LINE__ << ": "\
         << #EXPR << " is "\
         << (EXPR) << "\n";\
    std::cout << sstr.str() << std::flush;\
  }\
}

template <typename T>
inline std::string to_string(const T& obj)
{
  std::stringstream sstr;
  sstr << obj;
  return sstr.str();
}

#endif
