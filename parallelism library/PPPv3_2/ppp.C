// Penn Parallel Primitives Library
// Author: Prof. Milo Martin
// University of Pennsylvania
// Spring 2010

#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <ctype.h>
#include <time.h>
#include <ctime>

#include "ppp.h"
#include "machine.h"
#include "random.h"
#include "atomic.h"
#include "TaskQueue.h"
#include "TaskGroup.h"

namespace ppp {

  namespace internal {
    static bool initialized = false;
    __thread int s_thread_id = 0;
    __thread int s_thread_count = 0;

    struct thread_info_t {
      pthread_t id;
      int thread_id;
      int thread_count;
      void (*thread_func) (void);
    };

    static atomic<int> g_stop_counter;
  }

  void reseed_random_per_thread();
  uint32_t threadsafe_random();


  using namespace internal;
  
  // Helper function to take care of the void* conversion
  static void* init_start(void *ptr) 
  {
    thread_info_t* info_ptr = (thread_info_t *) ptr;    // cast from void*
    s_thread_id = info_ptr->thread_id;
    s_thread_count = info_ptr->thread_count;
    reseed_random_per_thread(); // Give each thread its own set of random numbers
    
    PPP_DEBUG_MSG("Spawning thread " + to_string(s_thread_id));

    if (g_queues_ptr == NULL) {
      // Not doing task queues, spawn N threads
      (info_ptr->thread_func)();  // Call through function pointer
    }
    else {
      // Not doing task queues, spawn N threads
      g_queues_ptr[s_thread_id].setThreadID(s_thread_id);
      if (s_thread_id == 0) {
        (info_ptr->thread_func)();  // Call through function pointer
      }
      else {
        process_tasks(&g_stop_counter); // Start task worker thread
      }
    }
    PPP_DEBUG_MSG("Exiting thread " + to_string(s_thread_id));
    return NULL;  // terminates thread
  }

  void init_tasks(void (*func) (void), int num_threads)
  {
    g_queues_ptr = new TaskQueue[num_threads];
    init(func, num_threads);
      
  }

  void init(void (*func) (void), int num_threads)
  {
    // Only init threads once
    if (initialized) {
      printf("Error: init() called more than once\n");
      exit(1);
    }
    initialized = true;

    if (num_threads < 1) {
      printf("Error: number of threads passed to init() must be > 0\n");
      exit(1);
    }

    if (num_threads > MAX_THREADS) {
      printf("Error: number of threads (%d) exceeds MAX_THREADS (%d)\n", num_threads, MAX_THREADS);
      exit(1);
    }

    g_stop_counter.set(1);
    
    // Allocate and initialize the thread info array
    thread_info_t* info_array = new thread_info_t[num_threads];
    for (int i=0; i<num_threads; i++) {
      info_array[i].thread_id = i;
      info_array[i].thread_count = num_threads;
      info_array[i].thread_func = func;
    }
    
    PPP_DEBUG_MSG("Spawning threads");

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    
    // Set stack size of each thread
    pthread_attr_setstacksize(&attr, 1024 * 1024 * 512); 
    // 128MB per thread is pretty big.  Only required for centralized
    // work queue implementation, as that can blow the stack out.

    // Create threads 1 through N-1
    for (int i=1; i<num_threads; i++) {
      int error = pthread_create(&(info_array[i].id), &attr, init_start, &info_array[i]);
      if (error) { 
        printf("Error: Thread creation error\n");
        exit(1);
      }
    }
    
    // "Create" thread zero
    init_start(&info_array[0]);

    PPP_DEBUG_MSG("Stopping task worker threads");
    g_stop_counter.set(0); // Stop the other worker threads
    
    // Wait for all threads to terminate
    for (int i=1; i<num_threads; i++) {
      pthread_join(info_array[i].id, NULL);
    }
    PPP_DEBUG_MSG("All threads joined");
    
    delete [] info_array;
  }

  //////////////////////////////////////////

  const char* gethost()
  {
    const char* host = getenv("HOST");
    if (host == NULL) {
      host = "<unknown>";
    }
  return host;
  }

  const char* getuser()
  {
    const char* user = getenv("USER");
    if (user == NULL) {
      user = "<unknown>";
    }
    return user;
  }
  
  double get_seconds()
  {
    struct timespec ts;

    clock_gettime(CLOCK_REALTIME, &ts);
    double tv_sec = double(ts.tv_sec);
    double tv_nsec = double(ts.tv_nsec);
    double seconds = tv_sec + (tv_nsec / double(1000000000.0));
    return seconds;
  }
}


