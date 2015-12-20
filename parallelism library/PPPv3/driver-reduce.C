// CIS534 homework
// Author: Prof. Milo Martin
// University of Pennsylvania
// Spring 2010

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <time.h>
#include <ctime>
#include <iostream>
#include <set>

#include "ppp.h"
#include "compute.h"
#include "random.h"
#include "parallel_reduce.h"

// Parameters and default values
static int64_t num_particles = 0;
static int num_trials = 5;
static int num_threads = 1;
static int64_t grainsize = 0;

// Data arrays
static int64_t* data_array;

void driver()
{
  // Allocation 
  data_array = new int64_t[num_particles];
  
  double elapsed_time = 0.0;
  for (int trial = 1; trial <= num_trials; trial++) {

    for (int64_t i = 0; i<num_particles; i++) {
      data_array[i] = ppp::threadsafe_random() & 0xffff;
    }
    
    // Reduce
    double start_time = ppp::get_seconds();
    int64_t answer = ppp::parallel_reduce(data_array, 0, num_particles, grainsize);
    double end_time = ppp::get_seconds();
    double seconds = end_time - start_time;

    // Check reduce
    int64_t accumulator = 0;
    for (int64_t i = 0; i<num_particles; i++) {
      accumulator += data_array[i];
    }
    assert(answer == accumulator);
    
    std::cout << "trial: " << trial << ", ";
    std::cout << "user: " << ppp::getuser() << ", ";
    std::cout << "host: " << ppp::gethost() << ", ";
    std::cout << "threads: " << num_threads << ", ";
    std::cout << "particles: " << num_particles << ", ";
    std::cout << "grainsize: " << grainsize << ", ";
    std::cout << "seconds: " << seconds << "\n";
    
    elapsed_time += seconds;
  }
  
  std::cout << "average:  ";
  std::cout << "user: " << ppp::getuser() << ", ";
  std::cout << "host: " << ppp::gethost() << ", ";
  std::cout << "threads: " << num_threads << ", ";
  std::cout << "particles: " << num_particles << ", ";
  std::cout << "grainsize: " << grainsize << ", ";
  std::cout << "seconds: " << elapsed_time/double(num_trials) << "\n";
}

static struct option long_options[] =
  {
    {"particles", required_argument, 0, 'p'},
    {"trials", required_argument, 0, 't'},
    {"threads", required_argument, 0, 'n'},
    {"grainsize", required_argument, 0, 'g'},
    {0, 0, 0, 0}
  };

int main(int argc, char** argv)
{
  
  // Code based on the example code in the getopt documentation
  while (true) {

    int option_index = 0;    
    int c = getopt_long_only(argc, argv, "p:t:n:g:",
                             long_options, &option_index);
    
    /* Detect the end of the options. */
    if (c == -1)
      break;
    
    switch (c) {
    case 0:
      /* If this option set a flag, do nothing else now. */
      break;
      
    case 'p':
      num_particles = atoll(optarg);
      break;
            
    case 't':
      num_trials = atoi(optarg);
      break;

    case 'n':
      num_threads = atoi(optarg);
      break;

    case 'g':
      grainsize = atoll(optarg);
      break;
      
    case '?':
      /* getopt_long already printed an error message. */
      exit(1);
      break;
      
    default:
      exit(1);
    }
  }

  if (num_particles <= 0) {
    fprintf(stderr, "Error: --particles must be set\n");
    exit(1);
  }
  
  ppp::init_tasks(driver, num_threads);
}

