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
#include <unistd.h>
#include <assert.h>
#include <time.h>
#include <ctime>
#include <iostream>
#include <set>

#include "ppp.h"
#include "compute.h"
#include "random.h"
#include "parallel_sort.h"
#include "parallel_for.h"

static int64_t distance(int64_t a, int64_t b);
int64_t check_two(int64_t particle, int64_t num_particles, int64_t *location, int64_t *weight, int64_t *radius);

void fill_rand(int64_t n, int64_t* array, int max_value)
{
  // Floyd Algorithm for generating a set of unique random numbers
  // Programming Pearls, p. 225
  // Note: numbers are unique, but would need to be shuffled to
  // guarantee the order is random, too

  std::set<int64_t> S;
  int64_t j = max_value - n;
  for (int64_t i = 0; i<n; i++) {
    int64_t t = ppp::bigrand(j+1, 3);
    if (S.find(t) != S.end()) { // If already present, just use "j"
      t = j;
    }
    j++;
    S.insert(t);
    array[i] = t;
  }
}

// Parameters and default values
static int64_t num_particles = 0;
static int64_t sparsity = 16;
static int64_t max_weight = 1000;
static int64_t radius = 500;
static int num_trials = 5;
static int num_threads = 1;
static int64_t grainsize = 0;

// Data arrays
static int64_t* xlocation_array;
static int64_t* weight_array;
static int64_t* radius_array;
static int64_t* answer_array;
static int64_t* check_particle_array;
static int64_t* check_answer_array;

struct particle_t {
  int64_t xlocation;
  int64_t weight;
  int64_t radius;
};

bool operator<(const particle_t& p1, const particle_t& p2) {
  return p1.xlocation < p2.xlocation;
}

bool operator>(const particle_t& p1, const particle_t& p2) {
  return p1.xlocation > p2.xlocation;
}

static particle_t* sort_array;

void driver()  // Called for each thread
{
  const int num_to_check = 100;
  // Calculated values
  int64_t max_location = num_particles * sparsity;
  int64_t half_radius = int64_t((double(radius)/2) * double(sparsity));

  // Allocation 
  xlocation_array = new int64_t[num_particles];
  weight_array = new int64_t[num_particles];
  radius_array = new int64_t[num_particles];
  answer_array = new int64_t[num_particles];
  sort_array = new particle_t[num_particles];
  
  check_particle_array = new int64_t[num_to_check];
  check_answer_array = new int64_t[num_to_check];
  
  double sort_elapsed_time = 0.0;
  double compute_elapsed_time = 0.0;
  for (int trial = 1; trial <= num_trials; trial++) {

    // fill_rand(num_particles, xlocation_array, max_location);
    
    for (int64_t i = 0; i<num_particles; i++) {
      xlocation_array[i] = ppp::bigrand(max_location, 3);
      weight_array[i] = ppp::bigrand(max_weight);
      radius_array[i] = half_radius + ppp::bigrand(half_radius);
      answer_array[i] = 0;
    }
    
    /////////////////////////

    // Put in AOS (array of structures form)
    for (int64_t i = 0; i<num_particles; i++) {
      sort_array[i].xlocation = xlocation_array[i];
      sort_array[i].weight = weight_array[i];
      sort_array[i].radius = radius_array[i];
    }

    // Sort
    double sort_start_time = ppp::get_seconds();
    ppp::parallel_sort(sort_array, 0, num_particles);
    double sort_end_time = ppp::get_seconds();
    double sort_seconds = sort_end_time - sort_start_time;

    // Check the sort
    int64_t prior = 0;
    for (int64_t i = 0; i<num_particles; i++) {
      assert(prior <= sort_array[i].xlocation); // Check the sort
      prior = sort_array[i].xlocation;
    }

    // Put in SOA (structure of arrays form)
    for (int64_t i = 0; i<num_particles; i++) {
      xlocation_array[i] = sort_array[i].xlocation;
      weight_array[i] = sort_array[i].weight;
      radius_array[i] = sort_array[i].radius;
    }

    // Pre-calculate values for randomized spot checking
    for (int64_t i = 0; i<num_to_check; i++) {
      int64_t particle_to_check = ppp::bigrand(num_particles);
      check_particle_array[i] = particle_to_check;
      check_answer_array[i] = check_two(particle_to_check, num_particles, xlocation_array, weight_array, radius_array);
    }

    double compute_start_time = ppp::get_seconds();

    ComputationTwo f(num_particles, xlocation_array, weight_array, radius_array, answer_array);
    ppp::parallel_for(0, num_particles, &f, grainsize);

    //    compute_two(num_particles, xlocation_array, weight_array, radius_array, answer_array);

    double compute_end_time = ppp::get_seconds();
    double compute_seconds = compute_end_time - compute_start_time;

    /////////////////////////
        
    // Check the pre-calculated answers for randomized spot checking
    for (int64_t i = 0; i<num_to_check; i++) {
      int64_t particle_to_check = check_particle_array[i];
      int64_t expected_answer = check_answer_array[i];
      int64_t calculated_answer = answer_array[particle_to_check];
      if (expected_answer != calculated_answer) {
        std::cout << "ERROR: particle " << particle_to_check << " had value `" << calculated_answer
                  << "` not the expected value `" << expected_answer << "`\n";
      }
    }
    
    std::cout << "trial: " << trial << ", ";
    std::cout << "user: " << ppp::getuser() << ", ";
    std::cout << "host: " << ppp::gethost() << ", ";
    std::cout << "threads: " << num_threads << ", ";
    std::cout << "particles: " << num_particles << ", ";
    std::cout << "grainsize: " << grainsize << ", ";
    std::cout << "sort_seconds: " << sort_seconds << ", ";
    std::cout << "compute_seconds: " << compute_seconds << "\n";
    
    sort_elapsed_time += sort_seconds;
    compute_elapsed_time += compute_seconds;
  }
  
  std::cout << "average:  ";
  std::cout << "user: " << ppp::getuser() << ", ";
  std::cout << "host: " << ppp::gethost() << ", ";
  std::cout << "threads: " << num_threads << ", ";
  std::cout << "particles: " << num_particles << ", ";
  std::cout << "grainsize: " << grainsize << ", ";
  std::cout << "sort_seconds: " << sort_elapsed_time/double(num_trials) << ", ";
  std::cout << "compute_seconds: " << compute_elapsed_time/double(num_trials) << "\n";
}

static struct option long_options[] =
  {
    {"particles", required_argument, 0, 'p'},
    {"sparsity", required_argument, 0, 's'},
    {"weight", required_argument, 0, 'w'},
    {"radius", required_argument, 0, 'r'},
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
    int c = getopt_long_only(argc, argv, "p:s:w:r:t:n:g:",
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
      
    case 's':
      sparsity = atoll(optarg);
      break;
      
    case 'w':
      max_weight = atoll(optarg);
      break;
      
    case 'r':
      radius = atoll(optarg);
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



/******************** code for checking the results *********************/

static int64_t distance(int64_t a, int64_t b) 
{
  if (a > b) {
    return a-b;
  } else {
    return b-a;
  }
}

int64_t check_two(int64_t particle, int64_t num_particles, int64_t *location, int64_t *weight, int64_t *radius)
{
  // Computation 2
  int64_t answer = 0;
  for (int64_t j = 0; j < num_particles; j++) {
    if (distance(location[particle], location[j]) < radius[particle]) {
      answer += (weight[particle] + weight[j]) | distance(location[particle], location[j]);
    }
  }
  return answer;
}

