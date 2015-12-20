// CIS534 homework
// Author: Prof. Milo Martin
// University of Pennsylvania
// Spring 2010

#include "ppp.h"
#include "compute.h"
#include <math.h>
#include <algorithm>


// void compute_two(int64_t num_particles, 
//                  int64_t *location, 
//                  int64_t *weight, 
//                  int64_t *radius, 
//                  int64_t *answer)
// {
//   // Computation 2
//   for (int64_t i = 0; i < num_particles; i++) {

//     int64_t radius_i = radius[i];
//     int64_t weight_i = weight[i];
//     int64_t location_i = location[i];
    
//     int64_t* right_ptr = std::lower_bound(&location[0], &location[num_particles], location_i + radius_i);
//     int64_t* left_ptr = std::upper_bound(&location[0], &location[num_particles], location_i - radius_i);
//     int64_t right = right_ptr - location;
//     int64_t left = left_ptr - location;

//     int64_t answer_i = 0;
//     for (int64_t j = left; j < i; j++) {
//       int64_t weight_j = weight[j];
//       int64_t location_j = location[j];
//       answer_i += (weight_i + weight_j) | (location_i - location_j);
//     }

//     for (int64_t j = i; j < right; j++) {
//       int64_t weight_j = weight[j];
//       int64_t location_j = location[j];
//       answer_i += (weight_i + weight_j) | (location_j - location_i);
//     }
//     answer[i] = answer_i;
//   }
// }

