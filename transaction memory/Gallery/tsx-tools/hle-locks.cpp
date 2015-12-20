#include <immintrin.h>

int main()
{

  int lock;


while (__atomic_exchange_n(&lock, 1, __ATOMIC_ACQUIRE|__ATOMIC_HLE_ACQUIRE) != 0) {

  int val;

  /* Wait for lock to become free again before retrying. */
  do {

    _mm_pause();

    /* Abort speculation */
    __atomic_load(&lock, &val, __ATOMIC_CONSUME);

  } while (val == 1);

 }
 return 0;

}


