#ifndef ROGUE_H

#define ROGUE_H
#include <mutex>
#include <immintrin.h>
//#include "./tsx-tools/rtm.h"
#include <pthread.h>
#include <time.h>
#include <sys/time.h>
#include "Lanes.h"
#include <unistd.h>
#define MAXLANENUM 1000
#define MAXRETRY 5

int rounds_completed = 0;
/*
class Rogue
{
public:
	Rogue(int color, int rate);
	~Rogue();

	// data 
	Color Bullet; // The bullet color to paint the lane
	int ShotRate; // Rate/s required to shoot the lanes
	int Success; // Rate/s of lanes shot by ROGUE   
};*/

class RogueCoarse
{
    public:
    	RogueCoarse(int rate, Color color, Lanes* Gallery, int rounds);
    	~RogueCoarse(){};

    private:
    	Color playercolor;    
};

RogueCoarse::RogueCoarse(int rate, Color color, Lanes* Gallery, int rounds)
{
  printf("Coarse Rogue\n");
  // initialize the mutex lock
  static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;

	playercolor = color;

  // initialize random seed
  srand(time(NULL));

 	int count = Gallery->Count();
  int laneNum;
  int Success = 0;
  struct timeval start, finish;

  gettimeofday(&start, 0);
 	while(rounds_completed < rounds)
  {
   		usleep(1000000/rate);

    	pthread_mutex_lock(&mtx);		//lock
    	// generate a random number from those lanes not colored
       	laneNum = rand() % count;

       	// check if all lanes are colored, if so, clean them
       	int colored_lanes_cnt=0;
       	for (int i=0;i<count;i++)
       	{
       		if(Gallery->Get(i)!=Color::white)
       			colored_lanes_cnt++;
       	}
       	if(colored_lanes_cnt >= count)
       	{
          rounds_completed++;
          //printf("Round %d\n", rounds_completed);
          Gallery->Print();
          Gallery->Clear();
          pthread_mutex_unlock (&mtx);    //unlock
          continue;
       	}

       	// find the next random lane that is not colored
       	while (Gallery->Get(laneNum) != Color::white)
       	{
       		laneNum++;
       		laneNum = laneNum % count;
       	}

       	assert (Gallery->Get(laneNum) == Color::white);
       	Gallery->Set(laneNum,color);
        Success++;
       	pthread_mutex_unlock (&mtx);    //unlock
  }   
  gettimeofday(&finish, 0);
  double elapsed_time = (finish.tv_sec - start.tv_sec) + (finish.tv_usec - start.tv_usec)*0.000001;
  printf("\n");
  if(playercolor == Color::red)
    printf("Red rogue successfully shot %f times per second \n(time elapsed %f s, success %d times, rate %d)\n", 
            Success/elapsed_time, elapsed_time, Success, rate);
  else
    printf("Blue rogue successfully shot %f times per second \n(time elapsed %f s, success %d times, rate %d)\n", 
            Success/elapsed_time, elapsed_time, Success, rate);
};


class RogueFine
{
	public:
    	RogueFine(int rate, Color color, Lanes* Gallery, int rounds);
    	~RogueFine(){};

  private:
    	Color playercolor;
    
};

RogueFine::RogueFine(int rate, Color color, Lanes* Gallery, int rounds)
{
  printf("Fine Rogue\n");
	// initialize the mutex locks for each lane
  static pthread_mutex_t mtx[MAXLANENUM] = PTHREAD_MUTEX_INITIALIZER;
 	static pthread_mutex_t all_mtx = PTHREAD_MUTEX_INITIALIZER;
  static pthread_cond_t all_cond = PTHREAD_COND_INITIALIZER;

  playercolor = color;	

 	// initialize random seed
 	srand(time(NULL));

 	int count = Gallery->Count();
 	int laneNum;
  int Success = 0;

  struct timeval start, finish;

  gettimeofday(&start, 0);
 	while(rounds_completed < rounds)
  {
   		usleep(1000000/rate);
    	
    	// generate a random number from those lanes not colored
      laneNum = rand() % count;

       // check if all lanes are colored, if so, clean them
        int colored_lanes_cnt=0;
        for (int i=0;i<count;i++)
        {
          if(Gallery->Get(i)!=Color::white)
            colored_lanes_cnt++;
        }
        if(colored_lanes_cnt >= count)
        {
       		pthread_mutex_lock(&all_mtx);
          if (Gallery->Get(1) != Color::white)
          {
            rounds_completed++;
            //printf("Round %d\n", rounds_completed);
            Gallery->Print();
         		Gallery->Clear();
          }
       		pthread_mutex_unlock(&all_mtx);
          continue;
       	}

       	// find the next random lane that is not colored
       	while(1)
       	{
       		if (Gallery->Get(laneNum) == Color::white)
	       		break;
       		laneNum++;
       		laneNum = laneNum % count;
       	}

       	pthread_mutex_lock(&mtx[laneNum]);		//lock the lane
       	if(Gallery->Get(laneNum) == Color::white)
        {
       	  Gallery->Set(laneNum,color);
          Success++;
        }
     	  pthread_mutex_unlock(&mtx[laneNum]);    //unlock the lane
  }	
  gettimeofday(&finish, 0);
  double elapsed_time = (finish.tv_sec - start.tv_sec) + (finish.tv_usec - start.tv_usec)*0.000001;

  pthread_mutex_lock(&all_mtx);
  printf("\n");
  if(playercolor == Color::red)
    printf("Red rogue successfully shot %f times per second \n(time elapsed %f s, success %d times, rate %d)\n", 
            Success/elapsed_time, elapsed_time, Success, rate);
  else
    printf("Blue rogue successfully shot %f times per second \n(time elapsed %f s, success %d times, rate %d)\n", 
            Success/elapsed_time, elapsed_time, Success, rate);
  pthread_mutex_unlock(&all_mtx);
}

class RogueTM
{
    public:
      RogueTM(int rate, Color color, Lanes* Gallery, int rounds);
      ~RogueTM(){};

    private:
      Color playercolor; 
};

RogueTM::RogueTM(int rate, Color color, Lanes* Gallery, int rounds)
{
  printf("TM Rogue\n");
  static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
  playercolor = color;

  // initialize random seed
  srand(time(NULL));

  int count = Gallery->Count();
  int laneNum;
  int Success = 0;
  double Abort = 0.0;
  struct timeval start, finish;

  gettimeofday(&start, 0);
  while(rounds_completed < rounds)
  {
      int nretries = 0;
      int status;
      while (nretries < MAXRETRY)
      {
        usleep(1000000/rate);
        // generate a random number from those lanes not colored
        laneNum = rand() % count;

        // find the next random lane that is not colored
        int trytimes = 0;
        for (trytimes = 0; trytimes<count; trytimes++)
        {
          if(Gallery->Get(laneNum) != Color::white)
          {
            laneNum++;
            laneNum = laneNum % count;
          }
          else
            break;
        }

        if(trytimes == count)
        {
            // do cleaning
            pthread_mutex_lock(&mtx);
            if (Gallery->Get(1)!=Color::white) 
            {
              rounds_completed++;
              //printf("Round: %d\n", rounds_completed);
              Gallery->Print();
              Gallery->Clear();
            }
            pthread_mutex_unlock(&mtx);
        }
        else
        {
            // do shooting
            if ((status = _xbegin()) == _XBEGIN_STARTED) 
            {
              Gallery->Set(laneNum, color);
              Success++;
              _xend ();
              break;
            }
            else
            {
              nretries++;
            }
        }            
      }
      if(nretries == MAXRETRY)
        Abort++;
            
  }   
  gettimeofday(&finish, 0);
  double elapsed_time = (finish.tv_sec - start.tv_sec) + (finish.tv_usec - start.tv_usec)*0.000001;
  printf("\n");
  if(playercolor == Color::red)
    printf("Red rogue successfully shot %f times per second \n(time elapsed %f s, success %d times, rate %d)\n", 
            Success/elapsed_time, elapsed_time, Success, rate);
  else
    printf("Blue rogue successfully shot %f times per second \n(time elapsed %f s, success %d times, rate %d)\n", 
            Success/elapsed_time, elapsed_time, Success, rate);
  printf("Abort rate: %f (abort %f times)\n", Abort / (Abort + Success), Abort);

}

class RogueCoarse2
{
    public:
      RogueCoarse2(int rate, Color color, Lanes* Gallery, int rounds);
      ~RogueCoarse2(){};

    private:
      Color playercolor;    
};

RogueCoarse2::RogueCoarse2(int rate, Color color, Lanes* Gallery, int rounds)
{
  printf("Coarse2 Rogue\n");
  // initialize the mutex lock
  static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;

  playercolor = color;

  // initialize random seed
  srand(time(NULL));

  int count = Gallery->Count();
  int laneNum;
  int Success = 0;
  struct timeval start, finish;

  gettimeofday(&start, 0);
  while(rounds_completed < rounds)
  {
      usleep(1000000/rate);

      pthread_mutex_lock(&mtx);   //lock
      // check if all lanes are colored, if so, clean them
      int colored_lanes_cnt=0;
      for (int i=0;i<count;i++)
      {
        if(Gallery->Get(i)!=Color::white)
          colored_lanes_cnt++;
      }
      if(colored_lanes_cnt >= count)
      {
        rounds_completed++;
        //printf("Round %d\n", rounds_completed);
        Gallery->Print();
        Gallery->Clear();
        pthread_mutex_unlock(&mtx);
        continue;
      }

      // generate 2 random numbers from those lanes not colored
      for (int j=0;j<2;j++)
      {
        laneNum = rand() % count;

        // find the next random lane that is not colored
        while (Gallery->Get(laneNum) != Color::white)
        {
          laneNum++;
          laneNum = laneNum % count;
        }

        assert (Gallery->Get(laneNum) == Color::white);
        Gallery->Set(laneNum,color);
        Success++;
      }
      pthread_mutex_unlock (&mtx);    //unlock
  } 
  gettimeofday(&finish, 0);
  double elapsed_time = (finish.tv_sec - start.tv_sec) + (finish.tv_usec - start.tv_usec)*0.000001;
  printf("\n");
  if(playercolor == Color::red)
    printf("Red rogue successfully shot %f times per second \n(time elapsed %f s, success %d times, rate %d)\n", 
            Success/elapsed_time, elapsed_time, Success, rate);
  else
    printf("Blue rogue successfully shot %f times per second \n(time elapsed %f s, success %d times, rate %d)\n", 
            Success/elapsed_time, elapsed_time, Success, rate);
};

class RogueFine2
{
  public:
      RogueFine2(int rate, Color color, Lanes* Gallery, int rounds);
      ~RogueFine2(){};

  private:
      Color playercolor;
    
};

RogueFine2::RogueFine2(int rate, Color color, Lanes* Gallery, int rounds)
{
  printf("Fine2 Rogue\n");
  // initialize the mutex locks for each lane
  static pthread_mutex_t mtx[MAXLANENUM] = PTHREAD_MUTEX_INITIALIZER;
  static pthread_mutex_t all_mtx = PTHREAD_MUTEX_INITIALIZER;
  static pthread_cond_t all_cond = PTHREAD_COND_INITIALIZER;

  playercolor = color;  

  // initialize random seed
  srand(time(NULL));

  int count = Gallery->Count();
  int laneNum;
  int Success = 0;

  struct timeval start, finish;

  gettimeofday(&start, 0);
  while(rounds_completed < rounds)
  {
      usleep(1000000/rate);
      
       // check if all lanes are colored, if so, clean them
        int colored_lanes_cnt=0;
        for (int i=0;i<count;i++)
        {
          if(Gallery->Get(i)!=Color::white)
            colored_lanes_cnt++;
        } 
        if(colored_lanes_cnt >= count)
        {
          pthread_mutex_lock(&all_mtx);
          if (Gallery->Get(1) != Color::white)
          {
            rounds_completed++;
            //printf("Round %d\n", rounds_completed);
            Gallery->Print();
            Gallery->Clear();
          }
          pthread_mutex_unlock(&all_mtx);
        }

      for(int j=0;j<2;j++)
      {
        // generate a random number from those lanes not colored
        laneNum = rand() % count;

        // find the next random lane that is not colored
        for(int j=0;j<2*count;j++)
        {
          if (Gallery->Get(laneNum) == Color::white)
            break;
          laneNum++;
          laneNum = laneNum % count;
        }

        //if(pthread_mutex_trylock(&mtx[laneNum]) == EBUSY)
        // continue;
        pthread_mutex_lock(&mtx[laneNum]);    //lock the lane
        //if(Gallery->Get(laneNum) == Color::white)
        {
          Gallery->Set(laneNum,color);
          Success++;
        }
        pthread_mutex_unlock(&mtx[laneNum]);    //unlock the lane
      }
  } 
  gettimeofday(&finish, 0);
  double elapsed_time = (finish.tv_sec - start.tv_sec) + (finish.tv_usec - start.tv_usec)*0.000001;

  pthread_mutex_lock(&all_mtx);
  printf("\n");
  if(playercolor == Color::red)
    printf("Red rogue successfully shot %f times per second \n(time elapsed %f s, success %d times, rate %d)\n", 
            Success/elapsed_time, elapsed_time, Success, rate);
  else
    printf("Blue rogue successfully shot %f times per second \n(time elapsed %f s, success %d times, rate %d)\n", 
            Success/elapsed_time, elapsed_time, Success, rate);
  pthread_mutex_unlock(&all_mtx);
}

class RogueTM2
{
    public:
      RogueTM2(int rate, Color color, Lanes* Gallery, int rounds);
      ~RogueTM2(){};

    private:
      Color playercolor; 
};

RogueTM2::RogueTM2(int rate, Color color, Lanes* Gallery, int rounds)
{
  printf("TM2 Rogue\n");
  static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
  playercolor = color;

  // initialize random seed
  srand(time(NULL));

  int count = Gallery->Count();
  int laneNum[2];
  int Success = 0;
  double Abort = 0.0;
  struct timeval start, finish;

  gettimeofday(&start, 0);
  while(rounds_completed < rounds)
  {
      int nretries = 0;
      int status;
      while (nretries < MAXRETRY)
      {
        usleep(1000000/rate);
        // generate a random number from those lanes not colored
        laneNum[0] = rand() % count;
        laneNum[1] = rand() % count;

        // find the next random lane that is not colored
        int trytimes = 0;
        for (trytimes = 0; trytimes<count; trytimes++)
        {
          if(Gallery->Get(laneNum[0]) != Color::white)
          {
            laneNum[0]++;
            laneNum[0] = laneNum[0] % count;
          }
          else
            break;
        }

        if(trytimes < count)
          for (trytimes = 0; trytimes<count; trytimes++)
          {
            if(Gallery->Get(laneNum[1]) != Color::white)
            {
              laneNum[1]++;
              laneNum[1] = laneNum[1] % count;
            }
            else
              break;
          }

        if(trytimes == count)
        {
            // do cleaning
            pthread_mutex_lock(&mtx);
            if (Gallery->Get(1)!=Color::white) 
            {
              rounds_completed++;
              //printf("Round: %d\n", rounds_completed);
              Gallery->Print();
              Gallery->Clear();
            }
            pthread_mutex_unlock(&mtx);
        }
        else
        {
            // do shooting 2 times
            if ((status = _xbegin()) == _XBEGIN_STARTED) 
            {
              Gallery->Set(laneNum[0], color);
              Gallery->Set(laneNum[1], color);
              Success+=2;
              _xend ();
              break;
            }
            else
            {
              nretries++;
            }
        }            
      }
      if(nretries == MAXRETRY)
        Abort++;
            
  }   
  gettimeofday(&finish, 0);
  double elapsed_time = (finish.tv_sec - start.tv_sec) + (finish.tv_usec - start.tv_usec)*0.000001;
  printf("\n");
  if(playercolor == Color::red)
    printf("Red rogue successfully shot %f times per second \n(time elapsed %f s, success %d times, rate %d)\n", 
            Success/elapsed_time, elapsed_time, Success, rate);
  else
    printf("Blue rogue successfully shot %f times per second \n(time elapsed %f s, success %d times, rate %d)\n", 
            Success/elapsed_time, elapsed_time, Success, rate);
  printf("Abort rate: %f (abort %f times)\n", Abort / (Abort + Success), Abort);
}

//**************************************************
// called by the other ***Cleaner classes
// use conditional wait mutex to synchronize
bool clean_command = false;
pthread_mutex_t cln_mtx = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cln_cond = PTHREAD_COND_INITIALIZER;

class RogueCleaner
{
    public:
      RogueCleaner(Lanes* Gallery, int rounds);
      ~RogueCleaner(){};
};

RogueCleaner::RogueCleaner(Lanes* Gallery, int rounds)
{
  while(rounds_completed < rounds)
  {
    pthread_mutex_lock(&cln_mtx);
    pthread_cond_wait(&cln_cond, &cln_mtx);
    if(clean_command == true)
    {
      clean_command = false;
      rounds_completed++;
      //printf("Round %d\n", rounds_completed);
      Gallery->Print();
      Gallery->Clear();
      pthread_cond_broadcast(&cln_cond);
    }
    pthread_mutex_unlock(&cln_mtx);
  }
  
}

class RogueCoarseCleaner
{
  public:
    RogueCoarseCleaner(int rate, Color color, Lanes* Gallery, int rounds);
    ~RogueCoarseCleaner(){};

  private:
    Color playercolor;
    
};

RogueCoarseCleaner::RogueCoarseCleaner(int rate, Color color, Lanes* Gallery, int rounds)
{
  printf("Coarse_clean Rogue\n");
    // initialize the mutex lock
  static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;

  playercolor = color;

  // initialize random seed
  srand(time(NULL));

  int count = Gallery->Count();
  int laneNum;
  int Success = 0;
  struct timeval start, finish;

  gettimeofday(&start, 0);
  while(rounds_completed < rounds)
  {
      usleep(1000000/rate);

      pthread_mutex_lock(&mtx);   //lock
      // generate a random number from those lanes not colored
        laneNum = rand() % count;

        // check if all lanes are colored, if so, clean them
        int colored_lanes_cnt=0;
        for (int i=0;i<count;i++)
        {
          if(Gallery->Get(i)!=Color::white)
            colored_lanes_cnt++;
        }
        if(colored_lanes_cnt >= count)
        {
          pthread_mutex_lock(&cln_mtx);
          pthread_cond_signal(&cln_cond);
          clean_command = true;
          // wait after cleaning working is done
          pthread_cond_wait(&cln_cond, &cln_mtx);
          pthread_mutex_unlock(&cln_mtx);
          pthread_mutex_unlock (&mtx);
          continue;
        }

        // find the next random lane that is not colored
        while (Gallery->Get(laneNum) != Color::white)
        {
          laneNum++;
          laneNum = laneNum % count;
        }

        assert (Gallery->Get(laneNum) == Color::white);
        Gallery->Set(laneNum,color);
        Success++;
      pthread_mutex_unlock (&mtx);    //unlock
  } 
  gettimeofday(&finish, 0);
  double elapsed_time = (finish.tv_sec - start.tv_sec) + (finish.tv_usec - start.tv_usec)*0.000001;
  printf("\n");
  if(playercolor == Color::red)
    printf("Red rogue successfully shot %f times per second \n(time elapsed %f s, success %d times, rate %d)\n", 
            Success/elapsed_time, elapsed_time, Success, rate);
  else
    printf("Blue rogue successfully shot %f times per second \n(time elapsed %f s, success %d times, rate %d)\n", 
            Success/elapsed_time, elapsed_time, Success, rate);
    
    
};

class RogueFineCleaner
{
  public:
      RogueFineCleaner(int rate, Color color, Lanes* Gallery, int rounds);
      ~RogueFineCleaner(){};

  private:
      Color playercolor;
    
};

RogueFineCleaner::RogueFineCleaner(int rate, Color color, Lanes* Gallery, int rounds)
{
  printf("Fine_clean Rogue\n");
  // initialize the mutex locks for each lane
  static pthread_mutex_t mtx[MAXLANENUM] = PTHREAD_MUTEX_INITIALIZER;
  static pthread_mutex_t all_mtx = PTHREAD_MUTEX_INITIALIZER;
  static pthread_cond_t all_cond = PTHREAD_COND_INITIALIZER;

  playercolor = color;  

  // initialize random seed
  srand(time(NULL));

  int count = Gallery->Count();
  int laneNum;
  int Success = 0;

  struct timeval start, finish;

  gettimeofday(&start, 0);
  while(rounds_completed < rounds)
  {
      usleep(1000000/rate);
      
      // generate a random number from those lanes not colored
      laneNum = rand() % count;

       // check if all lanes are colored, if so, clean them
        int colored_lanes_cnt=0;
        for (int i=0;i<count;i++)
        {
          if(Gallery->Get(i)!=Color::white)
            colored_lanes_cnt++;
        }
        if(colored_lanes_cnt >= count)
        {
          pthread_mutex_lock(&all_mtx);
          if (Gallery->Get(1) != Color::white)
          {
            pthread_mutex_lock(&cln_mtx);
            pthread_cond_signal(&cln_cond);
            clean_command = true;
            // wait after cleaning working is done
            pthread_cond_wait(&cln_cond, &cln_mtx);
            pthread_mutex_unlock(&cln_mtx);
            pthread_mutex_unlock (&all_mtx);
            continue;
          }
          pthread_mutex_unlock (&all_mtx);
        }

        // find the next random lane that is not colored
        while(1)
        {
          if (Gallery->Get(laneNum) == Color::white)
            break;
          laneNum++;
          laneNum = laneNum % count;
        }

        //if(pthread_mutex_trylock(&mtx[laneNum]) == EBUSY)
        // continue;
        pthread_mutex_lock(&mtx[laneNum]);    //lock the lane
        //if(occupied[laneNum])
          //pthread_cond_wait(&cond[laneNum],&mtx[laneNum]);
        //occupied[laneNum] = true;
        if(Gallery->Get(laneNum) == Color::white)
        {
          Gallery->Set(laneNum,color);
          Success++;
        }
        //occupied[laneNum] = false;
        //pthread_cond_signal(&cond[laneNum]);
        pthread_mutex_unlock(&mtx[laneNum]);    //unlock the lane
  } 
  gettimeofday(&finish, 0);
  double elapsed_time = (finish.tv_sec - start.tv_sec) + (finish.tv_usec - start.tv_usec)*0.000001;

  pthread_mutex_lock(&all_mtx);
  printf("\n");
  if(playercolor == Color::red)
    printf("Red rogue successfully shot %f times per second \n(time elapsed %f s, success %d times, rate %d)\n", 
            Success/elapsed_time, elapsed_time, Success, rate);
  else
    printf("Blue rogue successfully shot %f times per second \n(time elapsed %f s, success %d times, rate %d)\n", 
            Success/elapsed_time, elapsed_time, Success, rate);
  pthread_mutex_unlock(&all_mtx);
}

class RogueTMCleaner
{
    public:
      RogueTMCleaner(int rate, Color color, Lanes* Gallery, int rounds);
      ~RogueTMCleaner(){};

    private:
      Color playercolor; 
};

RogueTMCleaner::RogueTMCleaner(int rate, Color color, Lanes* Gallery, int rounds)
{
  printf("TM_clean Rogue\n");
  static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
  playercolor = color;

  // initialize random seed
  srand(time(NULL));

  int count = Gallery->Count();
  int laneNum;
  int Success = 0;
  double Abort = 0.0;
  struct timeval start, finish;

  gettimeofday(&start, 0);
  while(rounds_completed < rounds)
  {
      int nretries = 0;
      int status;
      while (nretries < MAXRETRY)
      {
        usleep(1000000/rate);
        // generate a random number from those lanes not colored
        laneNum = rand() % count;

        // find the next random lane that is not colored
        int trytimes = 0;
        for (trytimes = 0; trytimes<count; trytimes++)
        {
          if(Gallery->Get(laneNum) != Color::white)
          {
            laneNum++;
            laneNum = laneNum % count;
          }
          else
            break;
        }

        if(trytimes == count)
        {
            // do cleaning
            pthread_mutex_lock(&mtx);
            if (Gallery->Get(1) != Color::white)
            {
              pthread_mutex_lock(&cln_mtx);
              pthread_cond_signal(&cln_cond);
              clean_command = true;
              // wait after cleaning working is done
              pthread_cond_wait(&cln_cond, &cln_mtx);
              pthread_mutex_unlock(&cln_mtx);
            }
            pthread_mutex_unlock(&mtx);
        }
        else
        {
            // do shooting
            if ((status = _xbegin()) == _XBEGIN_STARTED) 
            {
              Gallery->Set(laneNum, color);
              Success++;
              _xend ();
              break;
            }
            else
            {
              nretries++;
            }
        }            
      }
      if(nretries == MAXRETRY)
        Abort++;
            
  }   
  gettimeofday(&finish, 0);
  double elapsed_time = (finish.tv_sec - start.tv_sec) + (finish.tv_usec - start.tv_usec)*0.000001;
  printf("\n");
  if(playercolor == Color::red)
    printf("Red rogue successfully shot %f times per second \n(time elapsed %f s, success %d times, rate %d)\n", 
            Success/elapsed_time, elapsed_time, Success, rate);
  else
    printf("Blue rogue successfully shot %f times per second \n(time elapsed %f s, success %d times, rate %d)\n", 
            Success/elapsed_time, elapsed_time, Success, rate);
  printf("Abort rate: %f (abort %f times)\n", Abort / (Abort + Success), Abort);

}

#endif



