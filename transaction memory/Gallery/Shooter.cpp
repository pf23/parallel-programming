#include <iostream>
#include <thread>
#include <unistd.h>
#include "Lanes.h"
#include <vector>
#include <cstring> 
#include "Rogue.h"

Lanes* Gallery;
int nlanes;
using namespace std;
extern int rounds_completed;

int RedRate = 1000;
int BlueRate = 1000;
int Round = 10;

//
//***********************************
//** Change class you would use here
//***********************************
// Optional: coarse | coarse2 | coarse_clean | fine | fine2  | fine_clean  | tm  | tm2 | tm_clean
//
char *class_name = "coarse";
bool print_closed = false;

void ShooterAction(int rate,Color PlayerColor, int rounds) {

    /**
     *  Needs synchronization. Between Red and Blue shooters.
     *  Choose a random lane and shoot.
     *  Rate: Choose a random lane every 1/rate s.
     *  PlayerColor : Red/Blue.
     */
    if(!strcmp(class_name, "coarse"))
        RogueCoarse rogue_coarse(rate, PlayerColor, Gallery, rounds);
    else if(!strcmp(class_name, "coarse2"))
         RogueCoarse2 rogue_coarse2(rate, PlayerColor, Gallery, rounds);
    else if(!strcmp(class_name, "coarse_clean"))
        RogueCoarseCleaner rogue_coarse_cleaner(rate, PlayerColor, Gallery, rounds);
    else if(!strcmp(class_name, "fine"))
        RogueFine rogue_fine(rate, PlayerColor, Gallery, rounds);
    else if(!strcmp(class_name, "fine2"))
        RogueFine2 rogue_fine2(rate, PlayerColor, Gallery, rounds);
    else if(!strcmp(class_name, "fine_clean"))
        RogueFineCleaner rogue_fine_cleaner(rate, PlayerColor, Gallery, rounds);
    else if(!strcmp(class_name, "tm"))
        RogueTM rogue_tm(rate, PlayerColor, Gallery, rounds);
    else if(!strcmp(class_name, "tm2"))
        RogueTM2 rogue_tm2(rate, PlayerColor, Gallery, rounds);
    else if(!strcmp(class_name, "tm_clean"))
        RogueTMCleaner rogue_tm_cleaner(rate, PlayerColor, Gallery, rounds);
    else {
            printf("Wrong Class Name Input.\n");
            printf("[class_name]: coarse | coarse2 | coarse_clean |\n");
            printf("              fine   | fine2   | fine_clean   |\n");
            printf("              tm     | tm2     | tm_clean     |\n");
    }
    
}


void Cleaner(int rounds) {

    /**
     *  Cleans up lanes. Needs to synchronize with shooter.
     *  Use a monitor
     *  Should be in action only when all lanes are shot.
     *  Once all lanes are shot. Cleaner starts up.
     *  Once cleaner starts up shooters wait for cleaner to finish.
     */
    
    // implemented in Rogue.h classes
    if(!strcmp(class_name, "coarse_clean") || !strcmp(class_name, "fine_clean") || !strcmp(class_name, "tm_clean"))
        RogueCleaner cleaner(Gallery, rounds);
}


void Printer(int rate, int rounds) {

    /**
     *  NOT TRANSACTION SAFE; cannot be called inside a transaction. Possible conflict with other Txs Performs I/O
     *  Print lanes periodically rate/s.
     *  If it doesn't synchronize then possible mutual inconsistency between adjacent lanes.
     *  Not a particular concern if we don't shoot two lanes at the same time.
     *
     */
    
   while(rounds_completed < rounds)
   {
       sleep(1/rate);
       Gallery->Print();
       cout << Gallery->Count() << std::endl;
   }

}

int main(int argc, char** argv)
{

    std::vector<thread> ths;
    
    Gallery = new Lanes(8);
    Gallery->Clear();
    // std::thread RedShooterT,BlueShooterT,CleanerT,PrinterT;
    
    int c;
    while ( (c = getopt(argc, argv, "r:b:n:m:p")) != -1)
    {
        switch (c) {
                
            case 'r':
                RedRate = atoi(optarg);
                break;
                
            case 'b':
                BlueRate = atoi(optarg);
                break;
                
            case 'n':
                Round = atoi(optarg);
                break;
            case 'm':
                class_name = optarg;
                break;
            case 'p':
                print_closed = true;
                break;

            case '?':
                /* getopt_long already printed an error message. */
                printf("format: ./Shooter -r [red_rate] -b [blue_rate] -n [rounds] -m [class_name] -p\n");
                printf("[class_name]: coarse | coarse2 | coarse_clean |\n");
                printf("              fine   | fine2   | fine_clean   |\n");
                printf("              tm     | tm2     | tm_clean     |\n");
                exit(1);
                break;
                
            default:
                exit(1);
        }
    }
    cout << "Red Rate: " << RedRate << endl;
    cout << "Blue Rate: " << BlueRate << endl;
    cout << "Round: " << Round << endl;
    
    ths.push_back(std::thread(&ShooterAction,RedRate,red, Round));
    ths.push_back(std::thread(&ShooterAction,BlueRate,blue, Round));
    ths.push_back(std::thread(&Cleaner, Round));
    //ths.push_back(std::thread(&Printer,1, Round));


    // Join with threads
    // RedShooterT.join();
    // BlueShooterT.join();
    // CleanerT.join();
    // PrinterT.join();

    for (auto& th : ths) {

        th.join();

    }


    return 0;
}
