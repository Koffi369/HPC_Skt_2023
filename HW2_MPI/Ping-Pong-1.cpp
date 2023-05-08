/*1) Ping-Pong:  

1. Let's imagine that processes are human beings and they wanted to play ping-pong to get acquainted with each other. 
Processor 0 starts the play. It randomly passes a ball to 'i' and says its name.

Processor 'i' passes a ball to another guy 'j' saying all previous names in the mentioned order and says its own name. 
Passing occurs randomly; it means only processor 'i' knows who will be next, others don't know and they always need to 
be on the lookout. The game ends after N passes. Use the synchronous mode MPI_Ssend to send the data.*/

#include <functional>
#include <memory>
#include <mpi.h>
#include <random>
#include <stdexcept>
#include <vector>
//#include <iostream>
//#include <cstdlib>
//#include <ctime>


struct PingPongBall {
public:
  PingPongBall(int num_passes) {
    ball_data.resize(num_passes + 2);// Resize the vector to hold the number of passes plus two
    ball_data[0] = num_passes; // Total number of passes
    ball_data[1] = 0;          // current number of passes
  }
  size_t GetSize() const { // Get the size of the vector
    return ball_data.size(); 
    } 
  int &operator[](size_t i) { return ball_data[i]; } // Get the number of passes at index i
  void PrintPassedData() const { // Print the ranks of the processes that have passed the ball so far
    printf("List of processes that have passed the ball so far: ");
    for (int i = 2; i <= 1 + ball_data[1]; i++) {
      printf("%d, ", ball_data[i]);
    }
  }

private:
  std::vector<int> ball_data; // Vector to hold the data of the ball
};


int Random_Receiver_Generator(int rank, int size) {
  std::random_device rand_; // unique random number generator
  std::mt19937 engine(rand_()); // sequence engine based on Mersenne Twister algorithm
  std::vector<int> Vec(size, 1); // initialize vector with equal weights
  Vec[rank] = 0; // zero out the current index to prevent selecting it
  std::discrete_distribution<> dist{Vec.begin(), Vec.end()}; // discrete distribution with the given weights
  auto r = std::bind(dist, std::ref(engine)); // function object which generates random numbers based on the distribution
  int selectedReceiverIndex = r(); // invoke the function object to get a random index for the receiver
  return selectedReceiverIndex; 
}

/*

using namespace std;

vector<int> ball(10); // data structure to hold the ball

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(time(nullptr) + rank);

    if (size < 2) {
        cerr << "Error: the number of processes should be at least 2" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        // start the game by sending the ball to a random process
        int receiver = rand() % (size - 1) + 1;
        ball[0] = 1; // the number of passes
        ball[1] = 0; // the index of the last player
        ball[2] = rank; // the index of the current player
        cout << "I am " << rank << ", and I am starting the game by sending the ball to " << receiver << endl;
        MPI_Ssend(&ball[0], 10, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    } else {
        while (true) {
            MPI_Recv(&ball[0], 10, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (ball[0] == 1) {
                // the ball has reached its first receiver
                int sender = ball[2];
                int last_player = ball[1];
                cout << "I am " << rank << ", and I have just received the ball from " << sender << endl;
                if (ball[1] == rank) {
                    // the ball has reached the starting process after completing N passes
                    cout << "The game is over!" << endl;
                    ball[0] = -1; // to signal the end of the game
                    MPI_Ssend(&ball[0], 10, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD); // pass the ball to the next player
                    break;
                } else {
                    // continue the game by passing the ball to a random receiver
                    int receiver = rand() % (size - 2);
                    if (receiver >= rank) {
                        receiver++;
                    }
                    ball[1] = rank;
                    ball[2] = last_player;
                    cout << "I am " << rank << ", and I am passing the ball to " << receiver << endl;
                    MPI_Ssend(&ball[0], 10, MPI_INT, receiver, 0, MPI_COMM_WORLD);
                }
            } else if (ball[0] == -1) {
                // the game is over
                MPI_Ssend(&ball[0], 10, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD); // pass the ball to the next player
                break;
            } else {
                // the ball is being passed from one player to another
                int sender = ball[2];
                int last_player = ball[1];
                ball[0]--;
                cout << "I am " << rank << ", and I have just received the ball from " << sender << endl;
                if (ball[0] == 1) {
                    // the ball has reached its last receiver
                    ball[1] = rank;
                    ball[2] = last_player;
                    cout << "I am " << rank << ", and I am passing the ball back to the starting player" << endl;
                } else {
                    // continue the game by passing the ball to a random receiver
                    int receiver = rand() % (size - 2);
                    if (receiver >= rank) {
                        receiver++;
                    }
                    cout << "I am " << rank << ", and I am passing the ball to " << receiver << endl;
                }
                MPI_Ssend(&ball[0], 10, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD); // pass the ball to the next player
            }
        }
    }

    MPI_Finalize();
    return 0;
}

*/

////////////////


struct PassProcess {
public:
  /// Constructor initializes the ball parameter and gets the rank and size of the process.
  /// Throws an error if there are less than 2 processes.
  PassProcess(PingPongBall &ping_pong_ball, bool _iAmGameStarter = false)
      : ball(ping_pong_ball), _iAmGameStarter(_iAmGameStarter) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
      throw std::runtime_error("Number of players should be greather than 2");
    }
  }

  /// Main method to start playing the game of pass using MPI send and receive.
  void Play() {
    if (_iAmGameStarter) {
      // If this process is the game starter and enough passes have been made, end the game.
      if (IsSufficientPasses()) {
        return;
      }
      printf("Process 0 start the game \n");
      SetPass();
      MPI_Ssend(&ball[0], ball.GetSize(), MPI_INT, target, 0, MPI_COMM_WORLD);
    }
    while (true) {
      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &mpistatus);
      MPI_Recv(&ball[0], ball.GetSize(), MPI_INT, mpistatus.MPI_SOURCE, 0, MPI_COMM_WORLD, &mpistatus);

      // If enough passes have been made, end the game.
      if (IsSufficientPasses()) {
        break;
      }
      
      ball.PrintPassedData();
      printf("Process rank: %d\n", rank);
      SetPass();
      MPI_Ssend(&ball[0], ball.GetSize(), MPI_INT, target, 0, MPI_COMM_WORLD);
    }
  }

private:
  /// Method to check if the game has ended.
  bool IsSufficientPasses() {
    if (ball[0] == -1) {
      return true;
    }
    if (ball[0] == ball[1]) {
      printf("Game Over\n");
      ball.PrintPassedData();
      printf("Process rank: %d\n", rank);
      ball[0] = -1;
      // Send the final message to end the game.
      for (int i = 0; i < size; ++i) {
        if (i != rank) {
          MPI_Send(&ball[0], ball.GetSize(), MPI_INT, i, 0, MPI_COMM_WORLD);
        }
      }
      return true;
    }
    return false;
  }

  /// Method to prepare the ball for the next pass.
  void SetPass() {
    ball[1]++;
    ball[ball[1] + 1] = rank;
    target = Random_Receiver_Generator(rank, size);
    printf("For pass number: %d , process %d send the ball  to %d\n", ball[1], rank, target);
  }

private:
  int size;
  int rank;
  int target;
  bool _iAmGameStarter{false};
  PingPongBall &ball;
  MPI_Status mpistatus;
};




// Function to initialize the process based on the process rank
auto MakeUniquePlayer(PingPongBall &ping_pong_ball, int rank) {
  if (rank == 0) {
    // If the process rank is zero, the process is the game starter
    return std::unique_ptr<PassProcess>(new PassProcess(ping_pong_ball, true));
  } else {
    // If the process rank is not zero, the process plays as a regular player
    return std::unique_ptr<PassProcess>(new PassProcess(ping_pong_ball));
  }
}

int main(int argc, char **argv) {
  // Total number of passes
  int num_passes = 10;
  // Initialize MPI and get the process rank
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // Create a ball object with the given number of passes
  PingPongBall ping_pong_ball(num_passes);

  // Initialize the current process and run the game
  auto currentprocess = MakeUniquePlayer(ping_pong_ball, rank);
  currentprocess->Play();

  // Finalize MPI
  MPI_Finalize();
}








