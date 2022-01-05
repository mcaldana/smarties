#include "cart_pole_simple.h"
#include "smarties.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>

std::ofstream myfile;

inline void cartpole_simple_app(smarties::Communicator* const comm,
                                int argc,
                                char** argv) {
  const int control_vars = 1;  // force along x
  const int state_vars = 4;    // x, vel, angvel, angle
  comm->setStateActionDims(state_vars, control_vars);

  CartPoleSimple env;

  while (true) {                 // train loop
    env.reset(comm->getPRNG());  // prng with different seed on each process
    comm->sendInitState(env.state());  // send initial state

    while (true) {  // simulation loop
      std::vector<double> action = comm->recvAction();
      if (comm->terminateTraining())
        return;  // exit program

      const bool done = env.step(action.data());  // advance the simulation:
      const auto state = env.state();
      const auto reward = env.reward();

      if (done) {  // tell smarties that this is a terminal state
        comm->sendTermState(state, reward);
        break;
      } else
        comm->sendState(state, reward);
    }

    using namespace std::chrono;
    myfile << duration_cast<milliseconds>(
                  system_clock::now().time_since_epoch())
                  .count()
           << "," << env.numSteps << "\n";
  }
}

int main(int argc, char** argv) {
  smarties::Engine e(argc, argv);
  if (e.parse())
    return 1;
  myfile.open("timeVsEnvSteps.txt");
  e.runArg(cartpole_simple_app);
  return 0;
}
