//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Engine.h"
#include "Core/Master.h"

namespace smarties
{

Engine::Engine(int argc, char** argv) :
  m_ExecutionInfo(new ExecutionInfo(argc, argv)) { }

Engine::Engine(std::vector<std::string> args) :
  m_ExecutionInfo(new ExecutionInfo(args)) { }

Engine::Engine(MPI_Comm world, int argc, char** argv) :
  m_ExecutionInfo(new ExecutionInfo(world, argc, argv)) {}

Engine::~Engine() {
  assert(m_ExecutionInfo);
  delete m_ExecutionInfo;
}

int Engine::parse() {
  return m_ExecutionInfo->parse();
}

void Engine::setNthreads(const uint64_t nThreads) {
  m_ExecutionInfo->nThreads = nThreads;
}

void Engine::setNmasters(const uint64_t nMasters) {
  m_ExecutionInfo->nMasters = nMasters;
}

void Engine::setNenvironments(const uint64_t nEnvironments) {
  m_ExecutionInfo->nEnvironments = nEnvironments;
}

void Engine::setNworkersPerEnvironment(const uint64_t workerProcessesPerEnv) {
  m_ExecutionInfo->workerProcessesPerEnv = workerProcessesPerEnv;
}

void Engine::setRandSeed(const uint64_t randSeed) {
  m_ExecutionInfo->randSeed = randSeed;
}

void Engine::setNumTrainingTimeSteps(const uint64_t numSteps) {
  m_ExecutionInfo->nTrainSteps = numSteps;
  m_ExecutionInfo->bTrain = 1;
}

void Engine::setNumEvaluationEpisodes(const uint64_t numEpisodes) {
  m_ExecutionInfo->nEvalEpisodes = numEpisodes;
  m_ExecutionInfo->bTrain = 0;
}

void Engine::setSimulationArgumentsFilePath(const std::string& appSettings) {
  m_ExecutionInfo->appSettings = appSettings;
}

void Engine::setSimulationSetupFolderPath(const std::string& setupFolder) {
  m_ExecutionInfo->setupFolder = setupFolder;
}

void Engine::setRestartFolderPath(const std::string& restart) {
  m_ExecutionInfo->restart = restart;
}

void Engine::setIsLoggingAllData(const int logAllSamples) {
  m_ExecutionInfo->logAllSamples = logAllSamples;
}

void Engine::setAreLearnersOnWorkers(const bool learnersOnWorkers) {
  m_ExecutionInfo->learnersOnWorkers = learnersOnWorkers;
}

void Engine::setRedirectAppScreenOutput(const bool redirect) {
  m_ExecutionInfo->redirectAppStdoutToFile = redirect;
}

void Engine::init()
{
  m_ExecutionInfo->initialze();
  m_ExecutionInfo->figureOutWorkersPattern();

  if( (!m_ExecutionInfo->bTrain) && m_ExecutionInfo->restart == "none") {
   printf("Did not specify path for restart files, assumed current dir.\n");
   m_ExecutionInfo->restart = ".";
  }

  MPI_Barrier(m_ExecutionInfo->world_comm);
}

void Engine::run(const std::function<void(Communicator*const)> & callback)
{
  assert(m_ExecutionInfo->workerProcessesPerEnv <= 1);

  const environment_callback_t fullcallback = [&](
    Communicator*const sc, const MPI_Comm mc, int argc, char**argv) {
    return callback(sc);
  };

  run(fullcallback);
}

void Engine::run(const std::function<void(Communicator*const,
                                          int, char **      )> & callback)
{
  assert(m_ExecutionInfo->workerProcessesPerEnv <= 1);

  const environment_callback_t fullcallback = [&](
    Communicator*const sc, const MPI_Comm mc, int argc, char**argv) {
    return callback(sc, argc, argv);
  };

  run(fullcallback);
}

void Engine::run(const std::function<void(Communicator*const,
                                          MPI_Comm          )> & callback)
{
  const environment_callback_t fullcallback = [&](
    Communicator*const sc, const MPI_Comm mc, int argc, char**argv) {
    return callback(sc, mc);
  };

  run(fullcallback);
}

void Engine::run(const std::function<void(Communicator*const,
                                          MPI_Comm,
                                          int, char **      )> & callback)
{
  m_ExecutionInfo->forkableApplication = m_ExecutionInfo->workerProcessesPerEnv <= 1;
  init();
  if(m_ExecutionInfo->bIsMaster)
  {
    if(m_ExecutionInfo->nForkedProcesses2spawn > 0) {
      MasterSockets process(*m_ExecutionInfo);
      process.run(callback);
    } else {
      MasterMPI process(*m_ExecutionInfo);
      process.run();
    }
  }
  else
  {
    Worker process(*m_ExecutionInfo);
    process.run(callback);
  }
}

}
