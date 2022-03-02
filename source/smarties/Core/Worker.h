//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Worker_h
#define smarties_Worker_h

#include "../Utils/ParameterBlob.h"
#include "../Utils/TaskQueue.h"
#include "Environment.h"
#include "Launcher.h"
#include "../Learners/Learner.h"
#include "../Settings/ExecutionInfo.h"
#include <thread>

namespace smarties
{

class Worker
{
public:
  Worker(ExecutionInfo& distribinfo);
  virtual ~Worker() {}

  void synchronizeEnvironments();

  void runTraining();
  void loopSocketsToMaster();

  // may be called from application:
  void stepWorkerToMaster(Agent & agent) const;

  void run(const environment_callback_t & callback);

protected:
  ExecutionInfo& distrib;
  TaskQueue dataTasks, algoTasks;

  const std::unique_ptr<Launcher> COMM;

  const MPI_Comm& master_workers_comm = distrib.master_workers_comm;
  const MPI_Comm& workerless_masters_comm = distrib.workerless_masters_comm;
  const MPI_Comm& learners_train_comm = distrib.learners_train_comm;
  const MPI_Comm& envAppComm = distrib.environment_app_comm;
  const int envMPIrank = MPICommRank(envAppComm);
  const int envMPIsize = MPICommSize(envAppComm);

  std::vector<std::unique_ptr<Learner>> learners;

  Environment& m_Environment;
  const std::vector<std::unique_ptr<Agent>>& agents;

  const uint64_t nCallingEnvs = distrib.nOwnedEnvironments;
  const int bTrain = distrib.bTrain;

  // small utility functions:
  uint64_t getLearnerID(const uint64_t agentIDlocal) const;
  bool learnersBlockingDataAcquisition() const;

  void answerStateActionCaller(const int bufferID);

  void stepWorkerToMaster(const uint64_t bufferID) const;

  void answerStateAction(const uint64_t bufferID) const;
  void answerStateAction(Agent& agent) const;

  void sendStateRecvAction(const COMM_buffer& BUF) const;

  int getSocketID(const uint64_t worker) const;
  const COMM_buffer& getCommBuffer(const uint64_t worker) const;
};

} // end namespace smarties
#endif // smarties_Worker_h
