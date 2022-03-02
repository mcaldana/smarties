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
  Worker(ExecutionInfo& m_ExecutionInfoinfo);
  virtual ~Worker() {}

  void synchronizeEnvironments();

  void runTraining();
  void loopSocketsToMaster();

  // may be called from application:
  void stepWorkerToMaster(Agent & agent) const;

  void run(const environment_callback_t & callback);

protected:
  ExecutionInfo& m_ExecutionInfo;
  TaskQueue m_DataTasks, m_AlgoTasks;

  const std::unique_ptr<Launcher> m_Launcher;

  const MPI_Comm& m_MasterWorkersComm = m_ExecutionInfo.master_workers_comm;
  const MPI_Comm& m_WorkerlessMastersComm = m_ExecutionInfo.workerless_masters_comm;
  const MPI_Comm& m_LearnersTrainComm = m_ExecutionInfo.learners_train_comm;
  const MPI_Comm& m_EnvAppComm = m_ExecutionInfo.environment_app_comm;
  const int m_EnvMPIrank = MPICommRank(m_EnvAppComm);
  const int m_EnvMPIsize = MPICommSize(m_EnvAppComm);

  std::vector<std::unique_ptr<Learner>> m_Learners;

  Environment& m_Environment;
  const std::vector<std::unique_ptr<Agent>>& m_Agents;

  const uint64_t m_nCallingEnvs = m_ExecutionInfo.nOwnedEnvironments;
  const int m_bTrain = m_ExecutionInfo.bTrain;

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
