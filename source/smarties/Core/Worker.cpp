//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Worker.h"

#include "../Learners/AlgoFactory.h"
#include "../Utils/SocketsLib.h"
#include "../Utils/SstreamUtilities.h"

#include <fstream>

namespace smarties
{

Worker::Worker(ExecutionInfo&D) : m_ExecutionInfo(D),
  m_DataTasks( [&]() { return learnersBlockingDataAcquisition(); } ),
  m_AlgoTasks( [&]() { return learnersBlockingDataAcquisition(); } ),
  m_Launcher( std::make_unique<Launcher>(this, D) ),
  m_Environment( m_Launcher->m_Environment ), m_Agents( m_Environment.agents )
{}

void Worker::run(const environment_callback_t & callback)
{
  if( m_ExecutionInfo.nForkedProcesses2spawn > 0 )
  {
    const bool isChild = m_Launcher->forkApplication(callback);
    if(not isChild) {
      synchronizeEnvironments();
      loopSocketsToMaster();
    }
  }
  else m_Launcher->runApplication( callback );
}

void Worker::runTraining()
{

  const int learn_rank = MPICommRank(m_LearnersTrainComm);
  //////////////////////////////////////////////////////////////////////////////
  ////// FIRST SETUP SIMPLE FUNCTIONS TO DETECT START AND END OF TRAINING //////
  //////////////////////////////////////////////////////////////////////////////
  long minNdataB4Train = m_Learners[0]->nObsB4StartTraining;
  int firstLearnerStart = 0, isTrainingStarted = 0, percentageReady = -5;
  for(uint64_t i=1; i<m_Learners.size(); ++i)
    if(m_Learners[i]->nObsB4StartTraining < minNdataB4Train) {
      minNdataB4Train = m_Learners[i]->nObsB4StartTraining;
      firstLearnerStart = i;
    }

  const std::function<bool()> isOverTraining = [&] ()
  {
    if(isTrainingStarted==0 && learn_rank==0) {
      const auto nCollected = m_Learners[firstLearnerStart]->locDataSetSize();
      const int perc = nCollected * 100.0/(Real) minNdataB4Train;
      if(nCollected >= minNdataB4Train) {
        isTrainingStarted = 1;
        printf("\rCollected all data required to begin training.     \n");
        fflush(0);
      } else if(perc >= percentageReady+5) {
        percentageReady = perc;
        printf("\rCollected %d%% of data required to begin training. ", perc);
        fflush(0);
      }
    }
    if(isTrainingStarted==0) return false;

    bool over = true;
    const Real factor = m_Learners.size()==1? 1.0/m_Environment.nAgentsPerEnvironment : 1;
    for(const auto& L : m_Learners)
      over = over && L->nLocTimeStepsTrain() * factor >= m_ExecutionInfo.nTrainSteps;
    return over;
  };
  const std::function<bool()> isOverTesting = [&] ()
  {
    // if agents share learning algo, return number of turns performed by env
    // instead of sum of timesteps performed by each agent
    const long factor = m_Learners.size()==1? m_Environment.nAgentsPerEnvironment : 1;
    long nEnvSeqs = std::numeric_limits<long>::max();
    for(const auto& L : m_Learners)
      nEnvSeqs = std::min(nEnvSeqs, L->nSeqsEval() / factor);
    const Real perc = 100.0 * nEnvSeqs / (Real) m_ExecutionInfo.nEvalEpisodes;
    if(nEnvSeqs >= (long) m_ExecutionInfo.nEvalEpisodes) {
      printf("\rFinished collecting %d environment episodes (option " \
        "--nEvalEpisodes) to evaluate restarted policies.\n", (int) nEnvSeqs);
      return true;
    } else if(perc >= percentageReady+5) {
      percentageReady = perc;
      printf("\rCollected %d environment episodes out of %u to evaluate " \
        " restarted policies.", (int)nEnvSeqs, (unsigned)m_ExecutionInfo.nEvalEpisodes);
      fflush(0);
    }
    return false;
  };

  const auto isOver = m_bTrain? isOverTraining : isOverTesting;

  //////////////////////////////////////////////////////////////////////////////
  /////////////////////////// START DATA COLLECTION ////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  std::atomic<uint64_t> bDataCoordRunning {1};
  std::thread dataCoordProcess;

  #pragma omp parallel
  if( omp_get_thread_num() == std::min(omp_get_num_threads()-1, 2) )
    dataCoordProcess = std::thread( [&] () {
      while(1) {
        m_DataTasks.run();
        if (bDataCoordRunning == 0) break;
        usleep(1); // wait for workers to send data without burning a cpu
      }
    } );

  //////////////////////////////////////////////////////////////////////////////
  /////////////////////////////// TRAINING LOOP ////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  while(1) {
    m_AlgoTasks.run();
    if ( isOver() ) break;
  }

  // kill data gathering process
  bDataCoordRunning = 0;
  dataCoordProcess.join();
}

void Worker::answerStateAction(Agent& agent) const
{
  if(agent.agentStatus == FAIL) die("app crashed. TODO: handle");
  // get learning algorithm:
  Learner& algo = * m_Learners[getLearnerID(agent.localID)].get();
  //pick next action and ...do a bunch of other stuff with the data:
  algo.select(agent);
  #ifndef NDEBUG
    if (agent.stateIsInvalid())
      die("Learning algorithm picked a nan or inf action.");
  #endif
  //static constexpr auto vec2str = Utilities::vec2string<double>;
  //const int agentStatus = status2int(agent.agentStatus);
  //_warn("Agent %d %d:[%s]>[%s] r:%f a:[%s]", agent.ID, agentStatus,
  //      vec2str(agent.sOld,-1).c_str(), vec2str(agent.state,-1).c_str(),
  //      agent.reward, vec2str(agent.action,-1).c_str());

  // Some logging and passing around of step id:
  const Real factor = m_Learners.size()==1? 1.0/m_Environment.nAgentsPerEnvironment : 1;
  const uint64_t nSteps = std::max(algo.nLocTimeSteps(), (long) 0);
  agent.learnerTimeStepID = factor * nSteps;
  agent.learnerGradStepID = algo.nGradSteps();
  if(agent.agentStatus >= LAST) agent.action[0] = algo.getAvgCumulativeReward();
  //debugS("Sent action to worker %d: [%s]", worker, print(actVec).c_str() );
}

void Worker::answerStateAction(const uint64_t bufferID) const
{
  assert( (uint64_t) bufferID < m_Launcher->m_CommBuffers.size());
  const COMM_buffer& buffer = getCommBuffer(bufferID+1);
  const unsigned localAgentID = Agent::getMessageAgentID(buffer.dataStateBuf);
  // compute agent's ID within worker from the agentid within environment:
  const int agentID = bufferID * m_Environment.nAgentsPerEnvironment + localAgentID;
  //read from worker's buffer:
  assert( (uint64_t) agentID < m_Agents.size() );
  assert( (uint64_t) m_Agents[agentID]->workerID == bufferID );
  assert( (uint64_t) m_Agents[agentID]->localID == localAgentID );
  Agent& agent = * m_Agents[agentID].get();
  // unpack state onto agent
  agent.unpackStateMsg(buffer.dataStateBuf);
  answerStateAction(agent);
  agent.packActionMsg(buffer.dataActionBuf);
}

uint64_t Worker::getLearnerID(const uint64_t agentIDlocal) const
{
  // some asserts:
  // 1) agentID within environment must match what we know about environment
  // 2) either only one learner or ID of agent in m_Environment must match a learner
  // 3) if i have more than one learner, then i have one per agent in env
  assert(agentIDlocal < m_Environment.nAgentsPerEnvironment);
  assert(m_Learners.size() == 1 || agentIDlocal < m_Learners.size());
  if(m_Learners.size()>1)
    assert(m_Learners.size() == (size_t) m_Environment.nAgentsPerEnvironment);
  // if one learner, return learnerID=0, else learnID == ID of agent in m_Environment
  return m_Learners.size()>1? agentIDlocal : 0;
}

bool Worker::learnersBlockingDataAcquisition() const
{
  //When would a learning algo stop acquiring more data?
  //Off Policy algos:
  // - User specifies a ratio of observed trajectories to gradient steps.
  //    Comm is restarted or paused to maintain this ratio consant.
  //On Policy algos:
  // - if collected enough trajectories for current batch, then comm is paused
  //    untill gradient is applied (or nepocs are done), then comm restarts
  //    to obtain fresh on policy samples
  // However, no learner can stop others from getting data (vector of algos)
  bool lock = true;
  for (const auto& L : m_Learners) lock = lock && L->blockDataAcquisition();
  return lock;
}

void Worker::synchronizeEnvironments()
{
  // here cannot use the recurring template because behavior changes slightly:
  const std::function<void(void*, size_t)> recvBuffer =
    [&](void* buffer, size_t size)
  {
    assert(size>0);
    bool received = false;
    if( m_Launcher->m_Sockets.clients.size() > 0 ) { // master with apps connected through sockets (on the same compute node)
      SOCKET_Brecv(buffer, size, m_Launcher->m_Sockets.clients[0]);
      received = true;
      for(size_t i=1; i < m_Launcher->m_Sockets.clients.size(); ++i) {
        void * const testbuf = malloc(size);
        SOCKET_Brecv(testbuf, size, m_Launcher->m_Sockets.clients[i]);
        const int err = memcmp(testbuf, buffer, size); free(testbuf);
        if(err) die(" error: comm mismatch");
      }
    }

    if(m_MasterWorkersComm != MPI_COMM_NULL)
    if( MPICommSize(m_MasterWorkersComm) >  1 &&
        MPICommRank(m_MasterWorkersComm) == 0 ) {
      if(received) die("Sockets and MPI workers: should be impossible");
      MPI_Recv(buffer, size, MPI_BYTE, 1, 368637, m_MasterWorkersComm, MPI_STATUS_IGNORE);
      received = true;
      // size of comm is number of workers plus master:
      for(uint64_t i=2; i < MPICommSize(m_MasterWorkersComm); ++i) {
        void * const testbuf = malloc(size);
        MPI_Recv(testbuf, size, MPI_BYTE, i, 368637, m_MasterWorkersComm, MPI_STATUS_IGNORE);
        const int err = memcmp(testbuf, buffer, size); free(testbuf);
        if(err) die(" error: mismatch");
      }
    }

    if(m_MasterWorkersComm != MPI_COMM_NULL)
    if( MPICommSize(m_MasterWorkersComm) >  1 &&
        MPICommRank(m_MasterWorkersComm) >  0 ) {
      MPI_Send(buffer, size, MPI_BYTE, 0, 368637, m_MasterWorkersComm);
    }

    if(m_WorkerlessMastersComm == MPI_COMM_NULL) return;
    assert( MPICommSize(m_WorkerlessMastersComm) >  1 );

    if( MPICommRank(m_WorkerlessMastersComm) == 0 ) {
      if(not received) die("rank 0 of workerless masters comm has no worker");
      for(uint64_t i=1; i < MPICommSize(m_WorkerlessMastersComm); ++i)
        MPI_Send(buffer, size, MPI_BYTE, i, 368637, m_WorkerlessMastersComm);
    }

    if( MPICommRank(m_WorkerlessMastersComm) >  0 ) {
      if(received) die("rank >0 of workerless masters comm owns workers");
      MPI_Recv(buffer, size, MPI_BYTE, 0, 368637, m_WorkerlessMastersComm, MPI_STATUS_IGNORE);
    }
  };

  m_Environment.synchronizeEnvironments(recvBuffer, m_ExecutionInfo.nOwnedEnvironments);

  for(uint64_t i=0; i<m_Environment.nAgents; ++i) {
    m_Launcher->initOneCommunicationBuffer();
    m_Agents[i]->initializeActionSampling( m_ExecutionInfo.generators[0] );
  }
  m_ExecutionInfo.nAgents = m_Environment.nAgents;

  // return if this process should not host the learning algorithms
  if(not m_ExecutionInfo.bIsMaster and not m_ExecutionInfo.learnersOnWorkers) return;

  const uint64_t nAlgorithms =
    m_Environment.bAgentsHaveSeparateMDPdescriptors? m_Environment.nAgentsPerEnvironment : 1;
  m_ExecutionInfo.nOwnedAgentsPerAlgo =
    m_ExecutionInfo.nOwnedEnvironments * m_Environment.nAgentsPerEnvironment / nAlgorithms;
  m_Learners.reserve(nAlgorithms);
  for(uint64_t i = 0; i<nAlgorithms; ++i)
  {
    m_Learners.emplace_back( createLearner(i, m_Environment.getDescriptor(i), m_ExecutionInfo) );
    assert(m_Learners.size() == i+1);
    m_Learners[i]->restart();
    m_Learners[i]->setupTasks(m_AlgoTasks);
    m_Learners[i]->setupDataCollectionTasks(m_DataTasks);
  }
}

void Worker::loopSocketsToMaster()
{
  const size_t nClients = m_Launcher->m_Sockets.clients.size();
  std::vector<SOCKET_REQ> reqs = std::vector<SOCKET_REQ>(nClients);
  // worker's communication functions behave following mpi indexing
  // sockets's rank (bufferID) is its index plus 1 (master)
  for(size_t i=0; i<nClients; ++i) {
    const auto& B = getCommBuffer(i+1); const int SID = getSocketID(i+1);
    SOCKET_Irecv(B.dataStateBuf, B.sizeStateMsg, SID, reqs[i]);
  }

  const auto sendKillMsgs = [&] (const int clientJustRecvd)
  {
    for(size_t i=0; i<nClients; ++i) {
      if( (int) i == clientJustRecvd ) {
        assert(reqs[i].todo == 0);
        continue;
      } else assert(reqs[i].todo != 0);
      SOCKET_Wait(reqs[i]);
    }
    // now all requests are completed and waiting to recv an 'action': terminate
    for(size_t i=0; i<nClients; ++i) {
      const auto& B = getCommBuffer(i+1);
      Agent::messageLearnerStatus((char*) B.dataActionBuf) = KILL;
      SOCKET_Bsend(B.dataActionBuf, B.sizeActionMsg, getSocketID(i+1));
    }
  };

  for(size_t i=0; ; ++i) // infinite loop : communicate until break command
  {
    int completed = 0;
    const int workID = i % nClients, SID = getSocketID(workID+1);
    const COMM_buffer& B = getCommBuffer(workID+1);
    SOCKET_Test(completed, reqs[workID]);

    if(completed) {
      stepWorkerToMaster(workID);
      learnerStatus& S = Agent::messageLearnerStatus((char*) B.dataActionBuf);
      if(S == KILL) { // check if abort was called
        sendKillMsgs(workID);
        return;
      }
      SOCKET_Bsend(B.dataActionBuf, B.sizeActionMsg, SID);
    }
    else usleep(1); // wait for app to send a state without burning a cpu
  }
}

void Worker::stepWorkerToMaster(Agent & agent) const
{
  assert(MPICommRank(m_MasterWorkersComm) > 0 || m_Learners.size()>0);
  const COMM_buffer& BUF = *m_Launcher->m_CommBuffers[agent.ID].get();

  if (m_EnvMPIrank<=0 || !m_Launcher->m_bEnvDistributedAgents)
  {
    if(m_Learners.size()) // then episode/parameter communication loop
    {
      answerStateAction(agent);
      // pack action in mpi buffer for bcast if m_ExecutionInfouted agents
      if(m_EnvMPIsize) agent.packActionMsg(BUF.dataActionBuf);
    }
    else                // then state/action comm loop from worker to master
    {
      if(m_EnvMPIsize) assert( m_Launcher->m_Sockets.clients.size() == 0 );

      agent.packStateMsg(BUF.dataStateBuf);
      sendStateRecvAction(BUF); //up to here everything is written on the buffer
      agent.unpackActionMsg(BUF.dataActionBuf); // copy action onto agent
    }

    //m_ExecutionInfouted agents means that each agent exists on multiple computational
    //processes (i.e. it is m_ExecutionInfourted) therefore actions must be communicated
    if (m_Launcher->m_bEnvDistributedAgents && m_EnvMPIsize>1) {
      //Then this is rank 0 of an environment with centralized agents.
      //Broadcast same action to members of the gang:
      MPI_Bcast(BUF.dataActionBuf, BUF.sizeActionMsg, MPI_BYTE, 0, m_EnvAppComm);
    }
  }
  else
  {
    assert(m_Launcher->bEnvDistributedAgents);
    //then this function was called by rank>0 of an app with centralized agents.
    //Therefore, recv the action obtained from master:
    MPI_Bcast(BUF.dataActionBuf, BUF.sizeActionMsg, MPI_BYTE, 0, m_EnvAppComm);
    agent.unpackActionMsg(BUF.dataActionBuf);
  }
}

void Worker::stepWorkerToMaster(const uint64_t bufferID) const
{
  assert(m_MasterWorkersComm != MPI_COMM_NULL);
  assert(MPICommRank(m_MasterWorkersComm) > 0 || m_Learners.size()>0);
  assert(m_Launcher->m_Sockets.clients.size()>0 && "this method should be used by "
         "intermediary workers between socket-apps and mpi-learners");
  assert(m_EnvMPIsize<=1 && "intermediary workers do not support multirank apps");
  sendStateRecvAction( getCommBuffer(bufferID+1) );
}

void Worker::sendStateRecvAction(const COMM_buffer& BUF) const
{
  // MPI MSG to master of a single state:
  MPI_Request send_request, recv_request;
  MPI_Isend(BUF.dataStateBuf, BUF.sizeStateMsg, MPI_BYTE,
      0, 78283, m_MasterWorkersComm, &send_request);
  MPI_Request_free(&send_request);
  // MPI MSG from master of a single action:
  MPI_Irecv(BUF.dataActionBuf, BUF.sizeActionMsg, MPI_BYTE,
      0, 22846, m_MasterWorkersComm, &recv_request);
  while (1) {
    int completed = 0;
    MPI_Test(&recv_request, &completed, MPI_STATUS_IGNORE);
    if (completed) break;
    usleep(1); // wait action from master without burning cpu resources
  }
}

int Worker::getSocketID(const uint64_t worker) const
{
  assert( worker <= m_Launcher->m_Sockets.clients.size() );
  return worker>0? m_Launcher->m_Sockets.clients[worker-1] : m_Launcher->m_Sockets.server;
}

const COMM_buffer& Worker::getCommBuffer(const uint64_t worker) const
{
  assert( worker>0 && worker <= m_Launcher->m_CommBuffers.size() );
  return *m_Launcher->m_CommBuffers[worker-1].get();
}

}
