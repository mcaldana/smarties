//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Communicator.h"
#include "Utils/SocketsLib.h"
#include "Core/Worker.h"

namespace smarties
{

void Communicator::setStateActionDims(const int dimState,
                                         const int dimAct,
                                         const int agentID)
{
  if(m_Environment.bFinalized) {
    warn("Cannot edit env description after having sent first state."); return;
  }
  if( (size_t) agentID >= m_Environment.descriptors.size())
    die("Attempted to write to uninitialized MDPdescriptor");

  m_Environment.descriptors[agentID]->dimState = dimState;
  m_Environment.descriptors[agentID]->dimAction = dimAct;
}

void Communicator::setActionScales(const std::vector<double> uppr,
                                     const std::vector<double> lowr,
                                     const bool bound,
                                     const int agentID)
{
  setActionScales(uppr,lowr, std::vector<bool>(uppr.size(),bound), agentID);
}
void Communicator::setActionScales(const std::vector<double> upper,
                                   const std::vector<double> lower,
                                   const std::vector<bool>   bound,
                                   const int agentID)
{
  if(m_Environment.bFinalized) {
    warn("Cannot edit env description after having sent first state."); return;
  }
  if(agentID >= (int) m_Environment.descriptors.size())
    die("Attempted to write to uninitialized MDPdescriptor");
  if(upper.size() != m_Environment.descriptors[agentID]->dimAction or
     lower.size() != m_Environment.descriptors[agentID]->dimAction or
     bound.size() != m_Environment.descriptors[agentID]->dimAction )
    die("size mismatch");
  if(m_Environment.descriptors[agentID]->bDiscreteActions())
    die("either continuous or discrete actions");

  m_Environment.descriptors[agentID]->upperActionValue =
                Rvec(upper.begin(), upper.end());
  m_Environment.descriptors[agentID]->lowerActionValue =
                Rvec(lower.begin(), lower.end());
  m_Environment.descriptors[agentID]->bActionSpaceBounded =
    std::vector<bool>(bound.begin(), bound.end());
}

void Communicator::setActionOptions(const int options,
                                      const int agentID)
{
  setActionOptions(std::vector<int>(1, options), agentID);
}

void Communicator::setActionOptions(const std::vector<int> options,
                                      const int agentID)
{
  if(m_Environment.bFinalized) {
    warn("Cannot edit env description after having sent first state."); return;
  }
  if(agentID >= (int) m_Environment.descriptors.size())
    die("Attempted to write to uninitialized MDPdescriptor");
  if(options.size() != m_Environment.descriptors[agentID]->dimAction)
    die("size mismatch");

  m_Environment.descriptors[agentID]->discreteActionValues =
    std::vector<uint64_t>(options.begin(), options.end());
}

void Communicator::setStateObservable(const std::vector<bool> observable,
                                        const int agentID)
{
  if(m_Environment.bFinalized) {
    warn("Cannot edit env description after having sent first state."); return;
  }
  if(agentID >= (int) m_Environment.descriptors.size())
    die("Attempted to write to uninitialized MDPdescriptor");
  if(observable.size() != m_Environment.descriptors[agentID]->dimState)
    die("size mismatch");

  m_Environment.descriptors[agentID]->bStateVarObserved =
    std::vector<bool>(observable.begin(), observable.end());
}

void Communicator::setStateScales(const std::vector<double> upper,
                                    const std::vector<double> lower,
                                    const int agentID)
{
  if(m_Environment.bFinalized) {
    warn("Cannot edit env description after having sent first state.");
    return;
  }
  if(agentID >= (int) m_Environment.descriptors.size())
    die("Attempted to write to uninitialized MDPdescriptor");
  const uint64_t dimS = m_Environment.descriptors[agentID]->dimState;
  if(upper.size() != dimS or lower.size() != dimS )
    die("size mismatch");

  // For consistency with action space we ask user for a rough box of state vars
  // but in reality we scale with mean and stdev computed during training.
  // This function serves only as an optional initialization for statistiscs.
  NNvec meanState(dimS), diffState(dimS);
  for (uint64_t i=0; i<dimS; ++i) {
    meanState[i] = (upper[i]+lower[i])/2;
    diffState[i] = std::fabs(upper[i]-lower[i]);
  }
  m_Environment.descriptors[agentID]->stateMean   = meanState;
  m_Environment.descriptors[agentID]->stateStdDev = diffState;
}

void Communicator::setIsPartiallyObservable(const int agentID)
{
  if(m_Environment.bFinalized) {
    warn("Cannot edit env description after having sent first state."); return;
  }
  if(agentID >= (int) m_Environment.descriptors.size())
    die("Attempted to write to uninitialized MDPdescriptor");

  m_Environment.descriptors[agentID]->isPartiallyObservable = true;
}

void Communicator::setPreprocessingConv2d(
  const int input_width, const int input_height, const int input_features,
  const int kernels_num, const int filters_size, const int stride,
  const int agentID)
{
  if(m_Environment.bFinalized) {
    warn("Cannot edit env description after having sent first state."); return;
  }
  if(agentID >= (int) m_Environment.descriptors.size())
    die("Attempted to write to uninitialized MDPdescriptor");

  // can be made to be more powerful (different sizes in x/y, padding, etc)
  Conv2D_Descriptor descr;
  descr.inpFeatures = input_features;
  descr.inpY        = input_height;
  descr.inpX        = input_width;
  descr.outFeatures = kernels_num;
  descr.filterx     = filters_size;
  descr.filtery     = filters_size;
  descr.stridex     = stride;
  descr.stridey     = stride;
  descr.paddinx     = 0;
  descr.paddiny     = 0;
  descr.outY   = (descr.inpY -descr.filterx +2*descr.paddinx)/descr.stridex + 1;
  descr.outX   = (descr.inpX -descr.filtery +2*descr.paddiny)/descr.stridey + 1;
  m_Environment.descriptors[agentID]->conv2dDescriptors.push_back(descr);
}

void Communicator::setNumAppendedPastObservations(
  const int n_appended, const int agentID)
{
  if(m_Environment.bFinalized) {
    warn("Cannot edit env description after having sent first state."); return;
  }
  if(agentID >= (int) m_Environment.descriptors.size())
    die("Attempted to write to uninitialized MDPdescriptor");

  m_Environment.descriptors[agentID]->nAppendedObs = n_appended;
}

void Communicator::setNumAgents(int _nAgents)
{
  if(m_Environment.bFinalized) {
    warn("Cannot edit env description after having sent first state."); return;
  }
  assert(_nAgents > 0);

  m_Environment.nAgentsPerEnvironment = _nAgents;
}

void Communicator::envHasDistributedAgents()
{
  /*
  if(comm_inside_app == MPI_COMM_NULL) {
    printf("ABORTING: Distributed agents has no effect on single-process "
    " applications. It means that each simulation rank holds different agents.");
    fflush(0); abort();
    m_bEnvDistributedAgents = false;
    return;
  }
  */
  if(m_Environment.bFinalized) {
    warn("Cannot edit env description after having sent first state."); return;
  }
  if(m_Environment.bAgentsHaveSeparateMDPdescriptors)
    die("ABORTING: Smarties supports either distributed agents (ie each "
    "worker holds some of the agents) or each agent defining a different MDP "
    "(state/act spaces).");

  m_bEnvDistributedAgents =  true;
}

void Communicator::agentsDefineDifferentMDP()
{
  if(m_Environment.bFinalized) {
    warn("Cannot edit env description after having sent first state."); return;
  }
  if(m_bEnvDistributedAgents) {
    printf("ABORTING: Smarties supports either distributed agents (ie each "
    "worker holds some of the agents) or each agent defining a different MDP "
    "(state/act spaces)."); fflush(0); abort();
  }

  m_Environment.initDescriptors(true);
}

void Communicator::disableDataTrackingForAgents(int agentStart, int agentEnd)
{
  if(m_Environment.bFinalized) {
    warn("Cannot edit env description after having sent first state."); return;
  }

  m_Environment.bTrainFromAgentData.resize(m_Environment.nAgentsPerEnvironment, 1);
  for(int i=agentStart; i<agentEnd; ++i)
    m_Environment.bTrainFromAgentData[i] = 0;
}

void Communicator::agentsShareExplorationNoise(const int agentID)
{
  if(m_Environment.bFinalized) {
    warn("Cannot edit env description after having sent first state."); return;
  }
  if(m_bEnvDistributedAgents) {
    printf("ABORTING: Shared exploration noise is not yet supported for "
      "distributed agents (noise is not communicated with MPI to other ranks "
      "running the simulation)."); fflush(0); abort();
  }
  if(agentID >= (int) m_Environment.descriptors.size())
    die("Attempted to write to uninitialized MDPdescriptor");
  m_Environment.descriptors[agentID]->bAgentsShareNoise = true;
}

void Communicator::finalizeProblemDescription()
{
  if(m_Environment.bFinalized) {
    warn("Cannot edit env description after having sent first state."); return;
  }
  synchronizeEnvironments();
}

void Communicator::_sendState(const int agentID, const EpisodeStatus status,
    const std::vector<double>& state, const double reward)
{
  if( not m_Environment.bFinalized ) synchronizeEnvironments(); // race condition
  if(m_bTrainIsOver)
    die("App recvd end-of-training signal but did not abort on it's own.");

  //const auto& MDP = m_Environment.getDescriptor(agentID);
  assert(agentID>=0 && (uint64_t) agentID < agents.size());
  assert(agents[agentID]->localID == (unsigned) agentID);
  assert(agents[agentID]->ID == (unsigned) agentID);
  agents[agentID]->update(status, state, reward);
  #ifndef NDEBUG
    if (agents[agentID]->stateIsInvalid())
      die("Environment gave a nan or inf state or reward.");
  #endif

  if(m_Sockets.server == -1)
  {
    assert(m_Worker);
    m_Worker->stepWorkerToMaster( *agents[agentID].get() );
  }
  else
  {
    agents[agentID]->packStateMsg(m_CommBuffers[agentID]->dataStateBuf);
    SOCKET_Bsend(m_CommBuffers[agentID]->dataStateBuf,
                 m_CommBuffers[agentID]->sizeStateMsg,
                 m_Sockets.server);
    SOCKET_Brecv(m_CommBuffers[agentID]->dataActionBuf,
                 m_CommBuffers[agentID]->sizeActionMsg,
                 m_Sockets.server);
    agents[agentID]->unpackActionMsg(m_CommBuffers[agentID]->dataActionBuf);
  }

  if(status >= LAST) {
    agents[agentID]->learnerAvgCumulativeReward = agents[agentID]->action[0];
  }
  // we cannot control application. if we received a termination signal we abort
  if(agents[agentID]->learnStatus == KILL) {
    printf("App recvd end-of-training signal.\n");
    m_bTrainIsOver = true;
  }
}

const std::vector<double> Communicator::recvAction(const int agentID) const
{
  assert( agents[agentID]->agentStatus < LAST && "Application read action for "
    "a terminal state or truncated episode. Undefined behavior.");
  return agents[agentID]->getAction();
}

int Communicator::recvDiscreteAction(const int agentID) const
{
  assert( agents[agentID]->agentStatus < LAST && "Application read action for "
    "a terminal state or truncated episode. Undefined behavior.");
  return (int) agents[agentID]->getDiscreteAction();
}

void Communicator::synchronizeEnvironments()
{
  if ( m_Environment.bFinalized ) return;

  if(m_Sockets.server == -1)
  {
    assert(m_Worker);
    m_Worker->synchronizeEnvironments();
  }
  else
  {
    initOneCommunicationBuffer();
    const auto sendBufferFunc = [&](void* buffer, size_t size) {
      SOCKET_Bsend(buffer, size, m_Sockets.server);
    };
    m_Environment.synchronizeEnvironments(sendBufferFunc);

    // allocate rest of communication buffers:
    for(size_t i=1; i<agents.size(); ++i) initOneCommunicationBuffer();
  }
  assert(m_CommBuffers.size() > 0);
}

void Communicator::initOneCommunicationBuffer()
{
  uint64_t maxDimState  = 0, maxDimAction = 0;
  assert(m_Environment.descriptors.size() > 0);
  for(size_t i=0; i<m_Environment.descriptors.size(); ++i)
  {
    maxDimState  = std::max(maxDimState,  m_Environment.descriptors[i]->dimState );
    maxDimAction = std::max(maxDimAction, m_Environment.descriptors[i]->dimAction);
  }
  assert(m_Environment.nAgentsPerEnvironment>0);
  assert(maxDimAction>0); // state can be 0-D
  m_CommBuffers.emplace_back(std::make_unique<COMM_buffer>(maxDimState, maxDimAction) );
}

std::mt19937& Communicator::getPRNG() {
  return m_RandomGen;
}
Real Communicator::getUniformRandom(const Real begin, const Real end)
{
  std::uniform_real_distribution<Real> distribution(begin, end);
  return distribution(m_RandomGen);
}
Real Communicator::getNormalRandom(const Real mean, const Real stdev)
{
  std::normal_distribution<Real> distribution(mean, stdev);
  return distribution(m_RandomGen);
}

bool Communicator::isTraining() const {
  return m_bTrain;
}
bool Communicator::terminateTraining() const {
  return m_bTrainIsOver;
}

unsigned Communicator::getLearnersGradStepsNum(const int agentID)
{
  return agents[agentID]->learnerGradStepID;
}
unsigned Communicator::getLearnersTrainingTimeStepsNum(const int agentID)
{
  return agents[agentID]->learnerTimeStepID;
}
double Communicator::getLearnersAvgCumulativeReward(const int agentID)
{
  return agents[agentID]->learnerAvgCumulativeReward;
}

Communicator::Communicator(Worker*const W, std::mt19937&G, bool isTraining) :
m_RandomGen(G()), m_bTrain(isTraining), m_Worker(W) {}

}
