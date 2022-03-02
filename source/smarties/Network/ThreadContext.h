//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_ThreadContext_h
#define smarties_ThreadContext_h

#include "../ReplayMemory/MiniBatch.h"
#include "Network.h"
#include "../Core/Agent.h"

namespace smarties
{

enum ADDED_INPUT {NONE, NETWORK, ACTION, VECTOR};
struct ThreadContext
{
  const uint64_t threadID;
  const uint64_t nAddedSamples;
  const bool bHaveTargetW;
  const int64_t targetWeightIndex;
  const uint64_t allSamplCnt = 1 + nAddedSamples + bHaveTargetW;

  //vector over evaluations (eg. target/current or many samples) and over time:
  std::vector<std::vector<std::unique_ptr<Activation>>> activations;
  std::vector<ADDED_INPUT> _addedInputType;
  std::vector<std::vector<NNvec>> _addedInputVec;

  std::shared_ptr<Parameters> partialGradient;

  std::vector<int64_t> lastGradTstep = std::vector<int64_t>(allSamplCnt, -1);
  std::vector<int64_t> weightIndex = std::vector<int64_t>(allSamplCnt, 0);
  const MiniBatch * batch;
  uint64_t batchIndex;

  ThreadContext(const uint64_t thrID,
                const std::shared_ptr<Parameters> grad,
                const uint64_t nAdded,
                const bool bHasTargetWeights,
                const int64_t tgtWeightsID) :
    threadID(thrID), nAddedSamples(nAdded), bHaveTargetW(bHasTargetWeights),
    targetWeightIndex(tgtWeightsID), partialGradient(grad)
  {
    activations.resize(allSamplCnt);
    _addedInputVec.resize(allSamplCnt);
    for(uint64_t i=0; i < allSamplCnt; ++i) {
      activations[i].reserve(MAX_SEQ_LEN);
      _addedInputVec[i].reserve(MAX_SEQ_LEN);
    }
    _addedInputType.resize(allSamplCnt, NONE);
  }

  void setAddedInputType(const ADDED_INPUT type, const int64_t sample = 0)
  {
    addedInputType(sample) = type;
  }

  template<typename T>
  void setAddedInput(const std::vector<T>& addedInput,
                     const uint64_t t, int64_t sample = 0)
  {
    addedInputType(sample) = VECTOR;
    addedInputVec(sample) = NNvec(addedInput.begin(), addedInput.end());
  }

  void load(const std::shared_ptr<Network> NET,
            const MiniBatch & B,
            const uint64_t batchID,
            const uint64_t weightID)
  {
    batch = & B;
    batchIndex = batchID;
    lastGradTstep = std::vector<int64_t>(allSamplCnt, -1);
    weightIndex = std::vector<int64_t>(allSamplCnt, weightID);
    if(bHaveTargetW) weightIndex.back() = targetWeightIndex;

    for(uint64_t i=0; i < allSamplCnt; ++i)
      NET->allocTimeSeries(activations[i], batch->sampledNumSteps(batchIndex));
  }

  void overwrite(const int64_t t, const int64_t sample) const
  {
    if(sample<0) target(t)->written = false;
    else activation(t, sample)->written = false; // what about backprop?
  }

  int64_t& endBackPropStep(const int64_t sample = 0) {
    assert(sample<0 || lastGradTstep.size() > (uint64_t) sample);
    if(sample<0) return lastGradTstep.back();
    else return lastGradTstep[sample];
  }
  int64_t& usedWeightID(const int64_t sample = 0) {
    assert(sample<0 || weightIndex.size() > (uint64_t) sample);
    if(sample<0) return weightIndex.back();
    else return weightIndex[sample];
  }
  ADDED_INPUT& addedInputType(const int64_t sample = 0) {
    assert(sample<0 || _addedInputType.size() > (uint64_t) sample);
    if(sample<0) return _addedInputType.back();
    else return _addedInputType[sample];
  }
  NNvec& addedInputVec(const int64_t t, const int64_t sample = 0) {
    assert(sample<0 || _addedInputVec.size() > (uint64_t) sample);
    if(sample<0) return _addedInputVec.back()[ mapTime2Ind(t) ];
    else return _addedInputVec[sample][ mapTime2Ind(t) ];
  }

  const int64_t& endBackPropStep(const int64_t sample = 0) const {
    assert(sample<0 || lastGradTstep.size() > (uint64_t) sample);
    if(sample<0) return lastGradTstep.back();
    else return lastGradTstep[sample];
  }
  const int64_t& usedWeightID(const int64_t sample = 0) const {
    assert(sample<0 || weightIndex.size() > (uint64_t) sample);
    if(sample<0) return weightIndex.back();
    else return weightIndex[sample];
  }
  const ADDED_INPUT& addedInputType(const int64_t sample = 0) const {
    assert(sample<0 || _addedInputType.size() > (uint64_t) sample);
    if(sample<0) return _addedInputType.back();
    else return _addedInputType[sample];
  }
  const NNvec& addedInputVec(const int64_t t, const int64_t sample = 0) const {
    assert(sample<0 || _addedInputVec.size() > (uint64_t) sample);
    if(sample<0) return _addedInputVec.back()[ mapTime2Ind(t) ];
    else return _addedInputVec[sample][ mapTime2Ind(t) ];
  }

  Activation* activation(const int64_t t, const int64_t sample) const
  {
    assert(sample<0 || activations.size() > (uint64_t) sample);
    const auto& timeSeries = sample<0? activations.back() : activations[sample];
    assert( timeSeries.size() > (uint64_t) mapTime2Ind(t) );
    return timeSeries[ mapTime2Ind(t) ].get();
  }
  Activation* target(const int64_t t) const
  {
    assert(bHaveTargetW);
    return activation(t, -1);
  }
  int64_t mapTime2Ind(const int64_t t) const
  {
    assert(batch != nullptr);
    return batch->mapTime2Ind(batchIndex, t);
  }
  int64_t mapInd2Time(const int64_t k) const
  {
    assert(batch != nullptr);
    return batch->mapInd2Time(batchIndex, k);
  }
  const NNvec& getState(const int64_t t) const
  {
    assert(batch != nullptr);
    return batch->state(batchIndex, t);
  }
  const Rvec& getAction(const int64_t t) const
  {
    assert(batch != nullptr);
    return batch->action(batchIndex, t);
  }
};

struct AgentContext
{
  static constexpr uint64_t nAddedSamples = 0;
  const uint64_t agentID;
  const MiniBatch* batch;
  const Agent* agent;
  //vector over time:
  std::vector<std::unique_ptr<Activation>> activations;
  //std::shared_ptr<Parameters>> partialGradient;
  ADDED_INPUT _addedInputType = NONE;
  NNvec _addedInputVec;
  int64_t lastGradTstep;
  int64_t weightIndex;

  AgentContext(const uint64_t aID) : agentID(aID)
  {
    activations.reserve(MAX_SEQ_LEN);
  }

  void setAddedInputType(const ADDED_INPUT type, const int64_t sample = 0)
  {
    if(type == ACTION) {
      _addedInputType = VECTOR;
      _addedInputVec = NNvec(agent->action.begin(), agent->action.end());
    } else
      _addedInputType = type;
  }

  template<typename T>
  void setAddedInput(const std::vector<T>& addedInput,
                     const uint64_t t, int64_t sample = 0)
  {
    assert(addedInput.size());
    _addedInputType = VECTOR;
    _addedInputVec = NNvec(addedInput.begin(), addedInput.end());
  }

  void load(const std::shared_ptr<Network> NET,
            const MiniBatch& B, const Agent& A,
            const uint64_t weightID)
  {
    batch = & B;
    agent = & A;
    assert(A.ID == agentID);
    lastGradTstep = -1;
    weightIndex = weightID;
    NET->allocTimeSeries(activations, batch->sampledNumSteps(0));
  }

  void overwrite(const int64_t t, const int64_t sample = -1) const
  {
    activation(t)->written = false; // what about backprop?
  }

  int64_t& endBackPropStep(const int64_t sample =-1) {
    return lastGradTstep;
  }
  int64_t& usedWeightID(const int64_t sample =-1) {
    return weightIndex;
  }
  ADDED_INPUT& addedInputType(const int64_t sample =-1) {
    return _addedInputType;
  }
  NNvec& addedInputVec(const int64_t t, const int64_t sample = -1)
  {
    assert(t+1 == (int64_t) episode()->nsteps());
    return _addedInputVec;
  }

  const int64_t& endBackPropStep(const int64_t sample =-1) const {
    return lastGradTstep;
  }
  const int64_t& usedWeightID(const int64_t sample =-1) const {
    return weightIndex;
  }
  const ADDED_INPUT& addedInputType(const int64_t sample =-1) const {
    return _addedInputType;
  }
  const NNvec& addedInputVec(const int64_t t, const int64_t sample = -1) const {
    assert(t+1 == (int64_t) episode()->nsteps());
    return _addedInputVec;
  }

  Activation* activation(const int64_t t, const int64_t sample = -1) const
  {
    return activations[ mapTime2Ind(t) ].get();
  }
  int64_t mapTime2Ind(const int64_t t) const
  {
    assert(batch != nullptr);
    return batch->mapTime2Ind(0, t);
  }
  int64_t mapInd2Time(const int64_t k) const
  {
    assert(batch != nullptr);
    return batch->mapInd2Time(0, k);
  }
  const NNvec& getState(const int64_t t) const
  {
    assert(batch != nullptr);
    return batch->state(0, t);
  }
  const Rvec& getAction(const int64_t t) const
  {
    assert(batch != nullptr);
    return batch->action(0, t);
  }
  const Episode* episode() const
  {
    assert(batch != nullptr);
    assert(batch->episodes.size() == 1 && batch->episodes[0] != nullptr);
    return batch->episodes[0];
  }
};

} // end namespace smarties
#endif // smarties_Quadratic_term_h
