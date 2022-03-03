//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Approximator_h
#define smarties_Approximator_h

#include "../Math/Continuous_policy.h"
#include "../ReplayMemory/MemoryBuffer.h"
#include "../Utils/StatsTracker.h"
#include "Builder.h"
#include "ThreadContext.h"

namespace smarties {

class ParameterBlob;

class Approximator {
 public:
  Approximator(std::string name_,
               const HyperParameters&,
               const ExecutionInfo&,
               const MemoryBuffer* const replay_,
               const Approximator* const preprocessing_ = nullptr,
               const Approximator* const auxInputNet_ = nullptr);
  ~Approximator();

  // CMALearner, RACER::prepareCMALoss
  mutable std::atomic<uint64_t> nAddedGradients{0};

  //ACER, MixedPG
  void setNumberOfAddedSamples(const uint64_t nSamples = 0);

  // specify type (and size) of auxiliary input (set m_auxInputSize)
  void setAddedInput(const ADDED_INPUT type, int64_t size = -1);

  // specify whether we are using target networks
  void setUseTargetNetworks(const int64_t targetNetworkSampleID = -1,
                            const bool bTargetNetUsesTargetWeights = true);

  void buildFromSettings(const uint64_t outputSize);
  void buildFromSettings(const std::vector<uint64_t> outputSizes);
  void buildPreprocessing(const std::vector<uint64_t> outputSizes);
  Builder& getBuilder() { assert(build.get()); return *build.get(); }

  void initializeNetwork();

  uint64_t nOutputs() const { return net ? net->getnOutputs() : 0; }

  void setNgradSteps(const uint64_t iter) const { opt->nStep = iter; }

  void updateGradStats(const std::string& base, const uint64_t iter) const;

  void load(const MiniBatch& B,
            const uint64_t batchID,
            const int64_t wghtID) const;

  void load(const MiniBatch& B,
            const Agent& agent,
            const int64_t wghtID = 0) const;

  template <typename contextid_t, typename val_t>
  void setAddedInput(const std::vector<val_t>& addedInput,
                     const contextid_t& contextID,
                     const uint64_t t,
                     int64_t sampID = 0) const {
    getContext(contextID).setAddedInput(addedInput, t, sampID);
  }

  template <typename contextid_t>
  void setAddedInputType(const ADDED_INPUT& type,
                         const contextid_t& contextID,
                         const uint64_t t,
                         int64_t sampID = 0) const {
    getContext(contextID).setAddedInputType(type, sampID);
  }

  // forward: compute net output taking care also to gather additional required
  // inputs such as recurrent connections and auxiliary input networks.
  // It expects as input either the index over a previously loaded minibatch
  // or a previously loaded agent.
  template <typename contextid_t>
  Rvec forward(const contextid_t& contextID,
               const uint64_t t,
               int64_t sampID = 0,
               const bool overwrite = false) const {
    auto& C = getContext(contextID);
    if (sampID > (int64_t)C.nAddedSamples) {
      sampID = 0;
    }
    if (overwrite)
      C.activation(t, sampID)->written = false;
    if (C.activation(t, sampID)->written)
      return C.activation(t, sampID)->getOutput();
    const uint64_t ind = C.mapTime2Ind(t);

    // Compute previous outputs if needed by recurrencies. LIMITATION:
    // What should we do for target net / additional samples?
    // next line assumes we want to use curr W and sample 0 for recurrencies
    // as well as using actions as added input instead of vectors/networks
    if (ind > 0)
      assert(t > 0);
    if (ind > 0 && not C.activation(t - 1, 0)->written) {
      const auto myInputType = C.addedInputType(0);
      if (myInputType != NONE)
        C.addedInputType(0) = ACTION;
      forward(contextID, t - 1, 0);
      if (myInputType != NONE)
        C.addedInputType(0) = myInputType;
    }
    // if(ind>0 && not C.net(t, samp)->written) forward(C, t-1, samp);
    const Activation* const recur = ind > 0 ? C.activation(t - 1, 0) : nullptr;
    const Activation* const activation = C.activation(t, sampID);
    const Parameters* const W = opt->getWeights(C.usedWeightID(sampID));
    //////////////////////////////////////////////////////////////////////////////
    NNvec INP;
    if (preprocessing) {
      const Rvec preprocInp = preprocessing->forward(contextID, t, sampID);
      INP.insert(INP.end(), preprocInp.begin(), preprocInp.end());
    } else
      INP = C.getState(t);

    if (C.addedInputType(sampID) == NETWORK) {
      assert(auxInputNet);
      Rvec addedinp = auxInputNet->forward(contextID, t, sampID);
      assert((int64_t)addedinp.size() >= m_auxInputSize);
      addedinp.resize(m_auxInputSize);
      INP.insert(INP.end(), addedinp.begin(), addedinp.end());
    } else if (C.addedInputType(sampID) == ACTION) {
      const auto& addedinp = C.getAction(t);
      INP.insert(INP.end(), addedinp.begin(), addedinp.end());
    } else if (C.addedInputType(sampID) == VECTOR) {
      const auto& addedinp = C.addedInputVec(t, sampID);
      INP.insert(INP.end(), addedinp.begin(), addedinp.end());
    }
    assert(INP.size() == net->getnInputs());
    ////////////////////////////////////////////////////////////////////////////
    return net->forward(INP, recur, activation, W);
  }

  // forward target network
  template <typename contextid_t>
  Rvec forward_tgt(const contextid_t& contextID,
                   const uint64_t t,
                   const bool overwrite = false) const {
    return forward(contextID, t, -1, overwrite);
  }

  // run network for agent's recent step
  Rvec forward(const Agent& agent, const bool overwrite = false) const;

  void setGradient(const Rvec& gradient,
                   const uint64_t batchID,
                   const uint64_t t,
                   int64_t sampID = 0) const;

  Real& ESloss(const uint64_t ESweightID = 0) { return losses[ESweightID]; }

  Rvec oneStepBackProp(const Rvec& gradient,
                       const uint64_t batchID,
                       const uint64_t t,
                       int64_t sampID) const;

  Rvec getStepBackProp(const uint64_t batchID,
                       const uint64_t t,
                       int64_t sampID) const;

  void backProp(const uint64_t batchID) const;

  void prepareUpdate();
  bool ready2ApplyUpdate() {
    return (reducedGradients == 0) || opt->ready2UpdateWeights();
  }
  // apply optimizer
  void applyUpdate();

  // append 'this' parameters to the 'params'
  void gatherParameters(ParameterBlob& params) const;

  void getHeaders(std::ostringstream& buff) const;
  void getMetrics(std::ostringstream& buff) const;
  void save(const std::string base, const bool bBackup);
  void restart(const std::string base);
  void rename(std::string newname) { name = newname; }

 private:
  const HyperParameters& settings;
  const ExecutionInfo& m_ExecutionInfo;
  std::string name;
  const uint64_t nAgents = m_ExecutionInfo.nAgents,
                 nThreads = m_ExecutionInfo.nThreads;
  const uint64_t ESpopSize = settings.ESpopSize, batchSize = settings.batchSize;
  const MemoryBuffer* const replay;
  const Approximator* const preprocessing;
  const Approximator* const auxInputNet;
  const ActionInfo& aI = replay->aI;
  int64_t auxInputAttachLayer = -1;
  int64_t m_auxInputSize = -1;
  uint64_t m_numberOfAddedSamples = 0;
  bool m_UseTargetNetwork = false;
  bool m_bTargetNetUsesTargetWeights = true;
  int64_t m_targetNetworkSampleID = -1;
  uint64_t reducedGradients = 0;

  // when this flag is true, specification of network properties is disabled:
  bool bCreatedNetwork = false;

  // Whether to backprop gradients in the input network.
  // Some papers do not propagate policy gradients towards encoding layers
  bool m_blockInpGrad = false;

  std::shared_ptr<Network> net;
  std::shared_ptr<Optimizer> opt;
  std::unique_ptr<Builder> build;

  mutable std::vector<uint64_t> threadsPerBatch =
      std::vector<uint64_t>(batchSize, -1);
  std::vector<std::unique_ptr<ThreadContext>> contexts;
  std::vector<std::unique_ptr<AgentContext>> agentsContexts;
  StatsTracker* gradStats = nullptr;

  // For CMAES based optimization. Keeps track of total loss associate with
  // Each weight vector sample:
  mutable Rvec losses = Rvec(ESpopSize, 0);

  ThreadContext& getContext(const uint64_t batchID) const;
  AgentContext& getContext(const Agent& agent) const;
  void setBlockGradsToPreprocessing();
  uint64_t nLayers() const;
};

}  // end namespace smarties
#endif  // smarties_Approximator_h
