//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_CMALearner_h
#define smarties_CMALearner_h

#include "Learner_approximator.h"

namespace smarties
{

template<typename Action_t>
class CMALearner: public Learner_approximator
{
  const uint64_t ESpopSize = settings.ESpopSize;
  const uint64_t nOwnEnvs = m_ExecutionInfo.nOwnedEnvironments;
  const uint64_t nOwnAgents = m_ExecutionInfo.nOwnedAgentsPerAlgo;
  const uint64_t nOwnAgentsPerEnv = nOwnAgents / nOwnEnvs;

  // counter per each env of how many agents have currently terminated on this
  //   simulation. no agent can restart unless they all have terminated a sim
  std::vector<uint64_t> curNumEndedPerEnv = std::vector<uint64_t>(nOwnEnvs, 0);
  std::vector<uint64_t> curNumStartedPerEnv = std::vector<uint64_t>(nOwnEnvs, 0);

  std::mutex workload_mutex;
  uint64_t lastWorkLoadStarted = 0;

  std::vector<uint64_t> weightIDs = std::vector<uint64_t>(nOwnEnvs, 0);

  std::vector<Rvec> R = std::vector<Rvec>(nOwnEnvs, Rvec(ESpopSize, 0) );
  std::vector<std::vector<uint64_t>> Ns = std::vector<std::vector<uint64_t>>(nOwnEnvs,
                                            std::vector<uint64_t>(ESpopSize, 0) );

  static std::vector<uint64_t> count_pol_outputs(const ActionInfo*const aI);
  static std::vector<uint64_t> count_pol_starts(const ActionInfo*const aI);

  void prepareCMALoss() override;

  void assignWeightID(const Agent& agent);
  void computeAction(Agent& agent, const Rvec netOutput) const;
  void Train(const MiniBatch&MB,const uint64_t wID,const uint64_t bID) const override;

public:
  CMALearner(MDPdescriptor& MDP_, HyperParameters& S_, ExecutionInfo& D_);

  //main training functions:
  void setupTasks(TaskQueue& tasks) override;
  void selectAction(const MiniBatch& MB, Agent& agent) override;
  void processTerminal(const MiniBatch& MB, Agent& agent) override;

  bool blockGradientUpdates() const override;
  bool blockDataAcquisition() const override;

  static uint64_t getnDimPolicy(const ActionInfo*const aI);
};

template<> uint64_t CMALearner<uint64_t>::getnDimPolicy(const ActionInfo*const aI);

template<> uint64_t CMALearner<Rvec>::getnDimPolicy(const ActionInfo*const aI);

}
#endif
