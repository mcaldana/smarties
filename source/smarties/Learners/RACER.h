//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_RACER_h
#define smarties_RACER_h

#include "Learner_approximator.h"
#include "../Utils/FunctionUtilities.h"

namespace smarties
{

struct Discrete_policy;
struct Continuous_policy;

//template<uint64_t nExperts>
//class Gaussian_mixture;

struct Discrete_advantage;
#ifndef ADV_QUAD
#define Param_advantage Gaussian_advantage
#else
#define Param_advantage Quadratic_advantage
#endif
struct Param_advantage;
struct Zero_advantage;

//#ifndef NEXPERTS
//#define NEXPERTS 1
//#endif
//template<uint64_t nExperts>
//class Mixture_advantage;

#define RACER_simpleSigma
#define RACER_singleNet
//#define RACER_TABC

template<typename Advantage_t, typename Policy_t, typename Action_t>
class RACER : public Learner_approximator
{
 protected:
  // continuous actions: dimensionality of action vectors
  // discrete actions: number of options
  const uint64_t nA = Policy_t::compute_nA(aInfo);
  // number of parameters of advantage approximator
  const uint64_t nL = Advantage_t::compute_nL(aInfo);

  // indices identifying number and starting position of the different output 
  // groups from the network, that are read by separate functions
  // such as state value, policy mean, policy std, adv approximator
  const std::vector<uint64_t> net_outputs;
  const std::vector<uint64_t> net_indices = Utilities::count_indices(net_outputs);
  const std::vector<uint64_t> pol_start, adv_start;
  const uint64_t VsID = net_indices[0];

  const uint64_t batchSize = settings.batchSize, ESpopSize = settings.ESpopSize;
  // used for CMA:
  mutable std::vector<Rvec> rhos=std::vector<Rvec>(batchSize,Rvec(ESpopSize,0));
  mutable std::vector<Rvec> dkls=std::vector<Rvec>(batchSize,Rvec(ESpopSize,0));
  mutable std::vector<Rvec> advs=std::vector<Rvec>(batchSize,Rvec(ESpopSize,0));

  void prepareCMALoss() override;

  //void TrainByEpisodes(const uint64_t seq, const uint64_t wID,
  //  const uint64_t bID, const uint64_t tID) const override;

  void Train(const MiniBatch&MB, const uint64_t wID,const uint64_t bID) const override;

  Rvec policyGradient(const Rvec& MU, const Rvec& ACT,
                      const Policy_t& POL, const Advantage_t& ADV,
                      const Real A_RET, const Real IMPW, const uint64_t thrID) const;

  static std::vector<uint64_t> count_outputs(const ActionInfo& aI);
  static std::vector<uint64_t> count_pol_starts(const ActionInfo& aI);
  static std::vector<uint64_t> count_adv_starts(const ActionInfo& aI);
  void setupNet();
 public:
  RACER(MDPdescriptor& MDP_, HyperParameters& S, ExecutionInfo& D);

  void setupTasks(TaskQueue& tasks) override;
  void selectAction(const MiniBatch& MB, Agent& agent) override;
  void processTerminal(const MiniBatch& MB, Agent& agent) override;
  static uint64_t getnOutputs(const ActionInfo& aI);
  static uint64_t getnDimPolicy(const ActionInfo& aI);
};

template<> uint64_t
RACER<Discrete_advantage, Discrete_policy, uint64_t>::
getnDimPolicy(const ActionInfo& aI);

template<> uint64_t
RACER<Param_advantage, Continuous_policy, Rvec>::
getnDimPolicy(const ActionInfo& aI);

template<> uint64_t
RACER<Zero_advantage, Continuous_policy, Rvec>::
getnDimPolicy(const ActionInfo& aI);

}
#endif
