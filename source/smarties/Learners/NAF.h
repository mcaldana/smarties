//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_NAF_h
#define smarties_NAF_h

#include "Learner_approximator.h"

namespace smarties
{

class NAF : public Learner_approximator
{
  const uint64_t nA = aInfo.dim(), nL;
  //Network produces a vector. The two following vectors specify:
  // - the sizes of the elements that compose the vector
  // - the starting indices along the output vector of each
  const std::vector<uint64_t> net_outputs = {1, nL, nA, nA};
  const std::vector<uint64_t> net_indices = {0, 1, 1+nL, 1+nL+nA};
  const Real OrUhDecay = settings.clipImpWeight <= 0 ? 0.85 : 0;
  //const Real OrUhDecay = 0; // as in original
  std::vector<Rvec> OrUhState = std::vector<Rvec>( nAgents, Rvec(nA, 0) );

  void Train(const MiniBatch&MB, const uint64_t wID,const uint64_t bID) const override;

public:
  NAF(MDPdescriptor& MDP_, HyperParameters& S_, ExecutionInfo& D_);

  void setupTasks(TaskQueue& tasks) override;
  void selectAction(const MiniBatch& MB, Agent& agent) override;
  void processTerminal(const MiniBatch& MB, Agent& agent) override;

  void test();
};

}
#endif
