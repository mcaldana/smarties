//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_ACER_h
#define smarties_ACER_h

#include "Learner_approximator.h"

namespace smarties
{

class ACER : public Learner_approximator
{
protected:
  const uint64_t nA = aInfo.dim();
  const Real explNoise = settings.explNoise;
  const Real acerTrickPow = 1.0 / std::sqrt(nA);
  //const Real acerTrickPow = 1. / nA;
  static constexpr uint64_t nAexpectation = 5;
  Approximator * encoder = nullptr;
  Approximator * actor = nullptr;
  Approximator * value = nullptr;
  Approximator * advtg = nullptr;

  void Train(const MiniBatch& MB, const uint64_t wID, const uint64_t bID) const override;

public:

  static uint64_t getnDimPolicy(const ActionInfo*const aI)
  {
    return 2*aI->dim();
  }

  void setupTasks(TaskQueue& tasks) override;
  void selectAction(const MiniBatch& MB, Agent& agent) override;
  void processTerminal(const MiniBatch& MB, Agent& agent) override;

  ACER(MDPdescriptor&, HyperParameters&, ExecutionInfo&);
  ~ACER() override { };
};

}

#endif // smarties_ACER_h
