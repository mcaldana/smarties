//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Discrete_policy_h
#define smarties_Discrete_policy_h

#include "../Network/Layers/Functions.h"
#include "../Core/Agent.h"

namespace smarties
{

template<typename PosDefFunction>
struct Discrete_policy_t
{
  typedef uint64_t Action_t;
  const ActionInfo& aInfo;
  const uint64_t startProbs, nO;
  const Rvec netOutputs;
  const Rvec unnorm;
  const Real normalization;
  const Rvec probs;

  static uint64_t compute_nA(const ActionInfo & aI) {
    assert(aI.dimDiscrete());
    return aI.dimDiscrete();
  }
  static uint64_t compute_nPol(const ActionInfo & aI) {
    return aI.dimDiscrete();
  }

  static void setInitial_Stdev(const ActionInfo& aI, Rvec& O, const Real S) {
    printf("Stdev is not defined for discrete policy\n");
  }
  static Rvec initial_Stdev(const ActionInfo& aI, const Real S) {
    printf("Stdev is not defined for discrete policy\n");
    return Rvec();
  }

  static void setInitial_noStdev(const ActionInfo& aI, Rvec& initBias) {
    for(uint64_t e=0; e<aI.dimDiscrete(); e++) initBias.push_back(0);
  }

  Discrete_policy_t(const ActionInfo& aI, const Rvec& out) : aInfo(aI),
    startProbs(0), nO(aI.dimDiscrete()), netOutputs(out),
    unnorm(extract_unnorm()), normalization(compute_norm()),
    probs(extract_probabilities())
  {
  }

  Discrete_policy_t(const std::vector<uint64_t>& start, const ActionInfo& aI,
    const Rvec& out) : aInfo(aI), startProbs(start[0]), nO(aI.dimDiscrete()),
    netOutputs(out), unnorm(extract_unnorm()), normalization(compute_norm()),
    probs(extract_probabilities())
  {
  }

  Rvec extract_unnorm() const {
    assert(netOutputs.size() >= startProbs + nO);
    Rvec ret(nO);
    for (uint64_t j=0; j<nO; ++j)
        ret[j] = PosDefFunction::_eval(netOutputs[startProbs + j]);
    return ret;
  }

  Real compute_norm() const {
    assert(unnorm.size() == nO);
    Real ret = 0;
    for (uint64_t j=0; j<nO; ++j) { ret += unnorm[j]; assert(unnorm[j]>0); }
    return std::max(ret, std::numeric_limits<Real>::epsilon() );
  }

  Rvec extract_probabilities() const {
    assert(unnorm.size() == nO);
    Rvec ret(nO);
    for (uint64_t j=0; j<nO; ++j) ret[j] = unnorm[j]/normalization;
    return ret;
  }

  Real importanceWeight(const Rvec& action, const Rvec& beta) const {
    const uint64_t option = aInfo.actionMessage2label(action);
    return importanceWeight(option, beta);
  }
  Real importanceWeight(const uint64_t option, const Rvec& beta) const {
    assert(beta.size() == nO && option < nO);
    return probs[option] / beta[option];
  }

  static Real evalBehavior(const uint64_t option, const Rvec& beta) {
    return beta[option];
  }
  Real evalBehavior(const Rvec action, const Rvec& beta) const {
    const uint64_t option = aInfo.actionMessage2label(action);
    assert(beta.size() == nO && option < nO);
    return beta[option];
  }

  Real evalProbability(const uint64_t option) const {
    return probs[option];
  }
  Real evalProbability(const Rvec action) const {
    const uint64_t option = aInfo.actionMessage2label(action);
    assert(option < nO);
    return probs[option];
  }
  Real evalLogProbability(const uint64_t option) const {
    return std::log(evalProbability(option));
  }
  Real evalLogProbability(const Rvec& action) const {
    return std::log(evalProbability(action));
  }

  template<typename T>
  Real KLDivergence(const T*const tgt_pol) const {
    return KLDivergence(tgt_pol->getVector());
  }
  template<typename T>
  Real KLDivergence(const T& tgt_pol) const {
    return KLDivergence(tgt_pol.getVector());
  }
  Real KLDivergence(const Rvec& beta) const {
    Real ret = 0;
    for (uint64_t i=0; i<nO; ++i) ret += probs[i]*std::log(probs[i]/beta[i]);
    return ret;
  }

  Rvec policyGradient(const Rvec& action, const Real factor = 1) const {
    const uint64_t option = aInfo.actionMessage2label(action);
    return policyGradient(option, factor);
  }
  Rvec policyGradient(const uint64_t option, const Real factor = 1) const {
    Rvec ret(nO, 0);
    ret[option] = factor/unnorm[option];
    for (uint64_t i=0; i<nO; ++i) {
      ret[i] -= factor/normalization;
      ret[i] *= PosDefFunction::_evalDiff(netOutputs[startProbs + i]);
    }
    return ret;
  }

  template<typename T>
  Rvec KLDivGradient(const T*const tgt, const Real C = 1) const {
    return KLDivGradient(tgt->getVector(), C);
  }
  template<typename T>
  Rvec KLDivGradient(const T& tgt, const Real C = 1) const {
    return KLDivGradient(tgt.getVector(), C);
  }
  Rvec KLDivGradient(const Rvec& beta, const Real fac = 1) const {
    Rvec ret(nO, 0);
    for (uint64_t j=0; j<nO; ++j){
      const Real tmp = fac * (1 + std::log(probs[j]/beta[j])) / normalization;
      for (uint64_t i=0; i<nO; ++i) ret[i] += tmp * ((i==j) - probs[j]);
    }
    for (uint64_t j=0; j<nO; ++j)
      ret[j] *= PosDefFunction::_evalDiff(netOutputs[startProbs + j]);
    return ret;
  }

  void makeNetworkGrad(Rvec& netGradient, const Rvec& totPolicyG) const {
    assert(netGradient.size() >= startProbs+nO && totPolicyG.size() == nO);
    for (uint64_t j=0; j<nO; ++j) netGradient[startProbs + j] = totPolicyG[j];
  }
  Rvec makeNetworkGrad(const Rvec& totPolicyGrad) const {
    Rvec ret(nO);
    makeNetworkGrad(ret, totPolicyGrad);
    return ret;
  }

  Rvec getVector() const {
    return probs;
  }

  static uint64_t sample(std::mt19937& gen, const Rvec& beta) {
    std::discrete_distribution<uint64_t> dist(beta.begin(), beta.end());
    return dist(gen);
  }
  uint64_t sample(std::mt19937& gen) const {
    std::discrete_distribution<uint64_t> dist(probs.begin(), probs.end());
    return dist(gen);
  }

  template<typename Advantage_t>
  Rvec control_grad(const Advantage_t*const adv, const Real eta) const {
    Rvec ret(nO, 0);
    for (uint64_t j=0; j<nO; ++j) {
      ret[j] = eta * adv->computeAdvantage(j)/normalization;
      ret[j] *= PosDefFunction::_evalDiff(netOutputs[startProbs + j]);
    }
    return ret;
  }

  uint64_t selectAction(Agent& agent, const bool bTrain) const {
    const bool bSample = bTrain && agent.trackEpisodes;
    return bSample? sample(agent.generator) : Utilities::maxInd(probs);
  }

  void test(const uint64_t act, const Rvec& beta) const;
};

struct Discrete_policy : Discrete_policy_t<SoftPlus> {
    using Discrete_policy_t<SoftPlus>::Discrete_policy_t;
};

} // end namespace smarties
#endif // smarties_Discrete_policy_h
