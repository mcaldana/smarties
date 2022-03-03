//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Quadratic_advantage_h
#define smarties_Quadratic_advantage_h

#include "Continuous_policy.h"
#include "Quadratic_term.h"

namespace smarties
{

struct Quadratic_advantage: public Quadratic_term
{
  const Continuous_policy* const policy;

  //Normalized quadratic advantage, with own mean
  Quadratic_advantage(const std::vector<uint64_t>&starts,
                      const ActionInfo& aI,
                      const Rvec& out,
                      const Continuous_policy*const pol = nullptr) :
    Quadratic_term(aI, starts[0], starts.size()>1? starts[1] : 0, out,
                   pol ? pol->getMean() : Rvec()), policy(pol) { }

  void grad(const Rvec&act, const Real Qer, Rvec& netGradient) const
  {
    assert(act.size()==nA);
    Rvec dErrdP(nA*nA, 0), dPol(nA, 0), dAct(nA);
    for (uint64_t j=0; j<nA; ++j) dAct[j] = act[j] - mean[j];

    assert(!policy);
    //for (uint64_t j=0; j<nA; ++j) dPol[j] = policy->mean[j] - mean[j];

    for (uint64_t i=0; i<nA; ++i)
    for (uint64_t j=0; j<=i; ++j) {
      Real dOdPij = -dAct[j] * dAct[i];

      dErrdP[nA*j +i] = Qer*dOdPij;
      dErrdP[nA*i +j] = Qer*dOdPij; //if j==i overwrite, avoid `if'
    }

    for (uint64_t j=0, kl = start_matrix; j<nA; ++j)
      for (uint64_t i=0; i<=j; ++i) {
        Real dErrdL = 0;
        for (uint64_t k=i; k<nA; ++k) dErrdL += dErrdP[nA*j +k] * L[nA*k +i];

        if(i==j)
          netGradient[kl] = dErrdL * PosDefFunction::_evalDiff(netOutputs[kl]);
        else if(i<j)
          netGradient[kl] = dErrdL;
        kl++;
      }

    if(start_mean>0) {
      assert(netGradient.size() >= start_mean+nA);
      for (uint64_t a=0, ka=start_mean; a<nA; ++a, ++ka)
      {
        Real val = 0;
        for (uint64_t i=0; i<nA; ++i)
          val += Qer * matrix[nA*a + i] * (dAct[i]-dPol[i]);

        netGradient[ka] = val;
        if(aInfo.isBounded(a))
          netGradient[ka] *= BoundedActFunction::_evalDiff(netOutputs[ka]);
      }
    }
  }

  Real computeAdvantage(const Rvec& action) const
  {
    Real ret = -quadraticTerm(action);
    if(policy)
    { //subtract expectation from advantage of action
      ret += quadraticTerm(policy->getMean());
      for(uint64_t i=0; i<nA; ++i)
        ret += matrix[nA*i+i] * policy->getVariance(i);
    }
    return 0.5*ret;
  }

  Real computeAdvantageNoncentral(const Rvec& action) const
  {
    Real ret = -quadraticTerm(action);
    return ret / 2;
  }

  Rvec getMean() const
  {
    return mean;
  }
  Rvec getMatrix() const
  {
    return matrix;
  }

  Real advantageVariance() const
  {
    if(!policy) return 0;
    Rvec PvarP(nA*nA, 0);
    for (uint64_t j=0; j<nA; ++j)
    for (uint64_t i=0; i<nA; ++i)
    for (uint64_t k=0; k<nA; ++k) {
      const uint64_t k1 = nA*j + k;
      const uint64_t k2 = nA*k + i;
      PvarP[nA*j+i]+= matrix[k1] * std::pow(policy->getStdev(i),2) * matrix[k2];
    }
    Real ret = quadMatMul(policy->getMean(), PvarP);
    for (uint64_t i=0; i<nA; ++i)
      ret += PvarP[nA*i+i] * std::pow(policy->getStdev(i), 2) / 2;
    return ret;
  }
};

} // end namespace smarties
#endif // smarties_Quadratic_advantage_h
