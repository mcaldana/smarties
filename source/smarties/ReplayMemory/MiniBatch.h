//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_MiniBatch_h
#define smarties_MiniBatch_h

#include "Episode.h"

namespace smarties
{

struct MiniBatch
{
  const uint64_t size;

  MiniBatch(const uint64_t _size) : size(_size)
  {
    episodes.resize(size);
    begTimeStep.resize(size);
    endTimeStep.resize(size);
    sampledTimeStep.resize(size);
    S.resize(size); R.resize(size); PERW.resize(size);
  }
  MiniBatch(MiniBatch && p) = default;
  MiniBatch& operator=(MiniBatch && p) = delete;
  MiniBatch(const MiniBatch &p) = delete;
  MiniBatch& operator=(const MiniBatch &p) = delete;

  std::vector<Episode*> episodes;
  std::vector<int64_t> begTimeStep;
  std::vector<int64_t> endTimeStep;
  std::vector<int64_t> sampledTimeStep;
  int64_t sampledBegStep(const uint64_t b) const { return begTimeStep[b]; }
  int64_t sampledEndStep(const uint64_t b) const { return endTimeStep[b]; }
  int64_t sampledTstep(const uint64_t b) const { return sampledTimeStep[b]; }
  int64_t sampledNumSteps(const uint64_t b) const {
    assert(begTimeStep.size() > b);
    assert(endTimeStep.size() > b);
    return endTimeStep[b] - begTimeStep[b];
  }
  int64_t mapTime2Ind(const uint64_t b, const int64_t t) const
  {
    assert(begTimeStep.size() >  b and begTimeStep[b]     <= t);
    //ind is mapping from time stamp along trajectoy and along alloc memory
    return t - begTimeStep[b];
  }
  int64_t mapInd2Time(const uint64_t b, const int64_t k) const
  {
    assert(begTimeStep.size() > b);
    //ind is mapping from time stamp along trajectoy and along alloc memory
    return k + begTimeStep[b];
  }

  // episodes | time steps | dimensionality
  std::vector< std::vector< NNvec > > S;  // scaled state
  std::vector< std::vector< Real  > > R;  // scaled reward
  std::vector< std::vector< nnReal> > PERW;  // prioritized sampling

  Episode& getEpisode(const uint64_t b) const
  {
    return * episodes[b];
  }

  NNvec& state(const uint64_t b, const int64_t t)
  {
    return S[b][mapTime2Ind(b, t)];
  }
  const NNvec& state(const uint64_t b, const int64_t t) const
  {
    return S[b][mapTime2Ind(b, t)];
  }
  Real& reward(const uint64_t b, const int64_t t)
  {
    return R[b][mapTime2Ind(b, t)];
  }
  const Real& reward(const uint64_t b, const int64_t t) const
  {
    return R[b][mapTime2Ind(b, t)];
  }
  nnReal& PERweight(const uint64_t b, const int64_t t)
  {
    return PERW[b][mapTime2Ind(b, t)];
  }
  const nnReal& PERweight(const uint64_t b, const int64_t t) const
  {
    return PERW[b][mapTime2Ind(b, t)];
  }
  const Rvec& action(const uint64_t b, const uint64_t t) const
  {
    return episodes[b]->actions[t];
  }
  const Rvec& mu(const uint64_t b, const uint64_t t) const
  {
    return episodes[b]->policies[t];
  }
  nnReal& returnEstimate(const uint64_t b, const uint64_t t) const
  {
    return episodes[b]->returnEstimator[t];
  }
  std::vector<nnReal> returnEstimates(const uint64_t dt = 0) const
  {
    std::vector<nnReal> ret(size, 0);
    for(uint64_t b=0; b<size; ++b) {
      const auto t = sampledTstep(b);
      assert(t >= (int64_t) dt);
      ret[b] = episodes[b]->returnEstimator[t-dt];
    }
    return ret;
  }
  nnReal& value(const uint64_t b, const uint64_t t) const
  {
    return episodes[b]->stateValue[t];
  }
  nnReal& advantage(const uint64_t b, const uint64_t t) const
  {
    return episodes[b]->actionAdvantage[t];
  }


  bool isTerminal(const uint64_t b, const uint64_t t) const
  {
    return episodes[b]->isTerminal(t);
  }
  std::vector<int> isNextTerminal() const // pybind will not like vector of bool
  {
    std::vector<int> ret(size, 0);
    for(uint64_t b=0; b<size; ++b)
      ret[b] = episodes[b]->isTerminal(sampledTstep(b) + 1);
    return ret;
  }
  bool isTruncated(const uint64_t b, const uint64_t t) const
  {
    return episodes[b]->isTruncated(t);
  }
  std::vector<int> isNextTruncated() const //pybind will not like vector of bool
  {
    std::vector<int> ret(size, 0);
    for(uint64_t b=0; b<size; ++b)
      ret[b] = episodes[b]->isTruncated(sampledTstep(b)+1);
    return ret;
  }
  uint64_t nTimeSteps(const uint64_t b) const
  {
    return episodes[b]->nsteps();
  }
  uint64_t nDataSteps(const uint64_t b) const //terminal/truncated state not actual data
  {
    return episodes[b]->ndata();
  }
  uint64_t indCurrStep(const uint64_t b=0) const
  {
    assert(episodes[b]->nsteps() > 0);
    return episodes[b]->nsteps() - 1;
  }

  void setMseDklImpw(const uint64_t b, const uint64_t t, // batch id and time id
    const Fval E, const Fval D, const Fval W,    // error, dkl, offpol weight
    const Fval C, const Fval invC) const         // bounds of offpol weight
  {
    getEpisode(b).updateCumulative_atomic(t, E, D, W, C, invC);
  }

  void setValues(const uint64_t b, const uint64_t t, const Fval V) const
  {
    return setValues(b, t, V, V);
  }
  void setValues(const uint64_t b, const uint64_t t, const Fval V, const Fval Q) const
  {
    getEpisode(b).updateValues_atomic(t, V, Q);
  }
  void appendValues(const Fval V) const
  {
    return appendValues(V, V);
  }
  void appendValues(const Fval V, const Fval Q) const
  {
    getEpisode(0).stateValue.push_back(V);
    getEpisode(0).actionAdvantage.push_back(Q-V);
    assert(getEpisode(0).nsteps() == getEpisode(0).actionAdvantage.size());
    assert(getEpisode(0).nsteps() == getEpisode(0).stateValue.size());
    assert(size == 1 && "This should only be called by in-progress episodes");
  }

  template<typename T>
  void setAllValues(const T& Vs, const T& Qs) const
  {
    assert(Vs.size() == size and Qs.size() == size);
    #pragma omp parallel for schedule(static)
    for(uint64_t b=0; b<size; ++b) setValues(b, sampledTstep(b), Vs[b], Qs[b]);
  }

  template<typename T>
  void updateAllLastStepValues(const std::vector<T>& values) const
  {
    assert(values.size() == size);
    #pragma omp parallel for schedule(static)
    for(uint64_t b=0; b<size; ++b) {
      const auto t = sampledTstep(b);
      if( isTruncated(b, t+1) ) setValues(b, t+1, values[b]);
      else if( isTerminal (b, t+1) ) setValues(b, t+1, 0);
    }
  }

  template<typename T>
  void setAllMseDklImpw(const std::vector<T>& deltaVal,
                        const std::vector<T>& DKLs,
                        const std::vector<T>& rhos,
                        const Fval C, const Fval invC) const
  {
    assert(deltaVal.size() ==size && DKLs.size() ==size && rhos.size() ==size);
    for(uint64_t b=0; b<size; ++b)
      setMseDklImpw(b, sampledTstep(b), deltaVal[b], DKLs[b], rhos[b], C,invC);
  }

  void resizeStep(const uint64_t b, const uint64_t nSteps)
  {
    assert( S.size()>b and R.size()>b);
    S[b].resize(nSteps); R[b].resize(nSteps); PERW[b].resize(nSteps);
  }
};
} // namespace smarties
#endif // smarties_MiniBatch_h
