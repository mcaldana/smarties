//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_ThreadSafeVec_h
#define smarties_ThreadSafeVec_h

#include "../Settings/Definitions.h"
#include <cassert>
#include <memory>
#include <omp.h>

namespace smarties
{

template<typename T>
struct THRvec
{
  uint64_t nThreads;
  const T initial;
  std::vector<std::unique_ptr<T>> m_v;

  THRvec(const uint64_t size, const T init=T()) : nThreads(size), initial(init)
  {
    m_v.resize(nThreads);
    #pragma omp parallel for num_threads(nThreads) schedule(static, 1)
    for(uint64_t i=0; i<nThreads; ++i) m_v[i] = std::make_unique<T>(initial);
  }

  THRvec(const THRvec&c) = delete;

  void resize(const uint64_t N)
  {
    if(N == nThreads) return;

    m_v.resize(N);
    nThreads = N;
    #pragma omp parallel for schedule(static, 1)
    for(uint64_t i=0; i<N; ++i) {
      if(m_v[i]) continue;
      m_v[i] = std::make_unique<T>(initial);
    }
  }

  uint64_t size() const { return nThreads; }

  T& operator[] (const uint64_t i) const
  {
    assert(m_v[i]);
    return * m_v[i].get();
  }
};

} // end namespace smarties
#endif // smarties_Settings_h
