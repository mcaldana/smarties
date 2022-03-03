//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Parameters_h
#define smarties_Parameters_h

#include "Functions.h"
#include "../../Utils/MPIUtilities.h"

namespace smarties
{

struct Parameters;
using ParametersPtr_t = std::shared_ptr<Parameters>;

static inline nnReal* allocate_param(const uint64_t size, const Real mpiSize)
{
  // round up such that distributed ops can be vectorized on each rank:
  uint64_t extraSize = Utilities::roundUpSimd( std::ceil(size/mpiSize)) * mpiSize;
  return Utilities::allocate_ptr<nnReal>(extraSize);
}

struct Parameters
{
  const std::vector<uint64_t> nBiases, nWeights;
  std::vector<uint64_t> indBiases, indWeights;
  const uint64_t nParams, nLayers, mpiSize;
  mutable bool written = false;
  // array containing all parameters of network contiguously
  //(used by optimizer and for MPI reductions)
  nnReal* const params;

  ParametersPtr_t allocateEmptyAlike() const
  {
    return std::make_shared<Parameters>(nWeights, nBiases, mpiSize);
  }

  void broadcast(const MPI_Comm comm) const
  {
    MPI_Bcast(params, nParams, SMARTIES_MPI_NNVALUE_TYPE, 0, comm);
  }

  void copy(const ParametersPtr_t& tgt) const
  {
    assert(nParams == tgt->nParams);
    memcpy(params, tgt->params, nParams*sizeof(nnReal));
  }

  Parameters(const std::vector<uint64_t> _nWeights,
             const std::vector<uint64_t> _nBiases,
             const uint64_t _mpisize ) :
    nBiases(_nBiases), nWeights(_nWeights),
    nParams(_computeNParams(_nWeights, _nBiases)),
    nLayers(_nWeights.size()), mpiSize(_mpisize),
    params( allocate_param(nParams, _mpisize) )  { }

  ~Parameters() {
    if(params) free(params);
  }

  void reduceThreadsGrad(const std::vector<ParametersPtr_t>& grads) const
  {
    #pragma omp parallel num_threads(grads.size())
    {
      const uint64_t thrI = omp_get_thread_num(), thrN = omp_get_num_threads();
      assert( thrN == grads.size() && thrI < thrN );
      assert( nParams == grads[thrI]->nParams );
      const uint64_t shift = Utilities::roundUpSimd( nParams/ (Real)thrN );
      assert( thrN * shift >= nParams ); // ensure coverage
      const nnReal *const src = grads[thrI]->params;
            nnReal *const dst = params;
      for(uint64_t i=0; i<thrN; ++i)
      {
        const uint64_t turn = (thrI + i) % thrN;
        const uint64_t start = turn * shift;
        const uint64_t end = std::min(nParams, (turn+1)*shift);
        //#pragma omp critical
        //{ cout<<turn<<" "<<start<<" "<<end<<" "<<thrI<<" "
        //      <<thrN<<" "<<shift<<" "<<nParams<<endl; fflush(0); }
        if(grads[thrI]->written) {
          #pragma omp simd aligned(dst, src : VEC_WIDTH)
          for(uint64_t j=start; j<end; ++j) {
            assert( Utilities::isValidValue(src[j]) );
            dst[j] += src[j];
            #ifndef NDEBUG
              //gradMagn[thrI] += src[j]*src[j];
            #endif
          }
        }
        #pragma omp barrier
      }
      grads[thrI]->clear();
    }
    //cout<<endl;
    #ifndef NDEBUG
    //cout<<"Grad magnitudes:"<<print(gradMagn)<<endl;
    #endif
  }

  long double compute_weight_norm() const
  {
    long double sumWeights = 0;
    #pragma omp parallel for schedule(static) reduction(+:sumWeights)
    for (uint64_t w=0; w<nParams; ++w) sumWeights += std::pow(params[w],2);
    return std::sqrt(sumWeights);
  }
  long double compute_weight_L1norm() const
  {
    long double sumWeights = 0;
    #pragma omp parallel for schedule(static) reduction(+:sumWeights)
    for (uint64_t w=0; w<nParams; ++w) sumWeights += std::fabs(params[w]);
    return sumWeights;
  }
  long double compute_weight_dist(const ParametersPtr_t& TGT) const
  {
    long double dist = 0;
    #pragma omp parallel for schedule(static) reduction(+ : dist)
    for(uint64_t w=0; w<nParams; ++w) dist += std::pow(params[w]-TGT->params[w], 2);
    return std::sqrt(dist);
  }

  void clear() const
  {
    std::memset(params, 0, nParams*sizeof(nnReal));
    written = false;
  }
  void set(const nnReal val) const
  {
    #pragma omp parallel for schedule(static)
    for(uint64_t j=0; j<nParams; ++j) params[j] = val;
  }

  nnReal* W(const uint64_t layerID) const {
    assert(layerID < nLayers);
    return params + indWeights[layerID];
  }
  nnReal* B(const uint64_t layerID) const {
    assert(layerID < nLayers);
    return params + indBiases[layerID];
  }
  uint64_t NW(const uint64_t layerID) const {
    assert(layerID < nLayers);
    return nWeights[layerID];
  }
  uint64_t NB(const uint64_t layerID) const {
    assert(layerID < nLayers);
    return nBiases[layerID];
  }

private:
  //each layer requests a certain number of parameters, here compute contiguous
  //memory required such that each layer gets an aligned pointer to both
  //its first bias and and first weight, allowing SIMD ops on all layers
  uint64_t _computeNParams(std::vector<uint64_t> _nWeights, std::vector<uint64_t> _nBiases)
  {
    assert(_nWeights.size() == _nBiases.size());
    const uint64_t _nLayers = _nWeights.size();
    uint64_t nTotPara = 0;
    indBiases  = std::vector<uint64_t>(_nLayers, 0);
    indWeights = std::vector<uint64_t>(_nLayers, 0);
    for(uint64_t i=0; i<_nLayers; ++i) {
      indWeights[i] = nTotPara;
      nTotPara += Utilities::roundUpSimd(_nWeights[i]);
      indBiases[i] = nTotPara;
      nTotPara += Utilities::roundUpSimd( _nBiases[i]);
    }
    //printf("Weight sizes:[%s] inds:[%s] Bias sizes:[%s] inds[%s] Total:%u\n",
    //  print(_nWeights).c_str(), print(indWeights).c_str(),
    //  print(_nBiases).c_str(), print(indBiases).c_str(), nTotPara);
    return nTotPara;
  }
};

inline std::vector<ParametersPtr_t> allocManyParams(const ParametersPtr_t& W,
                                                    const uint64_t populationSize)
{
  std::vector<ParametersPtr_t> ret(populationSize, nullptr);
  for(uint64_t i=0; i<populationSize; ++i) ret[i] = W->allocateEmptyAlike();
  return ret;
}

} // end namespace smarties
#endif // smarties_Quadratic_term_h
