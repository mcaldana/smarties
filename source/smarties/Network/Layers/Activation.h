//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Activation_h
#define smarties_Activation_h

#include "Functions.h"

namespace smarties
{

struct Activation;
using ActivationPtr_t = std::unique_ptr<Activation>;

struct Activation
{
  uint64_t _nOuts(std::vector<uint64_t> _sizes, std::vector<uint64_t> _bOut)
  {
    assert(_sizes.size() == _bOut.size() && (size_t) nLayers == _bOut.size());
    uint64_t ret = 0;
    for(uint64_t i=0; i<_bOut.size(); ++i) if(_bOut[i]) ret += _sizes[i];
    if(!ret) die("err nOutputs");
    return ret;
  }
  uint64_t _nInps(std::vector<uint64_t> _sizes, std::vector<uint64_t> _bInp)
  {
    assert(_sizes.size() == _bInp.size() && (size_t) nLayers == _bInp.size());
    uint64_t ret = 0;
    for(uint64_t i=0; i<_bInp.size(); ++i) if(_bInp[i]) ret += _sizes[i];
    return ret;
  }

  Activation(const std::vector<uint64_t>& _sizes,
             const std::vector<uint64_t>& _bOut,
             const std::vector<uint64_t>& _bInp):
    nLayers(_sizes.size()), nOutputs(_nOuts(_sizes,_bOut)), nInputs(_nInps(_sizes,_bInp)),
    sizes(_sizes), output(_bOut), input(_bInp),
    suminps(Utilities::allocate_vec(_sizes)),
    outvals(Utilities::allocate_vec(_sizes)),
    errvals(Utilities::allocate_vec(_sizes)) {
    assert(suminps.size()== (size_t) nLayers);
    assert(outvals.size()== (size_t) nLayers);
    assert(errvals.size()== (size_t) nLayers);
  }

  ~Activation() {
    for(auto& p : suminps) if(p) free(p);
    for(auto& p : outvals) if(p) free(p);
    for(auto& p : errvals) if(p) free(p);
  }

  template<typename T>
  void setInput(const std::vector<T>& inp) const
  {
    assert( (size_t) nInputs == inp.size());
    for(int j=0; j<nInputs; ++j)
      assert(!std::isnan(inp[j]) && !std::isinf(inp[j]));
    int k=0;
    for(int i=0; i<nLayers; ++i) if(input[i]) {
      std::copy(&inp[k], &inp[k]+sizes[i], outvals[i]);
      //memcpy(outvals[i], &inp[k], sizes[i]*sizeof(nnReal));
      k += sizes[i];
    }
    assert(k == nInputs);
  }
  std::vector<Real> getInput() const
  {
    std::vector<Real> ret(nInputs);
    int k=0;
    for(int i=0; i<nLayers; ++i) if(input[i]) {
      std::copy(outvals[i], outvals[i]+sizes[i], &ret[k]);
      //memcpy(&ret[k], outvals[i], sizes[i]*sizeof(nnReal));
      k += sizes[i];
    }
    assert(k == nInputs);
    return ret;
  }

  std::vector<Real> getInputGradient(const uint64_t ID) const
  {
    assert(written == true);
    std::vector<Real> ret(sizes[ID]);
    std::copy(errvals[ID], errvals[ID]+sizes[ID], &ret[0]);
    //memcpy(&ret[0], errvals[ID], sizes[ID]*sizeof(nnReal));
    return ret;
  }

  template<typename T>
  void setOutputDelta(const std::vector<T>& delta) const
  {
    assert( (size_t) nOutputs == delta.size()); //alternative not supported
    for(int j=0; j<nOutputs; ++j)
      assert(!std::isnan(delta[j]) && !std::isinf(delta[j]));
    int k=0;
    for(int i=0; i<nLayers; ++i) if(output[i]) {
      std::copy(&delta[k], &delta[k]+sizes[i], errvals[i]);
      //memcpy(errvals[i], &delta[k], sizes[i]*sizeof(nnReal));
      k += sizes[i];
    }
    assert(k == nOutputs);
    written = true;
  }

  template<typename T>
  void addOutputDelta(const std::vector<T>& delta) const
  {
    assert( (size_t) nOutputs == delta.size()); //alternative not supported
    int k=0;
    for(int i=0; i<nLayers; ++i) if(output[i])
      for (uint64_t j=0; j<sizes[i]; ++j, ++k) errvals[i][j] += delta[k];
    assert(k == nOutputs);
    written = true;
  }

  std::vector<nnReal> getOutputDelta() const
  {
    assert(written == true);
    std::vector<nnReal> ret(nOutputs);
    int k=0;
    for(int i=0; i<nLayers; ++i) if(output[i]) {
      std::copy(errvals[i], errvals[i]+sizes[i], &ret[k]);
      //memcpy(&ret[k], errvals[i], sizes[i]*sizeof(nnReal));
      k += sizes[i];
    }
    assert(k == nOutputs);
    return ret;
  }

  std::vector<Real> getOutput() const
  {
    assert(written == true);
    std::vector<Real> ret(nOutputs);
    int k=0;
    for(int i=0; i<nLayers; ++i) if(output[i]) {
      std::copy(outvals[i], outvals[i]+sizes[i], &ret[k]);
      //memcpy(&ret[k], outvals[i], sizes[i]*sizeof(nnReal));
      k += sizes[i];
    }
    for(int j=0; j<nOutputs; ++j)
      assert(!std::isnan(ret[j]) && !std::isinf(ret[j]));
    assert(k == nOutputs);
    return ret;
  }

  void clearOutput() const
  {
    for(int i=0; i<nLayers; ++i) {
      assert(outvals[i]);
      memset( outvals[i], 0, Utilities::roundUpSimd(sizes[i])*sizeof(nnReal) );
    }
  }

  void clearErrors() const
  {
    for(int i=0; i<nLayers; ++i) {
      assert(errvals[i]);
      memset( errvals[i], 0, Utilities::roundUpSimd(sizes[i])*sizeof(nnReal) );
    }
  }

  void clearInputs() const
  {
    for(int i=0; i<nLayers; ++i) {
      assert(suminps[i]);
      memset( suminps[i], 0, Utilities::roundUpSimd(sizes[i])*sizeof(nnReal) );
    }
  }

  nnReal* X(const int layerID) const
  {
    assert(layerID < nLayers);
    return suminps[layerID];
  }
  nnReal* Y(const int layerID) const
  {
    assert(layerID < nLayers);
    return outvals[layerID];
  }
  nnReal* E(const int layerID) const
  {
    assert(layerID < nLayers);
    return errvals[layerID];
  }

  const int nLayers, nOutputs, nInputs;
  const std::vector<uint64_t> sizes, output, input;
  //contains all inputs to each neuron (inputs to network input layer is empty)
  const std::vector<nnReal*> suminps;
  //contains all neuron outputs that will be the incoming signal to linked layers (outputs of input layer is network inputs)
  const std::vector<nnReal*> outvals;
  //deltas for each neuron
  const std::vector<nnReal*> errvals;
  mutable bool written = false;
};

} // end namespace smarties
#endif // smarties_Quadratic_term_h
