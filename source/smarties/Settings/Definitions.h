//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Definitions_h
#define smarties_Definitions_h

#include <vector>
#include <array>
#include <limits>
#include <cstddef>
#include <functional>

namespace smarties
{

////////////////////////////////////////////////////////////////////////////////
#if 1 // MAIN CODE PRECISION
using Real = double;
#define SMARTIES_MPI_VALUE_TYPE MPI_DOUBLE
#else
using Real = float;
#define SMARTIES_MPI_VALUE_TYPE MPI_FLOAT
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef SINGLE_PREC // NETWORK PRECISION
  #define SMARTIES_gemv cblas_dgemv
  #define SMARTIES_gemm cblas_dgemm
  using nnReal = double;
  #define SMARTIES_MPI_NNVALUE_TYPE MPI_DOUBLE
  #define SMARTIES_EXP_CUT 16 //prevent under/over flow with exponentials
#else
  #define SMARTIES_gemv cblas_sgemv
  #define SMARTIES_gemm cblas_sgemm
  #define SMARTIES_MPI_NNVALUE_TYPE MPI_FLOAT
  using nnReal = float;
  #define SMARTIES_EXP_CUT 8 //prevent under/over flow with exponentials
#endif
////////////////////////////////////////////////////////////////////////////////
// Data format for storage in memory buffer. Switch to float for example for
// Atari where the memory buffer is in the order of GBs.
#if 1
using Fval = float;
#define SMARTIES_MPI_Fval MPI_FLOAT
#else
using Fval = double;
#define SMARTIES_MPI_Fval MPI_DOUBLE
#endif

using Fvec = std::vector<Fval>;
using Rvec = std::vector<Real>;
using NNvec = std::vector<nnReal>;
using LDvec = std::vector<long double>;

struct Conv2D_Descriptor
{
  uint64_t inpFeatures, inpY, inpX; //input image: channels, x:width, y:height:
  uint64_t outFeatures, outY, outX; // output image
  uint64_t filterx, filtery; // tot size : inpFeatures*outFeatures*filterx*filtery
  uint64_t stridex, stridey;
  uint64_t paddinx, paddiny;
};

}
#endif
