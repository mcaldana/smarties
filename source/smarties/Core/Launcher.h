//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Launcher_h
#define smarties_Launcher_h

#include "../Communicator.h"
#include "../Settings/ExecutionInfo.h"

namespace smarties
{

class Launcher: public Communicator
{
protected:
  ExecutionInfo & m_ExecutionInfo;

  std::vector<std::string> argsFiles;
  std::vector<uint64_t> argFilesStepsLimits;

  void initArgumentFileNames();
  void createGoRunDir(char* initDir, uint64_t folderID, MPI_Comm anvAppCom);
  std::vector<char*> readRunArgLst(const std::string& paramfile);

  void launch(const environment_callback_t & callback,
              const uint64_t workLoadID,
              const MPI_Comm envApplication_comm);

public:

  bool forkApplication( const environment_callback_t & callback );
  void runApplication( const environment_callback_t & callback );

  Launcher(Worker* const W, ExecutionInfo& D);
};

} // end namespace smarties
#endif // smarties_Launcher_h
