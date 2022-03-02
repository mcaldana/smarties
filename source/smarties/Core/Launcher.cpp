//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Launcher.h"
#include "Worker.h"
#include "../Utils/Warnings.h"
#include "../Utils/SocketsLib.h"
#include "../Utils/LauncherUtilities.h"
#include "../Utils/SstreamUtilities.h"

#include <omp.h>
#include <fstream>

namespace smarties
{

Launcher::Launcher(Worker* const W, ExecutionInfo& D) :
  Communicator(W, D.generators[0], D.bTrain), m_ExecutionInfo(D)
{
  initArgumentFileNames();
}

bool Launcher::forkApplication(const environment_callback_t & callback)
{
  //const uint64_t nThreads = m_ExecutionInfo.nThreads;
  const uint64_t nOwnEnvs = m_ExecutionInfo.nOwnedEnvironments;
  const uint64_t totNumEnvs = m_ExecutionInfo.nEnvironments;
  const uint64_t totNumProcess = MPICommSize(m_ExecutionInfo.world_comm);

  bool isChild = false;
  // TODO: reinstate the omp thread stuff for mpi implementations that do
  // process binding by default, avoiding pybind11 breaking.
  //#pragma omp parallel num_threads(nThreads)
  for(int i = 0; i < (int) nOwnEnvs; ++i)
  {
    //const int thrID = omp_get_thread_num(), thrN = omp_get_num_threads();
    const int thrID = 0, thrN = 1;
    const int tgtCPU =  ( ( (-1-i) % thrN ) + thrN ) % thrN;
    const int workloadID = i + totNumEnvs * totNumProcess;
    //assert(nThreads == (uint64_t) omp_get_num_threads());
    if( thrID==tgtCPU and isChild == false)
      //#pragma omp critical
      {
        std::cout << "New fork" << std::endl;
        const int success = fork();
        if ( success == -1 ) die("Failed to fork.");
        if ( success ==  0 ) {
          isChild = true;
          usleep(10); // IDK, wait for parent to create socket file to be sure
          //warn("entering SOCKET_clientConnect");
          m_Sockets.server = SOCKET_clientConnect();
          if(m_Sockets.server == -1) die("Failed to connect to parent process.");
          //warn("exiting SOCKET_clientConnect");
          launch(callback, workloadID, MPI_COMM_SELF);
        } else assert(isChild == false);
      }
  }

  if(not isChild) {
    //warn("entering SOCKET_serverConnect");
    SOCKET_serverConnect(nOwnEnvs, m_Sockets.clients);
    //warn("exiting SOCKET_serverConnect");
  }
  return isChild;
}

void Launcher::runApplication(const environment_callback_t & callback )
{
  const int64_t thisWorkerGroupID = m_ExecutionInfo.thisWorkerGroupID;
  const MPI_Comm envApplication_comm = m_ExecutionInfo.environment_app_comm;
  if(thisWorkerGroupID<0) die("Error in setup of envApplication_comm");
  assert(envApplication_comm != MPI_COMM_NULL);
  launch(callback, thisWorkerGroupID, envApplication_comm);
}

void Launcher::launch(const environment_callback_t & callback,
                      const uint64_t workLoadID,
                      const MPI_Comm envApplication_comm)
{
  const uint64_t appSize = MPICommSize(envApplication_comm);
  const uint64_t appRank = MPICommRank(envApplication_comm);
  // app only needs lower level functionalities:
  // ie. send state, recv action, specify state/action spaces properties...
  Communicator* const commptr = static_cast<Communicator*>(this);
  assert(commptr != nullptr);

  while(true)
  {
    char currDirectory[512];
    // create dedicated directory for the process:
    createGoRunDir(currDirectory, workLoadID, envApplication_comm);

    uint64_t settInd = 0;
    for(size_t i=0; i<argsFiles.size(); ++i)
      if(m_globalTstepCounter >= argFilesStepsLimits[i]) settInd = i;

    assert(argFilesStepsLimits.size() > settInd+1 && m_ExecutionInfo.nEnvironments > 0);
    uint64_t numTstepSett = argFilesStepsLimits[settInd+1] - m_globalTstepCounter;
    numTstepSett = numTstepSett * appSize / m_ExecutionInfo.nEnvironments;
    std::vector<char*> args = readRunArgLst(argsFiles[settInd]);

    // process stdout file descriptor, so that we can revert:
    std::pair<int, fpos_t> currOutputFdescriptor;
    if(m_ExecutionInfo.redirectAppStdoutToFile)
      redirect_stdout_init(currOutputFdescriptor, appRank);

    callback(commptr, envApplication_comm, args.size()-1, args.data());

    if(m_ExecutionInfo.redirectAppStdoutToFile)
      redirect_stdout_finalize(currOutputFdescriptor);

    for(size_t i = 0; i < args.size()-1; ++i) delete[] args[i];
    chdir(currDirectory);  // go to original directory
    if(m_bTrainIsOver) break;
  }
}

void Launcher::initArgumentFileNames()
{
  // appSettings is a list of text files separated by commas
  // e.g. settings1.txt,settings2.txt,...
  argsFiles = split(m_ExecutionInfo.appSettings, ',');
  if(argsFiles.size() == 0) {
    if(m_ExecutionInfo.appSettings != "")
      _die("error in splitting appSettings %s", m_ExecutionInfo.appSettings.c_str());
    argsFiles.push_back("");
  }
  assert(argsFiles.size() > 0);

  // nStepPerFile is a list of numbers representing how many timesteps we should
  // run with a settings file before switching to the next one. e.g. 1000,...
  // Here '0' means : run the settings file for ever
  // If empty, we assume the settings file should be run for ever
  if(m_ExecutionInfo.nStepPappSett == "") m_ExecutionInfo.nStepPappSett = "0";
  std::vector<std::string> stepNmbrs = split(m_ExecutionInfo.nStepPappSett, ',');
  using Utilities::vec2string;
  if(argsFiles.size() != stepNmbrs.size())
    _die("mismatch in sizes: argsFiles=%s stepNmbrs=%s",
      vec2string(argsFiles).c_str(), vec2string(stepNmbrs).c_str());

  argFilesStepsLimits = std::vector<uint64_t>(argsFiles.size(), 0);
  argFilesStepsLimits[0] = 0; // first settings file is used from step 0
  for (size_t i=1; i<stepNmbrs.size(); ++i)
    argFilesStepsLimits[i]= argFilesStepsLimits[i-1] +std::stol(stepNmbrs[i-1]);
  //last setup used for ever:
  argFilesStepsLimits.push_back(std::numeric_limits<uint64_t>::max());
  assert(argFilesStepsLimits.size() == argsFiles.size() + 1);
}

void Launcher::createGoRunDir(char* initDir, uint64_t folderID, MPI_Comm envAppCom)
{
  char newDir[1024];
  getcwd(initDir, 512);
  struct stat fileStat;
  unsigned long iter = 0;

  while(true)
  {
    snprintf(newDir, 1024, "%s/%s_%03u_%05u",
        initDir, "simulation", (unsigned) folderID, (unsigned) iter);
    if ( stat(newDir, &fileStat) >= 0 ) iter++; // directory already exists
    else
    {
      if( MPICommSize(envAppCom)>1 ) MPI_Barrier(envAppCom);

      if( MPICommRank(envAppCom)<1 ) // app's root sets up dir
      {
        mkdir(newDir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if(m_ExecutionInfo.setupFolder != "") //copy any file in the setup dir
        {
          if (copy_from_dir(("../"+m_ExecutionInfo.setupFolder).c_str()) != 0 )
            _die("Error in copy from dir %s\n", m_ExecutionInfo.setupFolder.c_str());
        }
      }

      if( MPICommSize(envAppCom)>1 ) MPI_Barrier(envAppCom);

      chdir(newDir);
      break;
    }
  }
}

std::vector<char*> Launcher::readRunArgLst(const std::string& paramFile)
{
  std::vector<char*> args;
  const auto addArg = [&](const std::string & token)
  {
    char *arg = new char[token.size() + 1];
    std::copy(token.begin(), token.end(), arg);  // write into char array
    arg[token.size()] = '\0';
    args.push_back(arg);
  };

  // first put argc argv into args:
  for(int i=0; i<m_ExecutionInfo.argc; ++i)
    if(m_ExecutionInfo.argv[i] != nullptr)
      addArg ( std::string( m_ExecutionInfo.argv[i] ) );

  if (paramFile != "")
  {
    std::ifstream t( ("../"+paramFile).c_str() );
    std::string linestr((std::istreambuf_iterator<char>(t)),
                         std::istreambuf_iterator<char>());
    if(linestr.size() == 0) die("did not find parameter file");
    std::istringstream iss(linestr); // params file is read into iss
    std::string token;
    while(iss >> token)
    {
      // If one runs an executable and provides a runarg like ./exec 'foo bar'
      // then `foo bar' is put in its entirety in argv[1]. However, when we ask
      // user to write a settingsfile, apostrophes are read as characters, not
      // special symbols, therefore we must do the following workaround to put
      // anything that is written between parenteses in a single argv entry.
      if(token[0]=='\'')
      {
        token.erase(0, 1); // remove apostrophe ( should have been read as \' )
        std::string continuation;
        while(token.back() != '\'') { // if match apostrophe, we are done
          if(!(iss >> continuation)) die("missing matching apostrophe");
          token += " " + continuation; // add next line to argv entry
        }
        token.erase(token.end()-1, token.end()); // remove trailing apostrophe
      }

      addArg ( token );
    }
  }

  args.push_back(nullptr); // push back nullptr as last entry
  return args; // remember to deallocate it!
}

}
