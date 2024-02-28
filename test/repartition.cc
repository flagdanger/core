#include <gmi_mesh.h>
#include <apf.h>
#include <apfMesh2.h>
#include <apfMDS.h>
#include <PCU.h>
#include <PCUObj.h>
#include <lionPrint.h>
#include <parma.h>
#include <apfZoltan.h>
#include <apfPartition.h>
#include <pcu_util.h>
#include <cstdlib>

namespace {

const char* modelFile = 0;
const char* meshFile = 0;
const char* outFile = 0;
int inputPartCount = 1;

void freeMesh(apf::Mesh* m)
{
  m->destroyNative();
  apf::destroyMesh(m);
}

struct SwitchToOriginalResult {
  bool isOriginal;
  MPI_Comm groupComm;
};

SwitchToOriginalResult switchToOriginals(pcu::PCU &worldPCU)
{
  apf::Contract contract(inputPartCount, worldPCU.Peers());
  int self = worldPCU.Self();
  int group;
  int groupRank;
  bool isOriginal = contract.isValid(self);
  if (isOriginal) {
    group = 0;
    groupRank = contract(self);
  } else {
    group = 1;
    /* MPI standard says they will be sorted by old rank,
       no need to spend time computing a good contiguous number */
    groupRank = 0;
  }
  MPI_Comm groupComm;
  MPI_Comm_split(MPI_COMM_WORLD, group, groupRank, &groupComm);
  //pcu::PCU PCUComm = new pcu::PCU(groupComm);
  return SwitchToOriginalResult{isOriginal, groupComm};
}

void switchToAll()
{
  MPI_Comm prevComm = PCU_Get_Comm();
  PCU_Switch_Comm(MPI_COMM_WORLD);
  MPI_Comm_free(&prevComm);
  PCU_Barrier();
}

void getConfig(int argc, char** argv)
{
  if ( argc != 5 ) {
    if ( !PCU_Comm_Self() )
      printf("Usage: mpirun -n <outPartCount> %s"
             " <model> <inPartCount> <inMesh> <outMesh>\n"
             "Increase the part count of inMesh from inPartCount to outPartCount.\n"
             "Unlike the [z]split tool, outPartCount does not have to be an integer\n"
             "multiple of inPartCount.\n",
             argv[0]);
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }
  modelFile = argv[1];
  inputPartCount = atoi(argv[2]);
  meshFile = argv[3];
  outFile = argv[4];
  PCU_ALWAYS_ASSERT(inputPartCount <= PCU_Comm_Peers());
}

void balance(apf::Mesh2* m)
{
  apf::MeshTag* weights = m->createDoubleTag("zoltan_weight", 1);
  {
    apf::MeshIterator* it = m->begin(m->getDimension());
    apf::MeshEntity* e;
    double value = 1.0;
    while ((e = m->iterate(it)))
      m->setDoubleTag(e, weights, &value);
    m->end(it);
  }
  apf::Balancer* b = apf::makeZoltanBalancer(m, apf::RCB, apf::REPARTITION, false);
  b->balance(weights, 1.1);
  m->destroyTag(weights);
  delete b;
}

}

/**
int main(int argc, char** argv)
{
  MPI_Init(&argc,&argv);
  PCU_Comm_Init();
  lion_set_verbosity(1);
  gmi_register_mesh();
  getConfig(argc,argv);
  gmi_model* g = gmi_load(modelFile);
  apf::Mesh2* m = 0;
  bool isOriginal = switchToOriginals();
  if (isOriginal)
    m = apf::loadMdsMesh(g, meshFile);
  switchToAll();
  m = apf::expandMdsMesh(m, g, inputPartCount);
  balance(m);
  Parma_PrintPtnStats(m, "");
  m->writeNative(outFile);
  freeMesh(m);
  PCU_Comm_Free();
  MPI_Finalize();
}
**/
int main(int argc, char** argv){
  MPI_Init(&argc, &argv);
  pcu::PCU *originalPCU = new pcu::PCU(MPI_COMM_WORLD);
  struct SwitchToOriginalResult result = switchToOriginals(*originalPCU);
  bool isOriginal = result.isOriginal;
  pcu::PCU *expandedPCU = new pcu::PCU(result.groupComm);

  lion_set_verbosity(1);
  gmi_register_mesh();
  getConfig(argc,argv);
  gmi_model* g = gmi_load(modelFile);
  apf::Mesh2* originalMesh = nullptr;
  apf::Mesh2* expandedMesh = nullptr;

  if (isOriginal)
    originalMesh = apf::loadMdsMesh(g, meshFile, expandedPCU);
  expandedPCU = pcu::PCU_GetGlobal();
  expandedPCU->Barrier();
  expandedMesh = apf::expandMdsMesh(originalMesh, g, inputPartCount, expandedPCU);

  
  balance(expandedMesh);
  Parma_PrintPtnStats(expandedMesh, "");
  expandedMesh->writeNative(outFile);
  freeMesh(originalMesh);
  freeMesh(expandedMesh);
  MPI_Finalize();
}

