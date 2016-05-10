/****************************************************************************** 

  (c) 2004-2016 Scientific Computation Research Center, 
      Rensselaer Polytechnic Institute. All rights reserved.
  
  This work is open source software, licensed under the terms of the
  BSD license as described in the LICENSE file in the top-level directory.
 
*******************************************************************************/
#include "pumi.h"
#include <mpi.h>
#include <apf.h>
#include <apfMesh2.h>
#include <apfMDS.h>
#include <PCU.h>
#include <vector>
#include <algorithm>

using std::vector;
int pumi_gent_getdim(pGeomEnt ge)
{
    return gmi_dim(pumi::instance()->model, ge);
}

int pumi_gent_getid(pGeomEnt ge)
{
  return gmi_tag(pumi::instance()->model, ge);
}

int pumi_ment_getdim(pMeshEnt e)
{
  return apf::getDimension(pumi::instance()->mesh, e);
}

int pumi_ment_getnumadj(pMeshEnt e, int target_dim)
{
  int ent_dim= apf::getDimension(pumi::instance()->mesh, e);
  if (ent_dim<target_dim) // upward
  {
    apf::Adjacent adjacent;
    pumi::instance()->mesh->getAdjacent(e,target_dim,adjacent);      
    return adjacent.getSize();
  }
  else if (ent_dim>target_dim)
  {
    apf::Downward adjacent;
    return pumi::instance()->mesh->getDownward(e,target_dim,adjacent); 
  }
  if (!pumi_rank()) std::cout<<"[pumi error] "<<__func__<<": invalid target dimension "<<target_dim<<"\n";
  return 0;
}

void pumi_ment_getadj(pMeshEnt e, int target_dim, std::vector<pMeshEnt>& vecAdjEnt)
{
  int num_adj, ent_dim= apf::getDimension(pumi::instance()->mesh, e);
  if (ent_dim<target_dim) // upward
  {
    apf::Adjacent adjacent;
    pumi::instance()->mesh->getAdjacent(e,target_dim,adjacent);      
    std::copy(adjacent.begin(), adjacent.end(), vecAdjEnt.begin());
  }
  else if (ent_dim>target_dim)
  {
    apf::Downward adjacent;
    int num_adj= pumi::instance()->mesh->getDownward(e,target_dim,adjacent); 
    for (int j = 0; j < num_adj; ++j) 
      vecAdjEnt.push_back(adjacent[j]);
  }
  if (!pumi_rank()) std::cout<<"[pumi error] "<<__func__<<": invalid target dimension "<<target_dim<<"\n";
}

void pumi_ment_get2ndadj (pMeshEnt e, int bridge_dim, int target_dim, std::vector<pMeshEnt>& vecAdjEnt)
{
  apf::Adjacent adjacent;
  apf::getBridgeAdjacent(pumi::instance()->mesh, e, bridge_dim, target_dim, adjacent);
  std::copy(adjacent.begin(), adjacent.end(), vecAdjEnt.begin());
}


int pumi_ment_getid(pMeshEnt e)
{
  return getMdsIndex(pumi::instance()->mesh, e);
}

pGeomEnt pumi_ment_getgeomclass(pMeshEnt e)
{
  return (pGeomEnt)(pumi::instance()->mesh->toModel(e));
}

pPartEnt pumi_ment_getptnclass(pMeshEnt e)
{  
  if (!pumi_rank()) std::cout<<"[pumi error] "<<__func__<<" not supported\n";
  return NULL;
}

// owner part information
int pumi_ment_getownrank(pMeshEnt e)
{
  return pumi::instance()->mesh->getOwner(e);
}

pMeshEnt pumi_ment_getownent(pMeshEnt e)
{

  if (!(pumi::instance()->mesh->isShared(e))) // internal ent
    return e;

  int own_partid = pumi_ment_getownrank(e);
  if (own_partid==pumi_rank()) return e;

  return pumi_ment_getrmt(e, own_partid);
}

bool pumi_ment_isowned(pMeshEnt e)
{  
  if (pumi_ment_getownrank(e)==pumi_rank())
    return true;
  else
    return false;
}

// remote copy information
bool pumi_ment_isonbdry (pMeshEnt e)
{
  if (pumi::instance()->mesh->isShared(e)) // internal ent
    return true;
  else 
    return false;
}

int pumi_ment_getnumrmt (pMeshEnt e)
{
  if (!(pumi::instance()->mesh->isShared(e))) // internal ent
    return 0;
  apf::Copies remotes;
  pumi::instance()->mesh->getRemotes(e,remotes);
  return remotes.size();
}

void pumi_ment_getallrmt(pMeshEnt e, pCopies& remotes)
{
  if (pumi::instance()->mesh->isShared(e))
    pumi::instance()->mesh->getRemotes(e,remotes);
}

pMeshEnt pumi_ment_getrmt(pMeshEnt& e, int pid)
{
  if (!(pumi::instance()->mesh->isShared(e))) // internal ent
    return NULL;

  pCopies remotes;
  pumi::instance()->mesh->getRemotes(e,remotes);
  return remotes[pid];
}

void pumi_ment_setrmt(pMeshEnt e, int partID, pMeshEnt r)
{
  if (!pumi_rank()) std::cout<<"[pumi error] "<<__func__<<" not supported\n";
}

void pumi_ment_deletermt (pMeshEnt e, int partID)
{
  if (!pumi_rank()) std::cout<<"[pumi error] "<<__func__<<" not supported\n";
}

void pumi_ment_cleanrmt (pMeshEnt e)
{
  if (!pumi_rank()) std::cout<<"[pumi error] "<<__func__<<" not supported\n";
}

void pumi_ment_setptntopology (pMeshEnt e)
{
  if (!pumi_rank()) std::cout<<"[pumi error] "<<__func__<<" not supported\n";
}

void pumi_ment_getresidence(pMeshEnt e, std::vector<int>& resPartId)
{
  resPartId.clear();
  pCopies remotes;
  pumi::instance()->mesh->getRemotes(e,remotes);
  APF_ITERATE(pCopies,remotes,rit)
    resPartId.push_back(rit->first);
  resPartId.push_back(pumi_rank());
}

void pumi_ment_getclosureresidence(pMeshEnt e, std::vector<int>& resPartId)
{
  resPartId.clear();
  apf::Downward vertices;
  int nvtx = pumi::instance()->mesh->getDownward(e,0,vertices);
  pMeshEnt vtx;
  
  for (int i=0; i<nvtx; ++i)
  {
    vtx=vertices[i];
    pCopies remotes;
    pumi::instance()->mesh->getRemotes(vtx,remotes);
    APF_ITERATE(pCopies,remotes,rit)
      if (std::find(resPartId.begin(), resPartId.end(), rit->first)!=resPartId.end())
        resPartId.push_back(rit->first);
  }
  resPartId.push_back(pumi_rank());
}

// ghosting information
bool pumi_ment_isghost(pMeshEnt e)
{}
bool pumi_ment_isghosted (pMeshEnt e)
{}
int pumi_ment_getnumghost (pMeshEnt e)
{}
void pumi_ment_getallghost (pMeshEnt e, pCopies& ghosts)
{}
pMeshEnt pumi_ment_getghost(pMeshEnt& meshEnt, int partID)
{}
