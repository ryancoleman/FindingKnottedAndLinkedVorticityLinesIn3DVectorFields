//Ryan Coleman
//MS thesis code
//computes Approximate Anisotropic Alpha-Shapes of vorticity vectors


#include <stdio.h>
#include <memory.h>
#include <stdlib.h>
#include <math.h>
#include <iostream.h>
#include <string.h>

#include "AAA.h"

//the following are grid sizes. now set by hand. could be read from file, commandline, etc.
int nx = 100;
int ny = nx, nz = nx;

double gridDist = 1.0; //check with bruce and lucas for real value

double maxNorm = 0.0; //for use in calcAlpha(...)
double maxAlpha = 1.1; //can be set in third input parameter
                       

float largestFloat = 3.4e38; //used as infinity is shortest cycles bits


//the following is the number of layers out from each point to generate distances for
int localLevel = 1;
int numDist; //cube this for # of distances needed per point per boxside

//the following is the big matrix where all the vectors are stored.
//the tails of the vectors are implicitly defined by their position xyz in the matrix
double *vorticities = NULL; // vorticities[nx][ny][nz][3]

//the following is the data structure to store the distances from each point to each other point in.
double *distances = NULL; // distances[nx][ny][nz][numDist^3] 

//the following is used to store the list of alpha shape segments
int *alphaSegs = NULL;  //integer[(nx*ny*nz*numDist^3)/2][3(xyz)][2(start,end)]
//this stores the actual distances. we'll sort by dists, and keep alphaSegs in the same order
double *alphaDists = NULL; //double[(nx*ny*nz*numDist^3)/2]

//this is for the graph & DFS searching bits
//idea is basically adjacency list
struct Node
  {
  int x,y,z;
  int color; //used for territorial marking
  Edge *edge;
  };

struct Edge
  {
  Node *node;
  Edge *edge;
  double dist;
  };

//used by graph, DFS, and union-find
inline int calcIndexN(int x, int y, int z)
  {
  return (x + ny * (y + nz * z));
  }

Node *graphNodes = NULL; //nodes[nx*ny*nz]

void InitGraph(Node *nodes)
  {
  for (int x = 0; x < nx; x++)
    {
    for (int y = 0; y < ny; y++)
      {
      for (int z = 0; z < nz; z++)
        {
        int index = calcIndexN(x,y,z);
        nodes[index].x = x;
        nodes[index].y = y;
        nodes[index].z = z;
        nodes[index].edge = NULL;
        nodes[index].color = 0;
        //fprintf(stderr, "%d\n", Value(sets[calcIndexN(x,y,z)]));
        }
      }
    }  
  }

void deleteEdge(Edge *thisEdge)
  {
  if (!(thisEdge == NULL))
    {
    deleteEdge(thisEdge->edge);
    free(thisEdge);
    }
  }

//do cleanup correctly in case code integrated later so no memory leaks.
void deleteGraph(Node *nodes)
  {
  for (int x = 0; x < nx; x++)
    {
    for (int y = 0; y < ny; y++)
      {
      for (int z = 0; z < nz; z++)
        {
        int index = calcIndexN(x,y,z);
        Edge *tempEdge = nodes[index].edge;
        deleteEdge(tempEdge);   
        }
      }
    }  
  free(nodes);
  }


//returns true if a directed path exists from start to target
//true is of course 1, 0 is false
int DepthFirstSearch(Node *start, Node *target, int colorVisit)
  {
  //fprintf(stderr, "%d %d %d\n", start->x, start->y, start->z);
  if (start == target)
    {
    return 1;
    }
  Edge *tempNext = start->edge;
  if (tempNext == NULL || start->color == colorVisit)
    {
    return 0;
    }
  else
    {
    start->color = colorVisit;
    int tempVal = 0;
    while (0 == tempVal && tempNext != NULL)
      {
      tempVal = DepthFirstSearch(tempNext->node, target, colorVisit);
      tempNext = tempNext->edge;
      }
    return tempVal;
    }
  }

//after a known path is found with DFS, use this to print cycle in reverse order
int PrintCycle(Node *start, Node *target, int colorVisit, FILE *outputFile)
  {
  if (start == target)
    {
    fprintf(outputFile, "%d %d %d\n", start->x, start->y, start->z);
    return 1;
    }
  Edge *tempNext = start->edge;
  if (tempNext == NULL || start->color == colorVisit)
    {
    return 0;
    }
  else
    {
    start->color = colorVisit;
    int tempVal = 0;
    while (0 == tempVal && tempNext != NULL)
      {
      tempVal = PrintCycle(tempNext->node, target, colorVisit, outputFile);
      tempNext = tempNext->edge;
      }
    if (1 == tempVal)
      {
      fprintf(outputFile, "%d %d %d\n", start->x, start->y, start->z);
      }
    return tempVal;
    }
  }


//union find structures -- adapted from CLRS
struct Parent
  {
  int id, rank;
  Parent *par;
  };


Parent *unionSets = NULL; //parents[nx*ny*nz]

Parent *FindSet(Parent *find)
  {
  if (find->par != NULL)
    {
    find->par = FindSet(find->par);
    return find->par;
    }
  return find;
  }

void Link(Parent *x, Parent *y)
  {
  if (x->rank > y->rank)
    {
    x->par = y;
    }
  else
    {
    y->par = x;
    if (x->rank == y->rank)
      {
      y->rank = y->rank + 1;
      }
    }
  }

void Union(Parent *x, Parent *y)
  {
  Link(FindSet(x), FindSet(y));
  }

int Value(Parent x)
  {
  Parent *y = FindSet(&x);
  return (y->id);
  }

void MakeSets(Parent *sets)
  {
  int unique = 100; //0 is reserved for superseded parent pointers.
  for (int x = 0; x < nx; x++)
    {
    for (int y = 0; y < ny; y++)
      {
      for (int z = 0; z < nz; z++)
        {
        sets[calcIndexN(x,y,z)].id = unique++;
        sets[calcIndexN(x,y,z)].rank = 0;
        sets[calcIndexN(x,y,z)].par = NULL;
        //fprintf(stderr, "%d\n", Value(sets[calcIndexN(x,y,z)]));
        }
      }
    }  
  }


//this code borrowed from Lucas Finn to read & write to the big matrix

// a[nx][ny][nz][3]
double GetArray(double *a, int i, int j, int k, int l)
	{
        int index = l + 3 * (k + nz * (j + ny * i));
        return a[index];
	}

// a[nx][ny][nz][3]
void SetArray(double *a, double v, int i, int j, int k, int l)
	{
        int index = l + 3 * (k + nz * (j + ny * i));
        a[index] = v;
	}

// a[nx][ny][nz][26 or 124]
double GetDist(double *a, int i, int j, int k, int l)
	{
        int index = l + numDist*numDist*numDist * (k + nz * (j + ny * i));
        return a[index];
	}

// a[nx][ny][nz][26 or 124]
void SetDist(double *a, double v, int i, int j, int k, int l)
	{
        int index = l + numDist*numDist*numDist * (k + nz * (j + ny * i));
        a[index] = v;
	}

int GetAlphaSeg(int *a, int order, int whichDim, int whichEnd)
  {
  int index = order * 6 + whichDim * 2 + whichEnd;
  return a[index];
  }

void SetAlphaSeg(int *a, int value, int order, int whichDim, int whichEnd)
  {
  int index = order * 6 + whichDim * 2 + whichEnd;
  a[index] = value;
  }

double GetAlphaDist(double *a, int order)
  {
  return a[order];
  }

void SetAlphaDist(double *a, double value, int order)
  {
  a[order] = value;
  }

void SwitchAlphas(double *a, int *b, int order, int other)
  {
  //switch distance
  double temp = a[order];
  a[order] = a[other];
  a[other] = temp;
  //switch 6 coordinates
  for (int whichDim = 0; whichDim < 3; whichDim++)
    {
    for (int whichEnd = 0; whichEnd < 2; whichEnd++)
      {
      int tempI = b[order * 6 + whichDim * 2 + whichEnd];
      b[order * 6 + whichDim * 2 + whichEnd] = b[other * 6 + whichDim * 2 + whichEnd];
      b[other * 6 + whichDim * 2 + whichEnd] = tempI;
      }
    }
  }


// a[nx][ny][nz][3]
int GetArray(int *a, int i, int j, int k, int l)
  {
  int index = l + 3 * (k + nz * (j + ny * i));
  return a[index];
  }


// a[nx][ny][nz][1]
float GetArray(float *a, int i, int j, int k)
  {
  int index = (k + nz * (j + ny * i));
  return a[index];
  }

// a[nx][ny][nz][3]
void SetArray(int *a, int v, int i, int j, int k, int l)
  {
  int index = l + 3 * (k + nz * (j + ny * i));
  a[index] = v;
  }

// a[nx][ny][nz][1]
void SetArray(float *a, float v, int i, int j, int k)
  {
  int index = (k + nz * (j + ny * i));
  a[index] = v;
  }




//following allows files from Lucas to be read in. expects indexX, Y, Z, vectorX, Y, Z format.
void readVorticitiesFile(double *vort, FILE *inputFile)
  {
  int i,j,k; //array indices
  double dx,dy,dz; //values of vector as put into vorticities matrix
  float x,y,z; //values of vector as read in
  while(fscanf(inputFile, "%d %d %d %e %e %e", &i, &j, &k, &x, &y, &z) != EOF)
    {
    dx =  x;
    dy =  y;
    dz =  z;
    double thisNorm = sqrt(dx*dx + dy*dy + dz*dz);
    if (thisNorm > maxNorm)
      {
      maxNorm = thisNorm;
      }
    SetArray(vort, dx, i, j, k, 0); 
    SetArray(vort, dy, i, j, k, 1); 
    SetArray(vort, dz, i, j, k, 2); 
    }
  //fprintf(stderr, "%e", maxNorm);
  }

//follows is a function to compute alpha based on the norm of the vector
//for now, linearly interpolate between maxNorm and 0 mapped to maxAlpha and 1
//could actually use a different interpolation...
double calcAlpha(double norm)
  {
  return (((norm/maxNorm) * (maxAlpha - 1.0)) + 1.0);
  }

//calculates the metric tensor, or the ellipse defining the way to measure distance
//returns a 9 element array, 0 1 2
//                           3 4 5
//                           6 7 8 
double *calcMetricTensor(double vX, double vY, double vZ, double vNormSqr, double alpha)
  {
  double *metTensor = (double *)malloc(9*sizeof(double));
  double *vvTDvNS = (double *)malloc(9*sizeof(double));
  if (0.0 == vNormSqr)
    {
    //this means all the vectors have 0 length
    //so no matter what, the metricTensor should be 1-diagonal
    for (int count = 0; count < 9; count++)
      {
      if (0 == count || 4 == count || 8 == count) //diagonal
        {
        metTensor[count] = 1.0;
        }
      else
        {
        metTensor[count] = 0.0;
        }
      //fprintf(stderr,  "%d %f \n", count, metTensor[count]);
      }
    }
  else //the vectors are positive, so we want to compute a real metric tensor
    {
    for (int count = 0; count < 9; count++)
      {
      if (count % 3 == 0)
        {
        vvTDvNS[count] = vX;
        }
      else if (count % 3 == 1)
        {
        vvTDvNS[count] = vY;
        }
      else if (count % 3 == 2)
        {
        vvTDvNS[count] = vZ;
        }
      if (count < 3)
        {
        vvTDvNS[count] *= vX;
        }
      else if (count < 6)
        {
        vvTDvNS[count] *= vY;
        }
      else if (count < 9)
        {
        vvTDvNS[count] *= vZ;
        }
      vvTDvNS[count] /= vNormSqr;
      //fprintf(stderr,  "%d %f \n", count, vvTDvNS[count]);
      } 
    //now with vvTDvNS (or v*transpose(v)/|v|^2), make the metric tensor according to formula
    for (int count = 0; count < 9; count++)
      {
      if (0 == count || 4 == count || 8 == count) //diagonal
        {
        metTensor[count] = 1.0;
        }
      else
        {
        metTensor[count] = 0.0;
        }
      //old but seems wrong... try new
      //metTensor[count] +=  - vvTDvNS[count] + (alpha*alpha) * vvTDvNS[count];
      //metTensor[count] /=  alpha;
      metTensor[count] +=  - vvTDvNS[count] + (1.0/(alpha*alpha)) * vvTDvNS[count];
      metTensor[count] *=  alpha;
      //fprintf(stderr,  "%d %e \n", count, metTensor[count]);
      }
    }
  //checking, does this all need reversed?
  free(vvTDvNS);
  return metTensor;
  }


//following is at least half the algorithm
//transforms each point locally
//and computes numDist distances out to localLevel away
void calcAppAniDel(double *vort, double *dist)
  {
  //big for loop. means do everything for every vector in vort.
  for (int x = 0; x < nx; x++)
    {
    for (int y = 0; y < ny; y++)
      {
      for (int z = 0; z < nz; z++)
        {
        //fprintf(stderr, "%d %d %d\n", x, y, z);
        //find transform based on this vector.
	double vX = GetArray(vort, x, y, z, 0);
	double vY = GetArray(vort, x, y, z, 1);
	double vZ = GetArray(vort, x, y, z, 2);
        double vNormSqr = vX*vX + vY*vY + vZ*vZ;  //used alot
	double alpha = calcAlpha(sqrt(vNormSqr)); //used equally alot
        //fprintf(stderr, "%e %f\n", vNormSqr, alpha);
        //fprintf(stderr, "%f %f %f\n", vX, vY, vZ);
	double *metricTensor = calcMetricTensor(vX, vY, vZ, vNormSqr, alpha);
	
        //calc distance and save in each point
	for (int boxX = 0; boxX < numDist; boxX++)
          {
	  for (int boxY = 0; boxY < numDist; boxY++)
            {  
            for (int boxZ = 0; boxZ < numDist; boxZ++)
              {
              //gridDist is global variable that determines grid spacing
              //void SetDist(double *a, double v, int i, int j, int k, int l) is function to set each distance
              double thisDist = 0.0;
              if (!((0 == boxX - localLevel) && (0 == boxY - localLevel) && (0 == boxZ - localLevel))) 
                {
                // we don't set the distance to this point, just to all surrounding.
                // treat current vector tail as origin, compute distances with metricTensor
                // and put total in thisDist
                
                //first compute the array representing the difference between this point and the one 
                //the distance to which is being calculated
                double *point = (double *)malloc(3 * sizeof(double));
                point[0] = (boxX - localLevel) * gridDist;
                point[1] = (boxY - localLevel) * gridDist;
                point[2] = (boxZ - localLevel) * gridDist;
                //fprintf(stderr, "%f %f %f \n", point[0], point[1], point[2]);
                //first step is metric (3x3) times the point (3x1) resulting in a 3x1 matrix
                double *firstStep = (double *)malloc(3 * sizeof(double));
                firstStep[0] = point[0] * metricTensor[0] + 
                               point[1] * metricTensor[1] + 
                               point[2] * metricTensor[2];
                firstStep[1] = point[0] * metricTensor[3] + 
                               point[1] * metricTensor[4] + 
                               point[2] * metricTensor[5];
                firstStep[2] = point[0] * metricTensor[6] + 
                               point[1] * metricTensor[7] + 
                               point[2] * metricTensor[8];
                //fprintf(stderr, "%f %f %f \n", firstStep[0], firstStep[1], firstStep[2]);
                //now take the point as a 1x3 matrix times this 3x1 resulting in a 1x1 (scalar) value.
                thisDist =       firstStep[0] * point[0] +
                                 firstStep[1] * point[1] +                 
                                 firstStep[2] * point[2] ;
                if (thisDist < 0.0)
                  {
                  thisDist = sqrt(-thisDist);
                  }
                else
                  {
                  thisDist = sqrt(thisDist);
                  }
                //the following lines transforms the dist in L2-norm to L(infinity)-norm
                //this cheap trick is only going to work for localLevel == 1
                //since in that case all 26 points have the same L(infinite)-distance
                double normalDist = sqrt( point[0] * point[0] +
                                          point[1] * point[1] +
                                          point[2] * point[2] );
                double infDist = abs(boxX - localLevel);
		if (infDist < abs(boxY - localLevel))
                  {
                  infDist = abs(boxY - localLevel);
                  }
		if (infDist < abs(boxZ - localLevel))
                  {
                  infDist = abs(boxZ - localLevel);
                  }
                //fprintf(stderr, "%f\n", infDist);
                thisDist = infDist*thisDist/normalDist;
                //end L(infinity)-norm conversion
                free(firstStep);
                free(point);
                //fprintf(stderr, "%e \n", thisDist);
                }
              SetDist(dist, thisDist, x, y, z, (boxZ + numDist * (boxY + numDist * boxX)) );
              }
            }
          }
	//that should be it for this step.
        free(metricTensor);
        }
      }
    }
  }

//perform an orientation (half-plane) test on one vector and one point away from that vector
//takes F=(fx,fy,fz) as the point with the vector V=(vX,vY,vZ) associated
//and T=(tx,ty,tz) as the other point to check. 
//assumes nx,ny,nz >> localLevel which is okay
//returns a double which is positive iff the point T is on the side of the oriented half-space defined by F,V
//the double is negative iff the point T is not on the side.
//a 0.0 answer is undecided--can indicate a 0-length vector
double orientation(int fx, int fy, int fz, int tx, int ty, int tz, double vX, double vY, double vZ)
  {
  //signs of each component of v and t-f etc should match ... complicated by wrap-around segments
  //first compute wX,wY,wZ = t - f
  //assumes nx,ny,nz >> localLevel
  double wX = tx - fx;
  double wY = ty - fy;
  double wZ = tz - fz;
  if (abs((int)wX) > localLevel)
    {
    if (wX > localLevel)
      wX -= nx;
    else if (wX < localLevel)
      wX += nx;
    }
  if (abs((int)wY) > localLevel)
    {
    if (wY > localLevel)
      wY -= ny;
    else if (wY < localLevel)
      wY += ny;
    }
  if (abs((int)wZ) > localLevel)
    {
    if (wZ > localLevel)
      wZ -= nz;
    else if (wZ < localLevel)
      wZ += nz;
    }
  //fprintf(stderr, "%f:(%d %d %d) -  (%d %d %d) = (%f %f %f)\n", minDist, tx, ty, tz, fx, fy, fz, wX, wY, wZ);
  //now determine orientation, whether f -> t (as assumed) or t -> f
  //all needed to check is sign of dot product of w and v    
  double signOrientation = wX*vX + wY*vY + wZ*vZ;
  return signOrientation;
  }

//put into arrays, check orientation... then sort.
int  averageStoreDistances(double *dists, int *aSegs, double *aDists, double *vort)
  {
  //okay, go through the whole big array of distances, setting each on the first pass, then adding, then /2
  //also store the distances and the point indices they connect
  int countStore = 0;
  for (int x = 0; x < nx; x++)
    {
    for (int y = 0; y < ny; y++)
      {
      for (int z = 0; z < nz; z++)
        {
        //fprintf(stderr, "another point %d %d %d\n", x, y, z);
	for (int boxX = 0; boxX < numDist; boxX++)
          {
	  for (int boxY = 0; boxY < numDist; boxY++)
            {  
            for (int boxZ = 0; boxZ < numDist; boxZ++)
              {
              if (!((0==boxX - localLevel)&&(0==boxY - localLevel)&&(0==boxZ - localLevel)))
                {
                double thisDist =  GetDist(dists, x, y, z, (boxZ + numDist * (boxY + numDist * boxX)) );
                //now find other distance this is to.
                int otherX = (nx + x + boxX - localLevel) % nx;
                int otherY = (ny + y + boxY - localLevel) % ny;
                int otherZ = (nz + z + boxZ - localLevel) % nz;
                int otherBoxX = numDist - boxX - 1;
                int otherBoxY = numDist - boxY - 1;
                int otherBoxZ = numDist - boxZ - 1;
                double otherDist =  GetDist(dists, otherX, otherY, otherZ, 
                                            (otherBoxZ + numDist * (otherBoxY + numDist * otherBoxX)) );
                double newDist = (thisDist + otherDist)/2.0;
                SetDist(dists, newDist, x, y, z, (boxZ + numDist * (boxY + numDist * boxX)));
                SetDist(dists, newDist, otherX, otherY, otherZ,
                                            (otherBoxZ + numDist * (otherBoxY + numDist * otherBoxX)) );
                
                if ((z + nz * (y + ny * x)) < (otherZ + nz * (otherY + ny * otherX)) )
                  {
                  //fprintf(stderr, "(%d %d %d) (%d %d %d) %e\n", x, y, z, otherX, otherY, otherZ, newDist); 
                  //determine orientation now instead of later
                  //throw out poorly defined orientations as well
 	          double vX = GetArray(vort, x, y, z, 0);
 	          double vY = GetArray(vort, x, y, z, 1);
	          double vZ = GetArray(vort, x, y, z, 2);
                  double signOrient = orientation(x,y,z,otherX,otherY,otherZ,vX,vY,vZ);
 	          vX = GetArray(vort, otherX, otherY, otherZ, 0);
 	          vY = GetArray(vort, otherX, otherY, otherZ, 1);
 	          vZ = GetArray(vort, otherX, otherY, otherZ, 2);
                  double signBackOrient = orientation(otherX,otherY,otherZ,x,y,z,vX,vY,vZ);
                  //fprintf(stderr, "%f %f \n", signOrient,signBackOrient);
                  if (signOrient > 0.0 && signBackOrient < 0.0)
                    {
                    SetAlphaDist(aDists, newDist, countStore);
                    SetAlphaSeg(aSegs, x, countStore, 0, 0);
                    SetAlphaSeg(aSegs, y, countStore, 1, 0);
                    SetAlphaSeg(aSegs, z, countStore, 2, 0);
                    SetAlphaSeg(aSegs, otherX, countStore, 0, 1);
                    SetAlphaSeg(aSegs, otherY, countStore, 1, 1);
                    SetAlphaSeg(aSegs, otherZ, countStore, 2, 1);
                    countStore++;
                    }
                  else if (signOrient < 0.0 && signBackOrient > 0.0)
                    {
                    SetAlphaDist(aDists, newDist, countStore);
                    SetAlphaSeg(aSegs, otherX, countStore, 0, 0);
                    SetAlphaSeg(aSegs, otherY, countStore, 1, 0);
                    SetAlphaSeg(aSegs, otherZ, countStore, 2, 0);
                    SetAlphaSeg(aSegs, x, countStore, 0, 1);
                    SetAlphaSeg(aSegs, y, countStore, 1, 1);
                    SetAlphaSeg(aSegs, z, countStore, 2, 1);
                    countStore++;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  return countStore;
  }

//min-heap code here
//left, returns left child
int left(int i)
  {
  return (static_cast<int>(2*i));
  }

//right, returns right child
int right(int i)
  {
  return (1 + static_cast<int>(2*i));
  }

//makes sure value at i is correct, assumes rest of heap correct
void minHeapify(double *A, int *B, int i, int size)
  {
  int l = left(i);
  int r = right(i);
  int smallest;
  if ((l < size) && (A[l] < A[i]))
    {
    smallest = l;
    }
  else
    {
    smallest = i;
    }
  if ((r < size) && (A[r] < A[smallest]))
    {
    smallest = r;
    }
  if (smallest != i)
    {
    SwitchAlphas(A, B, i, smallest);
    minHeapify(A, B,  smallest, size);
    }
  }

//performs a bottom up min-heapify to build a heap
void buildMinHeap(double *A, int *B, int size)
  {
  for (int i = size; i >= 0; i--)
    {
    minHeapify(A, B, i, size);
    }
  }



void buildCycles(double *dist, int *segs, int sizeHeap, 
                 Parent *sets, Node *nodes, FILE *outputFile)
  {
  //note: heap should already be built.
  int curHeapLength = sizeHeap;
  int unions= 0;  //counters, useful for debugging
  int nons= 0;
  int continueFinding = 1; //change to 0 inside loop to stop
  for (int i = sizeHeap - 1; i > -1 && 1 == continueFinding ; i--)
    {
    double minDist = GetAlphaDist(dist, 0);
    int fx = GetAlphaSeg(segs, 0, 0, 0);
    int fy = GetAlphaSeg(segs, 0, 1, 0);
    int fz = GetAlphaSeg(segs, 0, 2, 0);
    int tx = GetAlphaSeg(segs, 0, 0, 1);
    int ty = GetAlphaSeg(segs, 0, 1, 1);
    int tz = GetAlphaSeg(segs, 0, 2, 1);
    SwitchAlphas(dist, segs, 0, i);    
    curHeapLength--;
    minHeapify(dist, segs, 0, curHeapLength);
    //fprintf(stderr, "%f\n", minDist);
    //fprintf(stderr, "%f (%d %d %d) (%d %d %d)\n", minDist, fx, fy, fz, tx, ty, tz);
    //now we process the two points.
    //fprintf(stderr, "%d %d\n", calcIndexN(fx, fy, fz), calcIndexN(tx, ty, tz));
    //we need these several places
    int fromIndex = calcIndexN(fx,fy,fz);
    int toIndex = calcIndexN(tx,ty,tz);
    if (Value(sets[fromIndex]) != Value(sets[toIndex]))
      {
      Union(&sets[fromIndex], &sets[toIndex]);
      //and add edge to new graph from fx,y,z to tx,y,z
      Edge *newEdge = (Edge *)malloc(sizeof(Edge));
      newEdge->node = &nodes[toIndex];
      newEdge->edge = NULL;
      newEdge->dist = minDist;
      if (nodes[fromIndex].edge == NULL)
        {
        nodes[fromIndex].edge = newEdge;
        }
      else
        {
        Edge *temp = nodes[fromIndex].edge;
        while (temp->edge != NULL)
          {
          temp = temp->edge;
          }
        temp->edge = newEdge;
        }
      unions++;
      }
    else
      {
      //fprintf(stderr, "%d %d\n"  ,Value(sets[calcIndexN(fx, fy, fz)]) ,Value(sets[calcIndexN(tx, ty, tz)]));
      //search, find and output! (or discard)
      int success = DepthFirstSearch(&nodes[toIndex], &nodes[fromIndex], nons);
      if (1 == success)
        {
        //fprintf(stderr, "cycle from %d %d %d to %d %d %d\n", tx, ty, tz, fx, fy, fz);
        nons++;
        fprintf(outputFile, "Component 1 of 1:\n");
        PrintCycle(&nodes[toIndex], &nodes[fromIndex], nons, outputFile);
        fprintf(outputFile, "\n");
        continueFinding = 0; //stop processing
        }
      //do the following steps in this case no matter what, i.e. add the edge to the graph
      Edge *newEdge = (Edge *)malloc(sizeof(Edge));
      newEdge->node = &nodes[toIndex];
      newEdge->edge = NULL;
      newEdge->dist = minDist;
      if (nodes[fromIndex].edge == NULL)
        {
        nodes[fromIndex].edge = newEdge;
        }
      else
        {
        Edge *temp = nodes[fromIndex].edge;
        while (temp->edge != NULL)
          {
          temp = temp->edge;
          }
        temp->edge = newEdge;
        }
      nons++;
      }
    }  
  //fprintf(stderr, "%d %d\n", unions, nons);
  }



int main(int argc, char* argv[])
  {

  //memory allocation could be better. could allocate as needed later in processing? could make loop to keep from
  //hand coding all the following steps.


  //initialize big matrix to store vectors
        if (!vorticities) {
                vorticities = (double *)malloc(3 * nx * ny * nz * sizeof(double));
        }
        if (!vorticities) {
              fprintf(stderr,  "Out of memory: malloc(vorticities) failed.\n");
                exit(1);
        }
        if ((unsigned int)vorticities % 8) {
               fprintf(stderr, "Vorticities array is not qword aligned.\n");
                exit(1);
        }
  numDist = (localLevel*2+1); //cube this for number to add.
  if (!distances) 
    {
    distances = (double *)malloc(numDist*numDist*numDist * nx * ny * nz * sizeof(double));
    }
        if (!distances) {
              fprintf(stderr, "Out of memory: malloc(distances) failed.\n");
                exit(1);
        }
        if ((unsigned int)distances % 8) {
              fprintf(stderr, "Distances array is not qword aligned.\n");
                exit(1);
        }

  if (!alphaSegs)
    {
    alphaSegs = (int *)malloc(3 * (numDist*numDist*numDist-1) * nx * ny * nz * sizeof(int));
    }
        if (!alphaSegs) {
              fprintf(stderr, "Out of memory: malloc(alphaSegs) failed.\n");
                exit(1);
        }
        if ((unsigned int)alphaSegs % 8) {
              fprintf(stderr, "alphaSegs array is not qword aligned.\n");
                exit(1);
        }

  if (!alphaDists)
    {
    alphaDists = (double *)malloc( ((numDist*numDist*numDist-1) * nx * ny * nz * sizeof(double))/2);
    }
        if (!alphaDists) {
              fprintf(stderr, "Out of memory: malloc(alphaDists) failed.\n");
                exit(1);
        }
        if ((unsigned int)alphaDists % 8) {
              fprintf(stderr, "alphaDists array is not qword aligned.\n");
                exit(1);
        }
  if (!unionSets)
    {
    unionSets = (Parent *)malloc( nx * ny * nz * sizeof(Parent));
    }
  if (!unionSets) 
    {
    fprintf(stderr, "Out of memory: malloc(unionSets) failed.\n");
    exit(1);
    }
  if ((unsigned int)unionSets % 8)
    {
    fprintf(stderr, "unionSets array is not qword aligned.\n");
    exit(1);
    }
  if (!graphNodes)
    {
    graphNodes = (Node *)malloc( nx * ny * nz * sizeof(Node));
    }
  if (!graphNodes) 
    {
    fprintf(stderr, "Out of memory: malloc(graphNodes) failed.\n");
    exit(1);
    }
  if ((unsigned int)graphNodes % 8)
    {
    fprintf(stderr, "graphNodes array is not qword aligned.\n");
    exit(1);
    }



  if (argc >= 3)
    {
    maxAlpha = atof(argv[3]);
    }
  //hardcoded file input
  FILE *inputFile = fopen(argv[1], "r");
  FILE *outputFile = fopen(argv[2], "w");
  readVorticitiesFile(vorticities, inputFile);
  fclose(inputFile);

  //calculate distances from nearby points to other nearby points
  calcAppAniDel(vorticities, distances);

  //average the distances calculated
  int sizeHeap = averageStoreDistances(distances, alphaSegs, alphaDists, vorticities);

  //free this memory now.
  free(vorticities);
  free(distances);

  //put the distances into a min-heap in-place
  buildMinHeap(alphaDists, alphaSegs, sizeHeap);

  MakeSets(unionSets); //initialize
  InitGraph(graphNodes);  

  //now we call the big function that builds cycles--need vorticities to determine orientation
  buildCycles(alphaDists, alphaSegs, sizeHeap, unionSets, graphNodes, outputFile);
  fclose(outputFile);

  free(alphaDists);
  free(alphaSegs);

  deleteGraph(graphNodes);

  free(unionSets);
 

  }
