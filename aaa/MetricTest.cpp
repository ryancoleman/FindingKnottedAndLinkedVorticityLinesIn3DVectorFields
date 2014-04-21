//Ryan Coleman
//MS thesis code
//computes Approximate Anisotropic Alpha-Shapes of vorticity vectors

//this code is HORRIBLE from design point of view
//lots of things are hardcoded in, but oh well
//heck there are global variables

#include <stdio.h>
#include <memory.h>
#include <stdlib.h>
#include <math.h>
#include <iostream.h>
#include <string.h>

//the following are grid sizes. now set by hand. could be read from file, commandline, etc.
int nx = 100;
int ny = nx, nz = nx;

double gridDist = 1.0; //check with bruce and lucas for real value
double scaleVec = 10000.0;  //scale each input vector by this value

double maxNorm = 0.0; //for use in calcAlpha(...)
double maxAlpha = 1.1; //otherwise we get imaginary transformations.????

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

//stores the 'linked-list' of cycles from point to point. indexed by xyz
int *outEdges = NULL; // outEdges[nx][ny][nz][3]
int *inEdges = NULL; // inEdges[nx][ny][nz][3]


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

//since this initialization is actually important
void initEdges(int *outE, int *inE)
  {
  for (int count = 0; count < nx * ny * nz * 3; count++)
    {
    outE[count] = -1;
    inE[count] = -1;
    }
  }

// a[nx][ny][nz][3]
int GetArray(int *a, int i, int j, int k, int l)
  {
  int index = l + 3 * (k + nz * (j + ny * i));
  return a[index];
  }

// a[nx][ny][nz][3]
void SetArray(int *a, int v, int i, int j, int k, int l)
  {
  int index = l + 3 * (k + nz * (j + ny * i));
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
    dx = scaleVec * x;
    dy = scaleVec * y;
    dz = scaleVec * z;
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
      metTensor[count] +=  - vvTDvNS[count] + (1.0/(alpha*alpha)) * vvTDvNS[count];
      metTensor[count] *=  alpha;
      fprintf(stderr,  "%d %e \n", count, metTensor[count]);
      }
    }
  //checking, does this all need reversed?
  for (int count = 0; count < 4; count++)
    {
    int other = 8 - count;
    double temp = metTensor[other];
    metTensor[other] = metTensor[count];
    metTensor[count] = temp;
    }
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
        fprintf(stderr, "%d %d %d\n", x, y, z);
        //find transform based on this vector.
	double vX = GetArray(vorticities, x, y, z, 0);
	double vY = GetArray(vorticities, x, y, z, 1);
	double vZ = GetArray(vorticities, x, y, z, 2);
        double vNormSqr = vX*vX + vY*vY + vZ*vZ;  //used alot
	double alpha = calcAlpha(sqrt(vNormSqr)); //used equally alot
        fprintf(stderr, "%e %f\n", vNormSqr, alpha);
        fprintf(stderr, "%f %f %f\n", vX, vY, vZ);
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
                //double normalDist = sqrt( point[0] * point[0] +
                //                          point[1] * point[1] +
                //                          point[2] * point[2] );
                //thisDist = thisDist/normalDist;
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

//put into arrays, then sort.
void  averageStoreDistances(double *dists, int *aSegs, double *aDists)
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
                  SetAlphaDist(aDists, newDist, countStore);
                  SetAlphaSeg(aSegs, x, countStore, 0, 0);
                  SetAlphaSeg(aSegs, y, countStore, 1, 0);
                  SetAlphaSeg(aSegs, z, countStore, 2, 0);
                  SetAlphaSeg(aSegs, otherX, countStore, 0, 1);
                  SetAlphaSeg(aSegs, otherY, countStore, 1, 1);
                  SetAlphaSeg(aSegs, otherZ, countStore, 2, 1);
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

void buildCycles(double *dist, int *segs, int *inE, int *outE, double *vort, int sizeHeap)
  {
  //note: heap should already be built.
  int curHeapLength = sizeHeap;
  for (int i = sizeHeap - 1; i > -1; i--)
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
    //determine orientation by examining their vorticity vectors.
    double vX = GetArray(vort, fx, fy, fz, 0);
    double vY = GetArray(vort, fx, fy, fz, 1);
    double vZ = GetArray(vort, fx, fy, fz, 2);
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
    //fprintf(stderr, "%f:w(%f %f %f) v(%f %f %f) = %f\n", minDist, wX, wY, wZ, vX, vY, vZ, signOrientation);
    if (signOrientation < 0.0)
      {
      //we were wrong, so switch the to and froms
      int temp = fx;
      fx = tx;
      tx = temp;
      temp = fy;
      fy = ty;
      ty = temp;
      temp = fz;
      fz = tz;
      tz = temp;
      }
    //now check to see if we can add this edge
    //if f has no out edge yet
    // and t has no in edge yet
    if (GetArray(outE, fx, fy, fz, 0) == -1 &&
        GetArray(inE, tx, ty, tz, 0) == -1) 
      {
      //fprintf(stderr, "Adding edge from (%d %d %d) to (%d %d %d)\n", fx, fy, fz, tx, ty, tz);
      //fprintf(stdout, "%d %d %d  %d %d %d\n", fx, fy, fz, tx, ty, tz);
      //fprintf(stdout, "%d %d %d  %f %f %f\n", fx, fy, fz, fx+vX, fy+vY, fz+vZ);
      SetArray(outE, tx, fx, fy, fz, 0);
      SetArray(outE, ty, fx, fy, fz, 1);
      SetArray(outE, tz, fx, fy, fz, 2);
      SetArray(inE, fx, tx, ty, tz, 0);
      SetArray(inE, fy, tx, ty, tz, 1);
      SetArray(inE, fz, tx, ty, tz, 2);
      }
    //else
    //  {
    //  fprintf(stderr, "Skipping edge from (%d %d %d) to (%d %d %d)\n", fx, fy, fz, tx, ty, tz);
    //  }
    }  
  }

//writes the cycles out as a list of segments.
void writeCycles(int *outE)
  {
  for (int x = 0; x < nx; x++)
    {
    for (int y = 0; y < ny; y++)
      {
      for (int z = 0; z < nz; z++)
        {
        int i = GetArray(outE, x, y, z, 0);
        int j = GetArray(outE, x, y, z, 1);
        int k = GetArray(outE, x, y, z, 2);
        if (-1 != i && -1 != j && -1 != k)
          {
//          fprintf(stdout, "%d %d %d %d %d %d\n", x, y, z, i, j, k);
          }
        }
      }
    }  
  }



int main(int argc, char* argv[])
  {

  maxNorm = sqrt(1.0);

  double vX = 0.0;
  double vY = 0.0;
  double vZ = 1.0;
  double vNormSqr = vX*vX + vY*vY + vZ*vZ;  
  fprintf(stderr, "%f %f %f\n", vX, vY, vZ);

  double alpha = calcAlpha(sqrt(vNormSqr));
  fprintf(stderr, "%e %f\n", vNormSqr, alpha);
  double *metricTensor = calcMetricTensor(vX, vY, vZ, vNormSqr, alpha);

  numDist = (localLevel*2+1);

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
                thisDist = thisDist/normalDist;
                //end L(infinity)-norm conversion
                free(firstStep);
                free(point);
                fprintf(stderr, "%f ", thisDist);
                }
              //SetDist(dist, thisDist, x, y, z, (boxZ + numDist * (boxY + numDist * boxX)) );
              }
            fprintf(stderr, "\n");
            }
          fprintf(stderr, "\n");
          }
	//that should be it for this step.
        free(metricTensor);
	

  }
