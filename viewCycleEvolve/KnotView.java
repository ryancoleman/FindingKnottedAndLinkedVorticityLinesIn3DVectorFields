//Ryan Coleman MS Thesis java3d viewer
//because everyone else's sucks.

import java.applet.Applet;
import java.awt.BorderLayout;
import java.awt.event.*;
import java.awt.GraphicsConfiguration;
import com.sun.j3d.utils.applet.*;
import com.sun.j3d.utils.geometry.*;
import com.sun.j3d.utils.universe.*;
import com.sun.j3d.utils.behaviors.vp.*;
import com.sun.j3d.utils.behaviors.keyboard.*;
import javax.media.j3d.*;
import javax.vecmath.*;
import java.io.*;
import java.util.*;

public class KnotView
  {
  //create a nice little reader/parser thingy for each type of input file you'd like
  //good design? i don't care. i need viz!
  
  //the point3d[] returned is formatted as a tail, head pair of points for each 
  //vorticity vector, dump into something to display segments should be good enough
  //returns a vector of Link[]
  static Vector readVorticitiesFile(String fileName)
    { 
    File pointsFile = new File(fileName);
    Vector returnV = new Vector(5);
    Vector pointArray = null;
    try
      {
      BufferedReader pointsFileReader = new BufferedReader(new FileReader(pointsFile));
      String lineIn = null;
      double xCoord = -100.0, yCoord = -100.0, zCoord = -100.0;
      pointArray = new Vector(100);
      while ((lineIn = pointsFileReader.readLine()) != null)
        {
        StringTokenizer tokens = new StringTokenizer(lineIn);
        if (lineIn.startsWith("Component"))
          {
          if (pointArray == null)
            {
            pointArray = new Vector(100); //holds Link's
            }
          else
            {
            if (pointArray.size() > 0)
              {
              pointArray.add(new Link(((Link)pointArray.elementAt(pointArray.size()-1)).end, (
                               (Link)pointArray.elementAt(0)).beg,
                               1.0, false));
              Link[] retP = new Link[pointArray.size()];
              pointArray.toArray(retP);
              returnV.add(retP);
              }
            pointArray = new Vector(100); //holds Link's
            }
          }
        if (3 == tokens.countTokens())
          {
          double iCoord = -100.0;
          double jCoord = -100.0;
          double kCoord = -100.0;
          if (xCoord > -100.0 && yCoord > -100.0 && zCoord > -100.0)
            {
            iCoord = xCoord;
            jCoord = yCoord;
            kCoord = zCoord;
            }
          xCoord = Double.parseDouble(tokens.nextToken());
          yCoord = Double.parseDouble(tokens.nextToken());
          zCoord = Double.parseDouble(tokens.nextToken());
          double distance = 1.0;
          if (tokens.hasMoreTokens())
            {
            distance = Double.parseDouble(tokens.nextToken());
            }

          //cheap hack for now
          if (iCoord > -100.0 && (((Math.abs(xCoord - iCoord) < 5) &&
             (Math.abs(yCoord - jCoord) < 5) && 
             (Math.abs(zCoord - kCoord) < 5))  ))
            {
            if (tokens.hasMoreTokens() && (Integer.parseInt(tokens.nextToken()) == 1))
              {
              pointArray.add(new Link(new Point3d(xCoord, yCoord, zCoord), new Point3d(iCoord, jCoord, kCoord), 
                             distance, true));
              }  
            else
              {
              pointArray.add(new Link(new Point3d(xCoord, yCoord, zCoord), new Point3d(iCoord, jCoord, kCoord), 
                             distance, false));
              }  
            }
          else //make wrapped links
            {
            /*
            if (tokens.hasMoreTokens() && (Integer.parseInt(tokens.nextToken()) == 1))
              {
              pointArray.add(new Link(new Point3d(xCoord, yCoord, zCoord), 
                             new Point3d(iCoord, jCoord, kCoord), 
                             distance, true, true));
              pointArray.add(new Link(new Point3d(xCoord, yCoord, zCoord), 
                             new Point3d(iCoord, jCoord, kCoord), 
                             distance, true, true));
              }  
            else
              {
              pointArray.add(new Link(new Point3d(xCoord, yCoord, zCoord), 
                             new Point3d(iCoord, jCoord, kCoord), 
                             distance, false, true));
              pointArray.add(new Link(new Point3d(xCoord, yCoord, zCoord), 
                             new Point3d(iCoord, jCoord, kCoord), 
                             distance, false, true));
              }  
            */
            }
          }
        }
      } 
    catch (IOException ioe)
      {
      System.err.println("error reading the file: " + fileName);
      ioe.printStackTrace();
      }

    if (pointArray.size() > 0)
      {
      pointArray.add(new Link(((Link)pointArray.elementAt(pointArray.size()-1)).end, ((Link)pointArray.elementAt(0)).beg,
                               1.0, false));
      Link[] retP = new Link[pointArray.size()];
      pointArray.toArray(retP);
      returnV.add(retP);
      }
 
    return returnV;   
    }


  //main is used to run the algorithm by itself and display the results 
  public static void main(String[] args)
    {
    //args[count] should be a files with points in it!
    Vector evolveLinkArrays = new Vector(args.length);
    for (int count = 0; count < args.length; count++)
      {
      Vector links = readVorticitiesFile(args[count]);
      if (links.size() > 0)
        {
        evolveLinkArrays.add(links);
        }
      }
    Display display = new Display(evolveLinkArrays);
    JMainFrame mainFrame = new JMainFrame(display, 700, 700);
    mainFrame.setTitle("EvolveKnotView");


    }


  }
