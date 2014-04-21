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


//A data class
public class Link
  {
  public final Point3d beg;
  public final Point3d end;
  public final double dist;
  public final boolean rejected;
  public final boolean wrap;

  public Link(Point3d begIn, Point3d endIn, double distIn, boolean rejIn)
    {
    beg = begIn;
    end = endIn;
    dist = distIn;
    rejected = rejIn;
    wrap = false;
    }

  public Link(Point3d begIn, Point3d endIn, double distIn, boolean rejIn, boolean wrapIn)
    {
    beg = begIn;
    end = endIn;
    dist = distIn;
    rejected = rejIn;
    wrap = wrapIn;
    }


  }
