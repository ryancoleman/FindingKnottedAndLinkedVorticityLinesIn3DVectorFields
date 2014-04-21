
import javax.media.j3d.*;
import javax.vecmath.*;
import javax.swing.*;
import java.awt.event.*;
import java.awt.GridLayout;
import java.io.*;
import java.util.*;
 
public class BoxSides extends Shape3D
  {

  public BoxSides(Link[] pointArrays) 
    {
    super();
    QuadArray cube = new QuadArray(24, QuadArray.COORDINATES | QuadArray.COLOR_3);
    //figure out min, max in each direction
    double xMin = 0.0;
    double xMax = 100.0;
    double yMin = 0.0;
    double yMax = 100.0;
    double zMin = 0.0;
    double zMax = 100.0;
    for (int count = 0; count < pointArrays.length; count++)
      {
      if (!pointArrays[count].wrap)
        {
        if (pointArrays[count].beg.x < xMin)
          xMin = pointArrays[count].beg.x;
        if (pointArrays[count].beg.x > xMax)
          xMax = pointArrays[count].beg.x;
        if (pointArrays[count].beg.y < yMin)
          yMin = pointArrays[count].beg.y;
        if (pointArrays[count].beg.y > yMax)
          yMax = pointArrays[count].beg.y;
        if (pointArrays[count].beg.z < zMin)
          zMin = pointArrays[count].beg.z;
        if (pointArrays[count].beg.z > zMax)
          zMax = pointArrays[count].beg.z;
        }
      }
    //adjust to move sides 1 unit away
    xMin -= 1.0;
    yMin -= 1.0;
    zMin -= 1.0;
    xMax += 1.0;
    yMax += 1.0;
    zMax += 1.0;
    //a cube has 6 faces.
    cube.setCoordinate(0, new Point3d(xMin, yMin, zMin));
    cube.setCoordinate(1, new Point3d(xMax, yMin, zMin));
    cube.setCoordinate(2, new Point3d(xMax, yMax, zMin));
    cube.setCoordinate(3, new Point3d(xMin, yMax, zMin));

    cube.setCoordinate(4, new Point3d(xMin, yMin, zMin));
    cube.setCoordinate(5, new Point3d(xMax, yMin, zMin));
    cube.setCoordinate(6, new Point3d(xMax, yMin, zMax));
    cube.setCoordinate(7, new Point3d(xMin, yMin, zMax));

    cube.setCoordinate(8, new Point3d(xMin, yMin, zMin));
    cube.setCoordinate(9, new Point3d(xMin, yMax, zMin));
    cube.setCoordinate(10, new Point3d(xMin, yMax, zMax));
    cube.setCoordinate(11, new Point3d(xMin, yMin, zMax));

    cube.setCoordinate(12, new Point3d(xMax, yMax, zMax));
    cube.setCoordinate(13, new Point3d(xMin, yMax, zMax));
    cube.setCoordinate(14, new Point3d(xMin, yMin, zMax));
    cube.setCoordinate(15, new Point3d(xMax, yMin, zMax));

    cube.setCoordinate(16, new Point3d(xMax, yMax, zMax));
    cube.setCoordinate(17, new Point3d(xMin, yMax, zMax));
    cube.setCoordinate(18, new Point3d(xMin, yMax, zMin));
    cube.setCoordinate(19, new Point3d(xMax, yMax, zMin));

    cube.setCoordinate(20, new Point3d(xMax, yMax, zMax));
    cube.setCoordinate(21, new Point3d(xMax, yMin, zMax));
    cube.setCoordinate(22, new Point3d(xMax, yMin, zMin));
    cube.setCoordinate(23, new Point3d(xMax, yMax, zMin));


    //color is dumb
    Color3f green = new Color3f(0.0f, 0.5f, 0.0f);
    for (int count = 0; count < 24; count++)
      {
      cube.setColor(count, green);
      }

    Appearance app = new Appearance();
    ColoringAttributes ca = new ColoringAttributes();
    ca.setColor(green);
    app.setColoringAttributes(ca);
    PointAttributes pta = new PointAttributes(1.0f, true);
    app.setPointAttributes(pta);
    LineAttributes lna = new LineAttributes(1.0f, LineAttributes.PATTERN_SOLID, true);
    app.setLineAttributes(lna);
    //probably add a polyatt
    app.setPolygonAttributes(new PolygonAttributes(PolygonAttributes.POLYGON_FILL, PolygonAttributes.CULL_NONE, 0.0f));
    app.setTransparencyAttributes(new TransparencyAttributes(TransparencyAttributes.NICEST, 0.8f));
 
    this.setAppearance(app);
    this.setGeometry(cube);
    }

  }
