
import java.applet.Applet;
import java.awt.BorderLayout;
import java.awt.event.*;
import java.awt.*;
import java.awt.GraphicsConfiguration;
import com.sun.j3d.utils.applet.MainFrame;
import com.sun.j3d.utils.geometry.*;
import com.sun.j3d.utils.universe.*;
import javax.media.j3d.*;
import javax.vecmath.*;
import com.sun.j3d.utils.behaviors.vp.*;
import com.sun.j3d.utils.behaviors.keyboard.*;
import java.io.*;
import java.util.*;
import javax.swing.*;

public class Display extends Applet 
  {
  private SimpleUniverse u = null;
  private OrbitBehavior orbit;    
  private Vector pointArrays = null;
  private DisplayObjects DisplayObjects = null;
  private Canvas3D canvas3d = null;
  private JSlider changeShown = null;

  public BranchGroup createSceneGraph(Vector pointArraysIn) 
    {
    // Create the root of the branch graph
    BranchGroup objRoot = new BranchGroup();

    // Create the TransformGroup node and initialize it to the
    // identity. Enable the TRANSFORM_WRITE capability so that
    // our behavior code can modify it at run time. Add it to
    // the root of the subgraph.
    TransformGroup objTrans = new TransformGroup();
    objTrans.setCapability(TransformGroup.ALLOW_TRANSFORM_WRITE);
    objTrans.setCapability(TransformGroup.ALLOW_TRANSFORM_READ);
    objRoot.addChild(objTrans);


    DisplayObjects = new DisplayObjects(pointArraysIn, canvas3d);
    objTrans.addChild(DisplayObjects);

    BoundingSphere bounds = new BoundingSphere(new Point3d(0.0,0.0,0.0), 1000.0);

    //lights
    AmbientLight ambient = new AmbientLight();
    ambient.setEnable(true);
    ambient.setColor(new Color3f(1.0f,1.0f, 1.0f));
    ambient.setInfluencingBounds(bounds);
    objRoot.addChild(ambient);

    DirectionalLight light1 = null;
    light1 = new DirectionalLight( );
    light1.setEnable( true );
    light1.setColor( new Color3f(0.2f, 0.2f, 0.2f) );
    light1.setDirection( new Vector3f( 1.0f, 0.0f, -1.0f ) );
    light1.setInfluencingBounds( bounds );
    objRoot.addChild( light1 );

    DirectionalLight light2 = new DirectionalLight();
    light2.setEnable(true);
    light2.setColor(new Color3f(0.2f, 0.2f, 0.2f));
    light2.setDirection(new Vector3f(-1.0f, 0.0f, 1.0f));
    light2.setInfluencingBounds(bounds);
    objRoot.addChild(light2);

    Background bg1 = new Background(new Color3f(1.0f, 1.0f, 1.0f));
    bg1.setApplicationBounds(bounds);
    objRoot.addChild(bg1);


    //fog doesn't seem to work very well :-/ 
    //objRoot.addChild(new LinearFog(new Color3f(1.0f, 1.0f, 1.0f), 5.0, 30.0));

    // Have Java 3D perform optimizations on this scene graph.
    objRoot.compile();

    return objRoot;
    }

  public Display(Vector pointArraysIn) 
    {
    pointArrays = pointArraysIn;
    }

  public void init() 
    {
    setLayout(new BorderLayout());
    GraphicsConfiguration config = SimpleUniverse.getPreferredConfiguration();

    canvas3d = new Canvas3D(config);
    add("Center", canvas3d);


    // Create a simple scene and attach it to the virtual universe
    BranchGroup scene = createSceneGraph(pointArrays);
    u = new SimpleUniverse(canvas3d);

    //add the menu bar-nothing too fancy
    JMenuBar menuBar = DisplayObjects.getJMenuBar();
    add("North", menuBar);

    changeShown = new JSlider(0, pointArrays.size()-1, 0);
    changeShown.addChangeListener(DisplayObjects);
    changeShown.setMajorTickSpacing(50);
    changeShown.setMinorTickSpacing(5);
    changeShown.setPaintTicks(true);
    changeShown.setPaintLabels(true);
    add("South", changeShown);

    // This will move the ViewPlatform back a bit so the
    // objects in the scene can be viewed.

    Transform3D t3d = new Transform3D();
    double maxX = -100000, maxY = -100000, maxZ = -100000;
    double minX = 100000, minY = 100000, minZ = 100000;
    int distBack = 150;
    Link[] pointArraysTemp = (Link[])((Vector)pointArrays.elementAt(0)).elementAt(0);
    for (int count = 0; count < pointArraysTemp.length; count++)
      {
      Point3d current = (Point3d)((Link)pointArraysTemp[count]).beg;
      maxX = (maxX > current.x ? maxX : current.x);
      maxY = (maxY > current.y ? maxY : current.y);
      maxZ = (maxZ > current.z ? maxZ : current.z);
      minX = (minX < current.x ? minX : current.x);
      minY = (minY < current.y ? minY : current.y);
      minZ = (minZ < current.z ? minZ : current.z);
      } 

    t3d.set(new Vector3d((maxX+minX)/2, (minY+maxY)/2, distBack+(maxZ+minZ)/2));
    u.getViewingPlatform().getViewPlatformTransform().setTransform(t3d);

    //u.getViewingPlatform().setNominalViewingTransform();

    orbit = new OrbitBehavior(canvas3d, OrbitBehavior.REVERSE_ALL );

    BoundingSphere bounds = new BoundingSphere
      (new Point3d((maxX+minX)/2, (minY+maxY)/2, (maxZ+minZ)/2), 200.0);
    orbit.setSchedulingBounds(bounds);
    orbit.setRotationCenter(new Point3d((maxX+minX)/2, (minY+maxY)/2, (maxZ+minZ)/2));
    u.getViewingPlatform().setViewPlatformBehavior(orbit);
    View view = u.getViewer().getView();
    view.setFrontClipPolicy(View.VIRTUAL_EYE);
    view.setBackClipPolicy(View.VIRTUAL_EYE);
    view.setBackClipDistance(2000.0);
    view.setFrontClipDistance(2.0);  
   
    u.addBranchGraph(scene);
    }

  public void destroy() 
    {
    //u.cleanup();
    }

  public void captureImage()
    {
    //re-enable later
    //canvas3d.captureImage();
    }

  }
