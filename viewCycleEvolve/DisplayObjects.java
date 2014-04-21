
import com.sun.j3d.utils.applet.MainFrame;
import com.sun.j3d.utils.geometry.*;
import com.sun.j3d.utils.universe.*;
import javax.media.j3d.*;
import javax.vecmath.*;
import javax.swing.*;
import javax.swing.event.ChangeListener;
import javax.swing.event.ChangeEvent;
import java.awt.event.*;
import java.awt.GridLayout;
import java.io.*;
import java.util.*;
 
public class DisplayObjects extends Switch implements ItemListener, ChangeListener
  {
  private JMenuBar menuBar;
  //contains JCheckBoxMenuItems
  private Vector menuItems;
  //store the mask
  private BitSet mask;

  private Vector actionMenuItems; //contains JCheckBoxMenuItems... that do stuff

  //store the canvas3d so you can capture images
  private Canvas3D canvas3d;

  //dealing with several link[]
  private Vector groups; //vector of Switch's
  private BitSet groupMask;

  public DisplayObjects(Vector pointArraysEvolve, 
                        Canvas3D canvas3dIn)
    {
    super();

    //store canvas
    canvas3d = canvas3dIn;

    //this bitset is for controlling which link[] shown
    groupMask = new BitSet(pointArraysEvolve.size());
    groupMask.set(0, pointArraysEvolve.size(), false); 
    groupMask.set(0); 
    groups = new Vector();
    /*
    groups = new Switch();
    groups.setCapability(ALLOW_SWITCH_WRITE);
    groups.setChildMask(groupMask);
    groups.setWhichChild(CHILD_MASK);
    */

    //menu
    menuBar = new JMenuBar();
    menuBar.setMargin(new java.awt.Insets(0,10,0,0));
    menuItems = new Vector();
    actionMenuItems = new Vector();
    JMenu tempMenu = null;
    JCheckBoxMenuItem tempItem = null;

    //make a default bitset, display only those elements we want to.
    mask = new BitSet();
    int maskCount = 0;



    tempMenu = new JMenu("Test");

    int maxLinks = 0;
    for (int ecount = 0; ecount < pointArraysEvolve.size(); ecount++)
      {
      Vector links = (Vector)pointArraysEvolve.elementAt(ecount);
      if (links.size() > maxLinks)
        {
        maxLinks = links.size();
        }
      }

    for (int ecount = 0; ecount < pointArraysEvolve.size(); ecount++)
      {
      Vector links = (Vector)pointArraysEvolve.elementAt(ecount);
      for (int lcount = 0; lcount < links.size(); lcount++)
        {
        Color3f colorMe = new Color3f((float)lcount/(maxLinks*0.5f), 
                                      (float)lcount/(maxLinks*2.0f), 
                                      (float)1.0-(lcount/maxLinks));
   

        if (groups.size() <= lcount)
          {
          Switch tempGroup = new Switch();
          tempGroup.setCapability(ALLOW_SWITCH_WRITE);
          tempGroup.setChildMask(groupMask);
          tempGroup.setWhichChild(CHILD_MASK);
          groups.add(tempGroup);
          }
        Switch group = (Switch)groups.elementAt(lcount); //now exists, or already did
        Link[] pointArrays = (Link[])links.elementAt(lcount);

        Shape3D oneSeg = new Shape3D();
        LineArray pa = new LineArray(pointArrays.length*2,
                                   LineArray.COORDINATES |
                                   LineArray.COLOR_3);
        Appearance atomApp = new Appearance();
        ColoringAttributes ca = new ColoringAttributes();
        ca.setColor(colorMe);
        atomApp.setColoringAttributes(ca);
        PointAttributes pta = new PointAttributes(4.0f, true);
        atomApp.setPointAttributes(pta);
        LineAttributes lna = new LineAttributes(4.0f, LineAttributes.PATTERN_SOLID, true);
        atomApp.setLineAttributes(lna);
        oneSeg.setAppearance(atomApp);
        for (int count = 0; count < pointArrays.length; count++)
          {
          pa.setCoordinate(count*2, pointArrays[count].beg);
          pa.setCoordinate(count*2+1, pointArrays[count].end);
          if (!pointArrays[count].rejected)
            {
            pa.setColor(count*2, colorMe);
            pa.setColor(count*2+1, colorMe);
            }
          }
        oneSeg.setGeometry(pa);
        group.addChild(oneSeg);  
        }
      for (int lcount = links.size(); lcount < maxLinks; lcount++)
        {
        if (groups.size() <= lcount)
          {
          Switch tempGroup = new Switch();
          tempGroup.setCapability(ALLOW_SWITCH_WRITE);
          tempGroup.setChildMask(groupMask);
          tempGroup.setWhichChild(CHILD_MASK);
          groups.add(tempGroup);
          }
        Switch group = (Switch)groups.elementAt(lcount); //now exists, or already did
        group.addChild(new Group()); //add null node so things work well
        }
      }
    for (int lcount = 0; lcount < maxLinks; lcount++)
      {
      Switch group = (Switch)groups.elementAt(lcount); //now exists, or already did

      addChild(group);
      mask.set(maskCount, true);
      tempItem = new JCheckBoxMenuItem("Cycles "+lcount, true);
      tempItem.addItemListener(this);
      menuItems.add(maskCount, tempItem);
      tempMenu.add(tempItem);
      maskCount++;
      }

    menuBar.add(tempMenu);


    tempMenu = new JMenu("Other Stuff");

    

    

    addChild(new BoxSides((Link[])((Vector)pointArraysEvolve.elementAt(0)).elementAt(0)));
    mask.set(maskCount, false);
    tempItem = new JCheckBoxMenuItem("Box Sides", false);
    tempItem.addItemListener(this);
    menuItems.add(maskCount, tempItem);
    tempMenu.add(tempItem);
    maskCount++;

    addChild(new BoxLines((Link[])((Vector)pointArraysEvolve.elementAt(0)).elementAt(0)));
    mask.set(maskCount, false);
    tempItem = new JCheckBoxMenuItem("Box Lines", false);
    tempItem.addItemListener(this);
    menuItems.add(maskCount, tempItem);
    tempMenu.add(tempItem);
    maskCount++;

    menuBar.add(tempMenu);



    //trying to make a snapshot menu...
    tempMenu = new JMenu("Snapshot");

    JCheckBoxMenuItem tempJMenuItem = new JCheckBoxMenuItem("Take Snapshot", false);
    tempJMenuItem.addItemListener(this);
    actionMenuItems.add(tempJMenuItem);
    tempMenu.add(tempJMenuItem);
  
    menuBar.add(tempMenu);

    //set the capabilities of this object to enable reading and writing
    setCapability(ALLOW_SWITCH_WRITE);
    setCapability(ALLOW_SWITCH_READ);

    setChildMask(mask);
    setWhichChild(CHILD_MASK);

    }

  public JMenuBar getJMenuBar()
    {
    return menuBar;
    }
  
  public void itemStateChanged(ItemEvent itemEvent)
    {
    int index = -1;
    boolean newValue = false;
    Object source = itemEvent.getItemSelectable();
    for (int count = 0; count < menuItems.size(); count++) 
      {
      JCheckBoxMenuItem tempItem = 
             (JCheckBoxMenuItem)menuItems.elementAt(count);
      if (tempItem == source)
        {
        index = count;
        }
      }
    if (index > -1 )
      {
      if (itemEvent.getStateChange() == ItemEvent.DESELECTED)
        {
        newValue = false;
        }
      else
        {
        newValue = true;
        }
      mask.set(index, newValue);
      setChildMask(mask);
      }
    else //maybe an action item
      {
      source = itemEvent.getItemSelectable();
      index = -1;
      for (int count = 0; count < actionMenuItems.size(); count++) 
        {
        //System.err.println("testing snapshot...");
        JCheckBoxMenuItem tempItem = 
              (JCheckBoxMenuItem)actionMenuItems.elementAt(count);
        if (tempItem == source)
          {
          index = count;
          tempItem.setState(false);
          }
        }
      if (index > -1 )
        {
        //    uhh,... only one event now
        //add this back later
        //canvas3d.captureImage();
        canvas3d.repaint();
        }
      }
    }


  public void stateChanged(ChangeEvent changeEvent)
    {
    JSlider source = (JSlider)changeEvent.getSource();
    int value = (int)source.getValue();
    //System.err.println(value);

    groupMask.set(0, groupMask.length(), false);
    groupMask.set(value, true);
    //System.err.println(groupMask.toString());
    for (int count = 0; count < groups.size(); count++)
      {
      ((Switch)groups.elementAt(count)).setChildMask(groupMask);
      }
    }


  }
