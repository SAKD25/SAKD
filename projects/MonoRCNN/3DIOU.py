# 3D IoU caculate code for 3D object detection 
# Kent 2018/12

import numpy as np
from scipy.spatial import ConvexHull
from numpy import *

def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.

   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**

   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def is_clockwise(p):
    x = p[:,0]
    y = p[:,1]
    return np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)) > 0

def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (kent): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
   
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])

    inter_vol = inter_area * max(0.0, ymax-ymin)
    
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d

# ----------------------------------
# Helper functions for evaluation
# ----------------------------------

def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.

    Input:
        box_size: tuple of (length,wide,height)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    R = roty(heading_angle)
    l,w,h = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d

    
if __name__=='__main__':
    print('------------------')
    # get_3d_box(box_size, heading_angle, center)
    #corners_3d_ground  = get_3d_box((1.497255,1.644981, 3.628938), -1.531692, (2.882992 ,1.698800 ,20.785644))
    #corners_3d_predict = get_3d_box((1.458242, 1.604773, 3.707947), -1.549553, (2.756923, 1.661275, 20.943280 ))

    ###42
    #corners_3d_ground = get_3d_box((1.55, 1.63, 3.92), -1.65, (-6.44, 1.67, 14.70))
    #corners_3d_ground = get_3d_box((1.60, 1.76, 3.84), -1.43, (2.56, 1.53, 22.05))
    #corners_3d_ground = get_3d_box((1.38, 1.56, 3.83), -1.64, (-6.94, 1.57, 22.64))
    #corners_3d_ground = get_3d_box((1.39, 1.56, 3.45), -1.08, (10.66, 1.49, 49.41))


    #corners_3d_predict_B3KD = get_3d_box((1.53, 1.64, 4.22), -1.60, (-5.93, 1.59, 13.58))  #B3KD
    #corners_3d_predict_B3KD = get_3d_box((1.49, 1.60, 3.37), -1.38, (2.57, 1.48, 21.41))  # B3KD
    #corners_3d_predict_B3KD = get_3d_box((1.49, 1.58, 3.80), -1.64, (-7.38, 1.72, 23.92))  # B3KD
    #corners_3d_predict_B3KD = get_3d_box((1.58, 1.63, 3.95), 1.93, (10.64, 1.60, 49.69))  # B3KD


    #corners_3d_predict_SimKD = get_3d_box((1.56, 1.63, 4.14), -1.57, (-5.89, 1.59, 13.55))  #SimKD
    #corners_3d_predict_SimKD = get_3d_box((1.53, 1.59, 3.75), -1.02, (2.72, 1.52, 22.84))  # SimKD
    #corners_3d_predict_SimKD = get_3d_box((1.55, 1.61, 4.34), 1.29, (-8.15, 1.80, 25.49))  # SimKD

    # ###134
    # corners_3d_ground = get_3d_box((1.74, 0.72, 1.06), -1.56, (-0.96, 1.52, 6.44))
    #
    # corners_3d_predict_B3KD = get_3d_box((1.77, 0.65, 0.96), 1.56, (-0.93, 1.51, 6.13))  #B3KD
    #
    # corners_3d_predict_SimKD = get_3d_box((1.75, 0.61, 0.96), -1.94, (-0.93, 1.52, 6.17))  # SimKD

    # ###148
    #corners_3d_ground = get_3d_box((1.38, 1.45, 3.63), 1.57, (-8.55, 1.90, 19.67))
    #corners_3d_ground = get_3d_box((1.65, 1.67, 3.64), 1.57, (0.16, 1.91, 45.33))
    corners_3d_ground = get_3d_box((1.41, 1.53, 3.98), 1.53, (-8.65, 2.03, 43.94))
    #
    #corners_3d_predict_B3KD = get_3d_box((1.44, 1.60, 3.76), -1.55, (-8.61, 1.93, 19.47))  # B3KD
    #corners_3d_predict_B3KD = get_3d_box((1.57, 1.60, 4.28), -1.58, (-0.07, 1.77, 42.50))  #B3KD
    corners_3d_predict_B3KD = get_3d_box((1.56, 1.69, 4.29), -1.49, (-8.71, 2.06, 43.63))  # B3KD
    #
    #corners_3d_predict_SimKD = get_3d_box((1.42, 1.59, 4.02), -1.94, (-8.34, 1.88, 18.70))  # SimKD
    corners_3d_predict_SimKD = get_3d_box((1.61, 1.61, 4.37), 1.58, (-0.08, 1.74, 41.80))  # SimKD


    # ###165
    # corners_3d_ground = get_3d_box((1.63, 1.78, 4.13), 1.56, (-13.65, 2.48, 32.02))
    #
    # corners_3d_predict_B3KD = get_3d_box((1.65, 1.71, 3.94), 1.60, (-13.58, 2.45, 31.89))  #B3KD
    #
    # corners_3d_predict_SimKD = get_3d_box((1.61, 1.64, 4.24), 1.33, (-13.49, 2.43, 31.71))  # SimKD


    (IOU_3d_B3KD,IOU_2d)=box3d_iou(corners_3d_predict_B3KD,corners_3d_ground)
    (IOU_3d_SimKD, IOU_2d) = box3d_iou(corners_3d_predict_SimKD, corners_3d_ground)



    print ('B3KD:',IOU_3d_B3KD,'SimKD:',IOU_3d_SimKD) #3d IoU/ 2d IoU of BEV(bird eye's view)
      
