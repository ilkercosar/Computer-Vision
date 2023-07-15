import numpy as np
import cv2
import open3d as o3d
import time 

fX = 1434.5292645408758
fY = 1434.5292645408758

cXL = 276.6739676704447
cYL = 234.88530263960794

cXR = 311.7297715950476
cYR = 244.4780596837885


R = -np.array([[-0.99701627, 0.03197139, 0.07025946],
              [-0.03584117, -0.99786889, -0.05452611],
              [-0.06836645, 0.0568816, -0.9960374]])

T = np.array([-2.0042344393065044, -2.5554584377451737, -0.17528157941610553])



cameraIns = o3d.camera.PinholeCameraIntrinsic(640, 480, fX, fY, cXR, cYR)

cameraExt = np.array([[0.997016, -0.0319714, -0.0702595, -2.00423], 
                      [0.0358412, 0.997869, 0.0545261, -2.55546], 
                      [0.0683665, -0.0568816, 0.996037, -0.175282], 
                      [0, 0, 0, 1]])


def computeDisp(imgL, imgR):
    
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    
    sadWindow = 6 # 6
    numDisp = sadWindow * 16
    bSize = 11 # 11
    lmbda = 70000
    sigma = 1.7
    
    matcher = cv2.StereoBM_create(numDisparities = numDisp,
                                       blockSize = bSize)
        
    rightMatcher = cv2.ximgproc.createRightMatcher(matcher)
        
    wlsFilter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=matcher)
    wlsFilter.setLambda(lmbda)
    wlsFilter.setSigmaColor(sigma)
        
    dispLeft = matcher.compute(imgL,imgR)
    dispRight = rightMatcher.compute(imgR,imgL)

    dispLeft = np.int16(dispLeft)
    dispRight = np.int16(dispRight)
    
    filteredImg = wlsFilter.filter(dispLeft, imgL, None, dispRight)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    
    return filteredImg

    
camera1 = cv2.VideoCapture(1,cv2.CAP_DSHOW).release()
camera1 = cv2.VideoCapture(1,cv2.CAP_DSHOW)

camera2 = cv2.VideoCapture(2,cv2.CAP_DSHOW).release()
camera2 = cv2.VideoCapture(2,cv2.CAP_DSHOW)
    
if camera1.isOpened() & camera2.isOpened():
    pass
else:
    print("Cameras are not opened")
    camera1.release()
    camera2.release()
    cv2.destroyAllWindows() 
         
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Class",width=1000, height=1000)

pcdO3D = o3d.geometry.PointCloud()

addedDataFrist = False

counter = 0

option = o3d.pipelines.odometry.OdometryOption()
odoInit = np.identity(4)

rgbd1 = o3d.geometry.RGBDImage()

while True:
    
    start = time.time()
    
    counter = counter + 1
    

    
    return_value1, image1 = camera1.read()
    return_value2, image2 = camera2.read()
    
    cv2.imshow('Right',image1) 
    cv2.imshow('left',image2) 
    
    cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        
    key = cv2.waitKey(1)
        
    if key & 0xFF == ord('q'):
        break

    disp = computeDisp(imgL = image2, imgR = image1) 
            
    disp =  o3d.geometry.Image(disp)
    image1 = o3d.geometry.Image(image1)
            
    rgbd2 = o3d.geometry.RGBDImage.create_from_color_and_depth(image1, disp, convert_rgb_to_intensity=False) 
    rgbd1 = rgbd2
    
    #print(rgbd2)
    
    [success, trans,info] = o3d.pipelines.odometry.compute_rgbd_odometry(
                                                    rgbd2, 
                                                    rgbd1, 
                                                    cameraIns,
                                                    odoInit, 
                                                    o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(), 
                                                    option)
        
        
    if success:
        print(trans)
        source_pcd_color_term = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd2, cameraIns)
        source_pcd_color_term.transform(trans)
        
        if addedDataFrist == False:
            
            pcdO3D.points = source_pcd_color_term.points
            pcdO3D.colors = source_pcd_color_term.colors
            vis.add_geometry(pcdO3D)
                
            addedDataFrist = True
        
        else:
                
            pcdO3D += source_pcd_color_term
    else:
        print("An Error Occured in Odometry Success Line")
        
            
    vis.update_geometry(pcdO3D)
            
    if vis.poll_events():
       vis.update_renderer()
    
    
    stop = time.time()
    
    print("FPS = {0}".format(np.uint8(2/(stop-start)))) 

camera1.release()
camera2.release()
cv2.destroyAllWindows() 

vis.run()
vis.destroy_window() 
