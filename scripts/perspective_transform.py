import cv2
import numpy as np
np.set_printoptions(suppress=True)

""" Warp an image perspective based on rpy angles."""

# camera fov
FOVY = 1.20428 # 69°

def DrawABCD(image, points_arr):
    ABCD = ['A', 'B', 'C', 'D']
    for point_ndx in range(points_arr.shape[0]):
        point = points_arr[point_ndx]
        point = (round(point[0]), round(point[1]))
        cv2.circle(image, point, 13, (255, 0, 0), thickness=-1)
        cv2.putText(image, ABCD[point_ndx], (point[0] - 40, point[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 0), thickness=6)

def CircleFixedPoints(image, fixed_points_arr):
    for point_ndx in range(fixed_points_arr.shape[0]):
        point = fixed_points_arr[point_ndx]
        point = (round(point[0]), round(point[1]))
        cv2.circle(image, point, 31, (0, 0, 255), thickness=3)

def warpMatrix(width, 
                                height,
                                theta, 
                                phi, 
                                gamma,
                                scale,
                                fovy):
    
    # setup transform
    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)
    sg = np.sin(gamma)
    cg = np.cos(gamma)

    halfFovy = fovy*0.5
    d = np.hypot(width, height)
    sideLength = scale*d/np.cos(halfFovy)
    h = d/(2.0*np.sin(halfFovy))
    n = h-(d/2.0)
    f = h+(d/2.0)

    Rtheta = np.identity(4, dtype=np.float32) # Allocate 4x4 rotation matrix around Z-axis by theta degrees
    Rphi = np.identity(4, dtype=np.float32) # Allocate 4x4 rotation matrix around X-axis by phi degrees
    Rgamma = np.identity(4, dtype=np.float32) # Allocate 4x4 rotation matrix around Y-axis by gamma degrees
    T = np.identity(4, dtype=np.float32) # Allocate 4x4 translation matrix along Z-axis by -h units
    P = np.zeros((4, 4), np.float32) # Allocate 4x4 projection matrix

    # Rtheta
    Rtheta[0,0]=Rtheta[1,1]=ct
    Rtheta[0,1]=-st
    Rtheta[1,0]=st
    # Rphi
    Rphi[1,1]=Rphi[2,2]=cp
    Rphi[1,2]=-sp
    Rphi[2,1]=sp
    # Rgamma
    Rgamma[0,0]=Rgamma[2,2]=cg
    Rgamma[0,2]=-sg
    Rgamma[2,0]=sg
    # T
    T[2,3]=-h
    # P
    P[0,0]=P[1,1]=1.0/np.tan(halfFovy)
    P[2,2]=-(f+n)/(f-n)
    P[2,3]=-(2.0*f*n)/(f-n)
    P[3,2]=-1.0
    # Compose transformations in 4x4 transformation matrix F
    F=P@T@Rphi@Rtheta@Rgamma

    # Transform points
    ptsIn = np.zeros([4*3])
    halfW=width/2
    halfH=height/2
    
    ptsIn[0]=-halfW;ptsIn[ 1]= halfH
    ptsIn[3]= halfW;ptsIn[ 4]= halfH
    ptsIn[6]= halfW;ptsIn[ 7]=-halfH
    ptsIn[9]=-halfW;ptsIn[10]=-halfH
    ptsIn[2]=ptsIn[5]=ptsIn[8]=ptsIn[11]=0

    # wrap ptsIn array and transform points by F
    ptsOutMat = cv2.perspectiveTransform(ptsIn.reshape(4,3)[None,:,:], F)[0]
    ptsOut = ptsOutMat.reshape(4*3)

    # Get 3x3 transform and warp image
    ptsInPt2f = np.zeros(4*2, dtype=np.float32).reshape(4,2)
    ptsOutPt2f = np.zeros(4*2, dtype=np.float32).reshape(4,2)
    for i in range(4):
        ptIn = np.array([ptsIn[i*3+0], ptsIn[i*3+1]], dtype=np.float32)
        ptOut = np.array([ptsOut[i*3+0], ptsOut[i*3+1]], dtype=np.float32)
        ptsInPt2f[i]  = ptIn + np.array([halfW,halfH], dtype=np.float32)
        ptsOutPt2f[i] = (ptOut + np.array([1,1], dtype=np.float32))*(sideLength*0.5)

    M = cv2.getPerspectiveTransform(ptsInPt2f, ptsOutPt2f)
    # print("ptsInPt2f\n", ptsInPt2f)
    # print("ptsOutPt2f\n", ptsOutPt2f)
    # print("M\n", M)
    
    # return warp matrix and corners [0 = Top Left corner, 1 =  Top Right corner, 
    # 2 = Bottom Right corner, 3 = Bottom Left corner]
    return M, ptsInPt2f, ptsOutPt2f

def warpImage(src,
                                theta=0,
                                phi=0,
                                gamma=0,
                                scale=1,
                                fovy=FOVY):
    
    halfFovy= fovy*0.5
    width = src.shape[1]
    height = src.shape[0]
    d=np.hypot(width,height)
    sideLength=scale*d/np.cos(halfFovy)

    # compute warp matrix
    M, corners_in, corners_out = warpMatrix(width, height, theta, phi, gamma, scale, fovy) 
    # warp image
    dst = cv2.warpPerspective(src, M, dsize=(int(sideLength), int(sideLength)))

    return dst, corners_in, corners_out

if __name__ == '__main__':
    src =  cv2.imread('/home/nic/catkin_ws/src/nicol_rh8d/datasets/perspective_transform/img_Color.png', cv2.IMREAD_COLOR)
    dst, corners_in, corners_out = warpImage(src, phi=0.872665)

    # draw corners
    CircleFixedPoints(src, corners_in)
    CircleFixedPoints(dst, corners_out)

    # mosaic image
    warped_image_size = (640, 480)
    image1 = cv2.resize(src, warped_image_size)
    warped_perspective_img1 = cv2.resize(dst, warped_image_size)
    mosaic_img = np.zeros((warped_image_size[1], 2 * warped_image_size[0], 3), dtype=np.uint8)
    mosaic_img[0: warped_image_size[1], 0: warped_image_size[0], :] = image1
    mosaic_img[0: warped_image_size[1], warped_image_size[0]:, :] = warped_perspective_img1

    # show
    cv2.namedWindow("src", cv2.WINDOW_NORMAL) # scrollable
    cv2.namedWindow("dst", cv2.WINDOW_NORMAL)
    cv2.namedWindow("mosaic", cv2.WINDOW_NORMAL)
    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.imshow("mosaic", mosaic_img)
    if cv2.waitKey(0) == ord("q"):
        cv2.destroyAllWindows()
