import os
import logging
import argparse
import cv2
import numpy as np
import operator

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

# def main(
#         imageFilepath1,
#         imageFilepath2,
#         outputDirectory
# ):
#     logging.info("compute_transforms.main()")

#     if not os.path.exists(outputDirectory):
#         os.makedirs(outputDirectory)

#     # Load the images
#     image1 = cv2.imread(imageFilepath1, cv2.IMREAD_COLOR)
#     image2 = cv2.imread(imageFilepath2, cv2.IMREAD_COLOR)

#     feature_points1 = np.array([[1069, 560], [681, 674], [616, 356], [1026, 288]], dtype=np.float32)
#     feature_points2 = np.array([[3286, 3024], [568, 2014], [1960, 334], [4384, 1062]], dtype=np.float32)
#     #                                                           rl                         ll                      lu                     ru
#     warped_image_size = (640, 480)
#     warped_feature_points = np.array([[0, 10], [640, 10], [640, 470], [0, 470]], dtype=np.float32)

#     # Perspective transform
#     # With OpenCV
#     perspective_mtx1 = cv2.getPerspectiveTransform(feature_points1, warped_feature_points)
#     perspective_mtx2 = cv2.getPerspectiveTransform(feature_points2, warped_feature_points)
#     logging.info("perspective_mtx1 =\n{}".format(perspective_mtx1))
#     logging.info("perspective_mtx2 =\n{}".format(perspective_mtx2))
#     # Warp perspective
#     warped_perspective_img1 = cv2.warpPerspective(image1, perspective_mtx1, dsize=warped_image_size)
#     warped_perspective_img2 = cv2.warpPerspective(image2, perspective_mtx2, dsize=warped_image_size)
#     CircleFixedPoints(warped_perspective_img1, warped_feature_points)
#     CircleFixedPoints(image1, feature_points1)
#     CircleFixedPoints(warped_perspective_img2, warped_feature_points)
#     CircleFixedPoints(image2, feature_points2)

#     warped_perspective_img1_filepath = os.path.join(outputDirectory, "marker_warped.jpg")
#     cv2.imwrite(warped_perspective_img1_filepath, warped_perspective_img1)
#     # warped_perspective_img2_filepath = os.path.join(outputDirectory, "bookComputeTransforms_main_warpedPerspective2.png")
#     # cv2.imwrite(warped_perspective_img2_filepath, warped_perspective_img2)


#     # Create a mosaic image
#     # image1 = cv2.resize(image1, warped_image_size)
#     # image2 = cv2.resize(image2, warped_image_size)
#     warped_perspective_img1 = cv2.resize(warped_perspective_img1, warped_image_size)
#     warped_perspective_img2 = cv2.resize(warped_perspective_img2, warped_image_size)
#     # mosaic_img = np.zeros((2 * warped_image_size[1], 2 * warped_image_size[0], 3), dtype=np.uint8)
#     # mosaic_img[0: warped_image_size[1], 0: warped_image_size[0], :] = resized_img1
#     # mosaic_img[0: warped_image_size[1], warped_image_size[0]:, :] = warped_perspective_img1
#     # mosaic_img[warped_image_size[1]:, 0: warped_image_size[0], :] = resized_img2
#     # mosaic_img[warped_image_size[1]:, warped_image_size[0]:, :] = warped_perspective_img2
#     # mosaic_img_filepath = os.path.join(outputDirectory, "bookComputeTransforms_main_mosaic.png")
#     # cv2.imwrite(mosaic_img_filepath, mosaic_img)

#     cv2.namedWindow("Perspective image1", cv2.WINDOW_NORMAL) # scrollable
#     cv2.imshow("Perspective image1", image1)
#     cv2.imshow("Perspective warped image1", warped_perspective_img1)
#     cv2.namedWindow("Perspective image", cv2.WINDOW_NORMAL) # scrollable
#     cv2.imshow("Perspective image", image2)
#     cv2.imshow("Perspective warped image", warped_perspective_img2)
#     # window_mosaic = cv2.namedWindow("Mosaic", cv2.WINDOW_NORMAL)
#     # cv2.imshow("Mosaic", mosaic_img)
#     if cv2.waitKey(0) == ord("q"):
#         cv2.destroyAllWindows()

# def DrawABCD(image, points_arr):
#     ABCD = ['A', 'B', 'C', 'D']
#     for point_ndx in range(points_arr.shape[0]):
#         point = points_arr[point_ndx]
#         point = (round(point[0]), round(point[1]))
#         cv2.circle(image, point, 13, (255, 0, 0), thickness=-1)
#         cv2.putText(image, ABCD[point_ndx], (point[0] - 40, point[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 0),
#                     thickness=6)

# def CircleFixedPoints(image, fixed_points_arr):
#     for point_ndx in range(fixed_points_arr.shape[0]):
#         point = fixed_points_arr[point_ndx]
#         point = (round(point[0]), round(point[1]))
#         cv2.circle(image, point, 31, (0, 0, 255), thickness=3)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--imageFilepath1', help="The filepath to the 1st image. Default: /home/nic/catkin_ws/src/nicol_rh8d/datasets/perspective_transform/book1.jpg'", default='/home/nic/catkin_ws/src/nicol_rh8d/datasets/perspective_transform/marker.jpg')
#     parser.add_argument('--imageFilepath2', help="The filepath to the 2nd image. Default: '/home/nic/catkin_ws/src/nicol_rh8d/datasets/perspective_transform/book2.jpg'", default='/home/nic/catkin_ws/src/nicol_rh8d/datasets/perspective_transform/book2.jpg')
#     parser.add_argument('--outputDirectory', help="The output directory. Default: '/home/nic/catkin_ws/src/nicol_rh8d/datasets/perspective_transform'", default='/home/nic/catkin_ws/src/nicol_rh8d/datasets/perspective_transform')
#     args = parser.parse_args()
#     main(args.imageFilepath1,
#          args.imageFilepath2,
#          args.outputDirectory)


def rad2Deg(rad):
    return rad*(180/np.pi)

def deg2Rad(deg):
    return deg*(np.pi/180)

def warpMatrix(sz, # size
                                theta, # r
                                phi, # p
                                gamma, # y
                                scale,
                                fovy,
                                corners=False):
    st = np.sin(deg2Rad(theta))
    ct = np.cos(deg2Rad(theta))
    sp = np.sin(deg2Rad(phi))
    cp = np.cos(deg2Rad(phi))
    sg = np.sin(deg2Rad(gamma))
    cg = np.cos(deg2Rad(gamma))

    halfFovy = fovy*0.5
    d = np.hypot(sz[0],sz[1])
    sideLength = scale*d/np.cos(deg2Rad(halfFovy))
    h = d/(2.0*np.sin(deg2Rad(halfFovy)))
    n = h-(d/2.0)
    f = h+(d/2.0)

    Rtheta = np.identity(4, dtype=np.float64) # Allocate 4x4 rotation matrix around Z-axis by theta degrees
    Rphi = np.identity(4, dtype=np.float64) # Allocate 4x4 rotation matrix around X-axis by phi degrees
    Rgamma = np.identity(4, dtype=np.float64) # Allocate 4x4 rotation matrix around Y-axis by gamma degrees

    T = np.identity(4, dtype=np.float64) # Allocate 4x4 translation matrix along Z-axis by -h units
    P = np.zeros((4, 4), np.float64) # Allocate 4x4 projection matrix

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
    P[0,0]=P[1,1]=1.0/np.tan(deg2Rad(halfFovy))
    P[2,2]=-(f+n)/(f-n)
    P[2,3]=-(2.0*f*n)/(f-n)
    P[3,2]=-1.0
    # Compose transformations in 4x4 transformation matrix F
    F=P*T*Rphi*Rtheta*Rgamma # Matrix-multiply to produce master matrix
    F =  np.delete(np.delete(F, -1, axis=1), -1, axis=0) # reduce to 3x3
    # print(F)
    # Transform 4x4 points
    ptsIn = np.zeros([4*3])
    ptsOut = np.ones([4*3]) * sideLength*0.5
    halfW=sz[0]/2
    halfH=sz[1]/2

    w, h = 512, 512
    src = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    dst = np.array(
        [[300, 350], [800, 300], [900, 923], [161, 923]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    print()
    print(src)
    print(dst)
    print(m)
    result = cv2.perspectiveTransform(src[None, :, :], m)
    print(result)
    print()

    # Set Z component to zero for all 4 components
    ptsInMat = np.array([[-halfW, halfH], [halfW, halfH], [halfW, -halfH], [-halfW, -halfH]], dtype=np.float32)
    ptsOutMat = cv2.perspectiveTransform(ptsInMat[None, :, :], F) # Transform points
    # print(ptsOutMat)
    # Get 3x3 transform and warp image
    # Point2f ptsInPt2f[4];
    # Point2f ptsOutPt2f[4];
    ptsInPt2f = []
    ptsOutPt2f = []

    for i in range(4):
        ptIn = tuple((ptsIn[i*3+0], ptsIn[i*3+1]))
        ptOut = tuple((ptsOut[i*3+0], ptsOut[i*3+1]))
        ptsInPt2f.append(tuple( (map(operator.add, ptIn, tuple((halfW,halfH)))) ))
        ptsOutPt2f.append(tuple((ptsOut[i*3+0], ptsOut[i*3+1]))) # tuple(map(operator.mul(tuple(map(operator.add, ptOut, tuple((1.0,1.0)))), (sideLength*0.5)))))
    print(np.array(ptsInPt2f))
    print(np.array(ptsOutPt2f))
    M = cv2.getPerspectiveTransform(src, dst)

    # Load corners vector
    if corners:
        corners = []
        corners.append(ptsOutPt2f[0]) # Push Top Left corner
        corners.append(ptsOutPt2f[1]) # Push Top Right corner
        corners.append(ptsOutPt2f[2]) # Push Bottom Right corner
        corners.append(ptsOutPt2f[3]) # Push Bottom Left corner

    return M, corners

def warpImage():
    src =  cv2.imread('/home/nic/catkin_ws/src/nicol_rh8d/datasets/perspective_transform/marker.jpg', cv2.IMREAD_COLOR)
    theta = 5
    phi = 50
    gamma = 0
    scale = 1
    fovy = 69
    halfFovy= fovy*0.5
    d=np.hypot(src.shape[0],src.shape[1])
    sideLength=scale*d/np.cos(deg2Rad(halfFovy))

    M, corners = warpMatrix(src.shape,theta,phi,gamma, scale,fovy,False) # Compute warp matrix
    cv2.typing.Size(1,1)
    dst = cv2.warpPerspective(src, M, dsize=[sideLength,sideLength])
    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    if cv2.waitKey(0) == ord("q"):
        cv2.destroyAllWindows()

if __name__ == '__main__':
    warpImage()