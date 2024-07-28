import logging
import argparse
import os
import cv2
import numpy as np

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

def main(
        imageFilepath1,
        imageFilepath2,
        outputDirectory
):
    logging.info("compute_transforms.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # Load the images
    image1 = cv2.imread(imageFilepath1, cv2.IMREAD_COLOR)
    image2 = cv2.imread(imageFilepath2, cv2.IMREAD_COLOR)

    feature_points1 = np.array([[702, 1596], [2480, 649], [4120, 1791], [2325, 3022]], dtype=np.float32)
    feature_points2 = np.array([[3286, 3024], [568, 2014], [1960, 334], [4384, 1062]], dtype=np.float32)
    warped_image_size = (1200, 1200)
    warped_feature_points = np.array([[100, 100], [1100, 100], [1100, 1100], [100, 1100]], dtype=np.float32)


    # Perspective transform
    # With OpenCV
    perspective_mtx1 = cv2.getPerspectiveTransform(feature_points1, warped_feature_points)
    perspective_mtx2 = cv2.getPerspectiveTransform(feature_points2, warped_feature_points)
    logging.info("perspective_mtx1 =\n{}".format(perspective_mtx1))
    logging.info("perspective_mtx2 =\n{}".format(perspective_mtx2))
    # Warp perspective
    warped_perspective_img1 = cv2.warpPerspective(image1, perspective_mtx1, dsize=warped_image_size)
    warped_perspective_img2 = cv2.warpPerspective(image2, perspective_mtx2, dsize=warped_image_size)
    CircleFixedPoints(warped_perspective_img1, warped_feature_points)
    CircleFixedPoints(warped_perspective_img2, warped_feature_points)

    warped_perspective_img1_filepath = os.path.join(outputDirectory, "bookComputeTransforms_main_warpedPerspective1.png")
    cv2.imwrite(warped_perspective_img1_filepath, warped_perspective_img1)
    warped_perspective_img2_filepath = os.path.join(outputDirectory, "bookComputeTransforms_main_warpedPerspective2.png")
    cv2.imwrite(warped_perspective_img2_filepath, warped_perspective_img2)

    window_warped = cv2.namedWindow("Perspective warped image", cv2.WINDOW_NORMAL)
    cv2.imshow("Perspective warped image", warped_perspective_img1)
    cv2.waitKey(0)
    cv2.imshow("Perspective warped image", warped_perspective_img2)
    cv2.waitKey(0)

    # Create a mosaic image
    resized_img1 = cv2.resize(image1, warped_image_size)
    resized_img2 = cv2.resize(image2, warped_image_size)
    mosaic_img = np.zeros((2 * warped_image_size[1], 2 * warped_image_size[0], 3), dtype=np.uint8)
    mosaic_img[0: warped_image_size[1], 0: warped_image_size[0], :] = resized_img1
    mosaic_img[0: warped_image_size[1], warped_image_size[0]:, :] = warped_perspective_img1
    mosaic_img[warped_image_size[1]:, 0: warped_image_size[0], :] = resized_img2
    mosaic_img[warped_image_size[1]:, warped_image_size[0]:, :] = warped_perspective_img2
    mosaic_img_filepath = os.path.join(outputDirectory, "bookComputeTransforms_main_mosaic.png")
    cv2.imwrite(mosaic_img_filepath, mosaic_img)

    window_mosaic = cv2.namedWindow("Mosaic", cv2.WINDOW_NORMAL)
    cv2.imshow("Mosaic", mosaic_img)
    cv2.waitKey(0)



def DrawABCD(image, points_arr):
    ABCD = ['A', 'B', 'C', 'D']
    for point_ndx in range(points_arr.shape[0]):
        point = points_arr[point_ndx]
        point = (round(point[0]), round(point[1]))
        cv2.circle(image, point, 13, (255, 0, 0), thickness=-1)
        cv2.putText(image, ABCD[point_ndx], (point[0] - 40, point[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 0),
                    thickness=6)

def CircleFixedPoints(image, fixed_points_arr):
    for point_ndx in range(fixed_points_arr.shape[0]):
        point = fixed_points_arr[point_ndx]
        point = (round(point[0]), round(point[1]))
        cv2.circle(image, point, 31, (0, 0, 255), thickness=3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageFilepath1', help="The filepath to the 1st image. Default: /home/nic/catkin_ws/src/nicol_rh8d/datasets/perspective_transform/book1.jpg'", default='/home/nic/catkin_ws/src/nicol_rh8d/datasets/perspective_transform/book1.jpg')
    parser.add_argument('--imageFilepath2', help="The filepath to the 2nd image. Default: '/home/nic/catkin_ws/src/nicol_rh8d/datasets/perspective_transform/book2.jpg'", default='/home/nic/catkin_ws/src/nicol_rh8d/datasets/perspective_transform/book2.jpg')
    parser.add_argument('--outputDirectory', help="The output directory. Default: '/home/nic/catkin_ws/src/nicol_rh8d/datasets/perspective_transform'", default='/home/nic/catkin_ws/src/nicol_rh8d/datasets/perspective_transform')
    args = parser.parse_args()
    main(args.imageFilepath1,
         args.imageFilepath2,
         args.outputDirectory)