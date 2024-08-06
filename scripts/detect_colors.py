import pyrealsense2 as rs
import numpy as np
import imutils
import cv2

# detection colors
color = (255,255,255)
colors = {
        # 'blue': [np.array([95, 255, 85]), np.array([120, 255, 255])],
        #   'red': [np.array([161, 165, 127]), np.array([178, 255, 255])],
          'yellow': [np.array([23,41,133], dtype="uint8"), np.array([40,150,255], dtype="uint8")],
        #   'green': [np.array([33, 19, 105]), np.array([77, 255, 255])]
          }

# detection
def detect_marker(frame, color_thresh):
    #create mask with boundaries
    mask = cv2.inRange(frame, color_thresh[0], color_thresh[1]) 
    # find contours from mask
    contours = cv2.findContours(mask, cv2.RETR_TREE,  cv2.CHAIN_APPROX_SIMPLE)
    # grab only contours
    contours = imutils.grab_contours(contours)
    print("Found n cnts: ", len(contours))
    all_contours = []
    for contour in contours:
        # get arc length
        perimeter = cv2.arcLength(contour, True)
        # polynomial approximation
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        # find how big countour is
        area = cv2.contourArea(contour) 
        # if countour is big enough and approx a circle
        if area > 10 and len(approx) > 5:       
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00']) # calculate X position
            cy = int(M['m01'] / M['m00']) # calculate Y position
            all_contours.append([contour, cx, cy])
    return all_contours, mask
        
# realsense setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# color_path = 'V00P00A00C00_rgb.avi'
# depth_path = 'V00P00A00C00_depth.avi'
# colorwriter = cv2.VideoWriter(color_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (640,480), 1)
# depthwriter = cv2.VideoWriter(depth_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (640,480), 1)

pipeline.start(config)

mask = np.zeros((640,480))

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        
        #convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        color_image = np.asanyarray(color_frame.get_data())
        
        # colorwriter.write(color_image)
        # depthwriter.write(depth_colormap)
        
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        # for each color in colors
        for name, clr in colors.items():
            # call find_color function above
            if detect_marker(hsv, clr): 
                all_contours, mask = detect_marker(hsv, clr)
                for contour in all_contours:
                    #draw contours
                    cv2.drawContours(hsv, [contour[0]], -1, color, 1)
                    # draw circle
                    cv2.circle(hsv, (contour[1], contour[2]), 2, color, -1)  
                    # put text
                    # cv2.putText(hsv, name, (contour[1], contour[2]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1) 
                
        cv2.imshow('mask', mask)
        cv2.imshow("frame: ", hsv)
        # cv2.imshow('original', color_image)
        # cv2.imshow('Stream', depth_colormap)
        
        if cv2.waitKey(1) == ord("q"):
            break
finally:
    # colorwriter.release()
    # depthwriter.release()
    cv2.destroyAllWindows()
    pipeline.stop()