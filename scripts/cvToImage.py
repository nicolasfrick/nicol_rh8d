
    # def _convert2RosImg(self, img, disp):
    #     # 3 mouth displays
    #     num_disp = 3 if disp == 'm' else 1
    #     disp_pixel_area = 121 # ~32 mm
    #     white_rgba = (255, 255, 255, 255)
    #     transp_rgba = (255, 255, 255, 0)
    #     black_rgba = (0, 0, 0, 255)
    #     ong_rgba = (255, 150, 50, 150)
    #     erodation_kernel = (5,5)
    #     erodation_iter =2
    #     blur = (25, 25) 
    #     # convert image to LED appearance        
    #     rgba = cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2RGBA)
    #     height, width, _ = rgba.shape
    #     for x in range(0,width):
    #         for y in range(0,height):
    #             pixel = rgba[y,x]
    #             if all(pixel == black_rgba):
    #                 rgba[y,x] = transp_rgba
    #             elif all(pixel == white_rgba):
    #                 rgba[y,x] = ong_rgba
    #     rgba  = cv2.resize(rgba, (disp_pixel_area, num_disp * disp_pixel_area), interpolation=cv2.INTER_LINEAR)
    #     rgba = cv2.rotate(rgba, cv2.ROTATE_90_CLOCKWISE)
    #     kernel = np.ones(erodation_kernel, np.uint8) 
    #     rgba = cv2.dilate(rgba, kernel, iterations=erodation_iter) 
    #     height, width, _ = rgba.shape
    #     for x in range(0,width):
    #         for y in range(0,height):
    #             pixel = rgba[y,x]
    #             if pixel[-1] > 0:
    #                 rgba[y,x] = ong_rgba
    #     rgba = cv2.blur(rgba, blur, cv2.BORDER_REPLICATE)
    #     return rgba

    # # copy cvbridge method to get rid of import
    # def cv2_to_imgmsg(self, cvim, encoding = "passthrough", header = None):
    #     if not isinstance(cvim, (np.ndarray, np.generic)):
    #         raise TypeError('Your input type is not a numpy array')
    #     img_msg = Image()
    #     img_msg.height = cvim.shape[0]
    #     img_msg.width = cvim.shape[1]
    #     if header is not None:
    #         img_msg.header = header
    #     if len(cvim.shape) < 3:
    #         # cv_type = self.dtype_with_channels_to_cvtype2(cvim.dtype, 1)
    #         cv_type = '%sC%d' % ('8U', 1)
    #     else:
    #         # cv_type = self.dtype_with_channels_to_cvtype2(cvim.dtype, cvim.shape[2])
    #         cv_type = '%sC%d' % ('8U', cvim.shape[2])
    #     if encoding == "passthrough":
    #         img_msg.encoding = cv_type
    #     else:
    #         img_msg.encoding = encoding
    #         # Verify that the supplied encoding is compatible with the type of the OpenCV image
    #         # if self.cvtype_to_name[self.encoding_to_cvtype2(encoding)] != cv_type:
    #         #     raise CvBridgeError("encoding specified as %s, but image has incompatible type %s" % (encoding, cv_type))
    #     if cvim.dtype.byteorder == '>':
    #         img_msg.is_bigendian = True
    #     img_msg.data = cvim.tostring()
    #     img_msg.step = len(img_msg.data) // img_msg.height

    #     return img_msg

    # def convertColors(self):
    #     demo_hands = cv2.imread(os.path.join(HOME, 'nicol_rh8d/datasets/OpenMM/data/2.jpeg'))

    #     # with open(os.path.join(HOME, 'nicol_rh8d/datasets/OpenMM/data/out.txt'), 'w') as f:
    #     #     json.dump(demo_hands.tolist(), f, indent=1)

    #     # convert
    #     # ong_rgba = (172,219,255)
    #     # thresh_rgba = (100, 100, 100)
    #     # height, width, _ = demo_hands.shape
    #     # for x in range(0,width):
    #     #     for y in range(0,height):
    #     #         pixel = demo_hands[y,x]
    #     #         if all(pixel <= thresh_rgba):
    #     #             demo_hands[y,x] = ong_rgba

    #     # invert
    #     demo_hands = ~demo_hands

    #     cv2.imwrite(os.path.join(HOME, 'nicol_rh8d/datasets/OpenMM/data/color_img3.jpg'), demo_hands)
    #     # cv2.imshow("image", demo_hands)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()