/*
 * Software License Agreement (Modified BSD License)
 *
 *  Copyright (c) 2016, PAL Robotics, S.L.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of PAL Robotics, S.L. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * @author Sammy Pfeiffer
 */

#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <dynamic_reconfigure/server.h>
#include <image_transport/image_transport.h>
#include <camera_info_manager/camera_info_manager.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sstream>
#include <stdexcept>
#include <boost/filesystem.hpp>
#include <boost/assign/list_of.hpp>
#include <boost/thread/thread.hpp>
#include <queue>
#include <mutex>
#include <nicol_rh8d/VideoStreamConfig.h>

namespace fs = boost::filesystem;

namespace nicol_rh8d {

class VideoStreamNodelet: public nodelet::Nodelet {
protected:
boost::shared_ptr<ros::NodeHandle> nh, pnh;
image_transport::CameraPublisher pub;
boost::shared_ptr<dynamic_reconfigure::Server<VideoStreamConfig> > dyn_srv;
VideoStreamConfig config;
std::mutex q_mutex, s_mutex, c_mutex, p_mutex;
std::queue<cv::Mat> framesQueue;
cv::Mat frame;
boost::shared_ptr<cv::VideoCapture> cap;
std::string video_stream_provider;
std::string video_stream_provider_type;
int subscriber_num;
bool capture_thread_running;
boost::thread capture_thread;
ros::Timer publish_timer;
sensor_msgs::CameraInfo cam_info_msg;

// Based on the ros tutorial on transforming opencv images to Image messages

virtual sensor_msgs::CameraInfo get_default_camera_info_from_image(sensor_msgs::ImagePtr img){
    sensor_msgs::CameraInfo cam_info_msg;
    cam_info_msg.header.frame_id = img->header.frame_id;
    // Fill image size
    cam_info_msg.height = img->height;
    cam_info_msg.width = img->width;
    NODELET_INFO_STREAM("The image width is: " << img->width);
    NODELET_INFO_STREAM("The image height is: " << img->height);
    // Add the most common distortion model as sensor_msgs/CameraInfo says
    cam_info_msg.distortion_model = "plumb_bob";
    // Don't let distorsion matrix be empty
    cam_info_msg.D.resize(5, 0.0);
    // Give a reasonable default intrinsic camera matrix
    cam_info_msg.K = boost::assign::list_of(img->width/2.0) (0.0) (img->width/2.0)
            (0.0) (img->height/2.0) (img->height/2.0)
            (0.0) (0.0) (1.0);
    // Give a reasonable default rectification matrix
    cam_info_msg.R = boost::assign::list_of (1.0) (0.0) (0.0)
            (0.0) (1.0) (0.0)
            (0.0) (0.0) (1.0);
    // Give a reasonable default projection matrix
    cam_info_msg.P = boost::assign::list_of (img->width/2.0) (0.0) (img->width/2.0) (0.0)
            (0.0) (img->height/2.0) (img->height/2.0) (0.0)
            (0.0) (0.0) (1.0) (0.0);
    return cam_info_msg;
}


virtual void do_capture() {
    NODELET_INFO("Capture thread started");
    cv::Mat frame;
    VideoStreamConfig latest_config = config;
    ros::Rate camera_fps_rate(latest_config.set_camera_fps);

    int frame_counter = 0;
    // Read frames as fast as possible
    capture_thread_running = true;
    while (nh->ok() && capture_thread_running && subscriber_num > 0) {
        {
          std::lock_guard<std::mutex> lock(c_mutex);
          latest_config = config;
        }
        if (!cap->isOpened()) {
          NODELET_WARN("Waiting for device...");
          cv::waitKey(100);
          continue;
        }
        if (!cap->read(frame)) {
          NODELET_ERROR_STREAM_THROTTLE(1.0, "Could not capture frame (frame_counter: " << frame_counter << ")");
          if (latest_config.reopen_on_read_failure) {
            NODELET_WARN_STREAM_THROTTLE(1.0, "trying to reopen the device");
            unsubscribe();
            subscribe();
          }
        }

        frame_counter++;

        if(!frame.empty()) {
            std::lock_guard<std::mutex> g(q_mutex);
            // accumulate only until max_queue_size
            while (framesQueue.size() >= latest_config.buffer_queue_size) {
              framesQueue.pop();
            }
            framesQueue.push(frame.clone());
        }
    }
    NODELET_INFO("Capture thread finished");
}

virtual void do_publish(const ros::TimerEvent& event) {
    bool is_new_image = false;
    sensor_msgs::ImagePtr msg;
    std_msgs::Header header;
    VideoStreamConfig latest_config;

    {
      std::lock_guard<std::mutex> lock(p_mutex);
      latest_config = config;
    }

    header.frame_id = latest_config.frame_id;
    {
        std::lock_guard<std::mutex> g(q_mutex);
        if (!framesQueue.empty()){
            frame = framesQueue.front();
            framesQueue.pop();
            is_new_image = true;
        }
    }

    // Check if grabbed frame is actually filled with some content
    if(!frame.empty() && is_new_image) {
        // From http://docs.opencv.org/modules/core/doc/operations_on_arrays.html#void flip(InputArray src, OutputArray dst, int flipCode)
        // FLIP_HORIZONTAL == 1, FLIP_VERTICAL == 0 or FLIP_BOTH == -1
        // Flip the image if necessary
        if (latest_config.flip_horizontal && latest_config.flip_vertical)
          cv::flip(frame, frame, -1);
        else if (latest_config.flip_horizontal)
          cv::flip(frame, frame, 1);
        else if (latest_config.flip_vertical)
          cv::flip(frame, frame, 0);
        cv_bridge::CvImagePtr cv_image =
          boost::make_shared<cv_bridge::CvImage>(header, "bgr8", frame);
        if (latest_config.output_encoding != "bgr8")
        {
          try {
            // https://github.com/ros-perception/vision_opencv/blob/melodic/cv_bridge/include/cv_bridge/cv_bridge.h#L247
            cv_image = cv_bridge::cvtColor(cv_image, latest_config.output_encoding);
          } catch (std::runtime_error &ex) {
            NODELET_ERROR_STREAM("cannot change encoding to " << latest_config.output_encoding
                                 << ": " << ex.what());
          }
        }
        msg = cv_image->toImageMsg();
        // Create a default camera info if we didn't get a stored one on initialization
        if (cam_info_msg.distortion_model == ""){
            NODELET_WARN_STREAM("No calibration file given, publishing a reasonable default camera info.");
            cam_info_msg = get_default_camera_info_from_image(msg);
        }
        // The timestamps are in sync thanks to this publisher
        pub.publish(*msg, cam_info_msg, ros::Time::now());
    }
}

virtual void subscribe() {
  ROS_DEBUG("Subscribe");
  VideoStreamConfig& latest_config = config;
  // initialize camera info publisher
  camera_info_manager::CameraInfoManager cam_info_manager(
      *nh, latest_config.camera_name, latest_config.camera_info_url);
  // Get the saved camera info if any
  cam_info_msg = cam_info_manager.getCameraInfo();
  cam_info_msg.header.frame_id = latest_config.frame_id;

  // initialize camera
  cap.reset(new cv::VideoCapture);
  try {
    int device_num = std::stoi(video_stream_provider);
    NODELET_INFO_STREAM("Opening VideoCapture with provider: /dev/video" << device_num);
    cap->open(device_num, cv::CAP_V4L2);
  } catch (std::invalid_argument &ex) {
    NODELET_INFO_STREAM("Opening VideoCapture with provider: " << video_stream_provider);
    cap->open(video_stream_provider, cv::CAP_V4L2);
    if (!cap->isOpened()) {
      NODELET_FATAL_STREAM("Invalid 'video_stream_provider': " << video_stream_provider);
      return;
    }
  }
  NODELET_INFO_STREAM("Video stream provider type detected: " << video_stream_provider_type);

  if(latest_config.use_mjpg)
    cap->set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
  if (latest_config.width != 0 && latest_config.height != 0){
    cap->set(cv::CAP_PROP_FRAME_WIDTH, latest_config.width);
    cap->set(cv::CAP_PROP_FRAME_HEIGHT, latest_config.height);
  }
  cap->set(cv::CAP_PROP_FPS, latest_config.set_camera_fps);
  if(!cap->isOpened()){
    NODELET_ERROR_STREAM("Could not open the stream.");
    return;
  }

  cap->set(cv::CAP_PROP_BRIGHTNESS, latest_config.brightness);
  cap->set(cv::CAP_PROP_CONTRAST, latest_config.contrast);
  cap->set(cv::CAP_PROP_SATURATION, latest_config.saturation);
  cap->set(cv::CAP_PROP_SHARPNESS, latest_config.sharpness);
  cap->set(cv::CAP_PROP_GAIN, latest_config.gain);
  cap->set(cv::CAP_PROP_AUTO_WB, latest_config.white_balance_temperature_auto);
  if(!latest_config.white_balance_temperature_auto)
    cap->set(cv::CAP_PROP_WB_TEMPERATURE, latest_config.white_balance_temperature);
  cap->set(cv::CAP_PROP_AUTO_EXPOSURE, latest_config.exposure_auto);
  cap->set(cv::CAP_PROP_EXPOSURE, latest_config.exposure_absolute);
  cap->set(cv::CAP_PROP_BACKLIGHT, latest_config.backlight_compensation);

  try {
    capture_thread = boost::thread(boost::bind(&VideoStreamNodelet::do_capture, this));
    publish_timer = nh->createTimer(ros::Duration(1.0 / latest_config.fps), &VideoStreamNodelet::do_publish, this);
  } catch (std::exception& e) {
    NODELET_ERROR_STREAM("Failed to start capture thread: " << e.what());
  }
}

virtual void unsubscribe() {
  ROS_DEBUG("Unsubscribe");
  publish_timer.stop();
  capture_thread_running = false;
  capture_thread.join();
  cap.reset();
  framesQueue = {};
}

virtual void connectionCallbackImpl() {
  std::lock_guard<std::mutex> lock(s_mutex);
  subscriber_num++;
  if (subscriber_num == 1) {
    subscribe();
  }
}

virtual void disconnectionCallbackImpl() {
  std::lock_guard<std::mutex> lock(s_mutex);
  if (video_stream_provider == "videofile") {
    return;
  }

  subscriber_num--;
  if (subscriber_num == 0) {
    unsubscribe();
  }
}

virtual void connectionCallback(const image_transport::SingleSubscriberPublisher&) {
  connectionCallbackImpl();
}

virtual void infoConnectionCallback(const ros::SingleSubscriberPublisher&) {
  connectionCallbackImpl();
}

virtual void disconnectionCallback(const image_transport::SingleSubscriberPublisher&) {
  disconnectionCallbackImpl();
}

virtual void infoDisconnectionCallback(const ros::SingleSubscriberPublisher&) {
  disconnectionCallbackImpl();
}

virtual void configCallback(VideoStreamConfig& new_config, uint32_t level) {
  NODELET_DEBUG("configCallback");

  if (new_config.fps > new_config.set_camera_fps) {
    NODELET_WARN_STREAM(
        "Asked to publish at 'fps' (" << new_config.fps
        << ") which is higher than the 'set_camera_fps' (" << new_config.set_camera_fps <<
        "), we can't publish faster than the camera provides images.");
    new_config.fps = new_config.set_camera_fps;
  }

  {
    std::lock_guard<std::mutex> c_lock(c_mutex);
    std::lock_guard<std::mutex> p_lock(p_mutex);
    config = new_config;
  }

  // show current configuration
  NODELET_INFO_STREAM("Camera name: " << new_config.camera_name);
  NODELET_INFO_STREAM("Provided camera_info_url: '" << new_config.camera_info_url << "'");
  NODELET_INFO_STREAM("Publishing with frame_id: " << new_config.frame_id);
  NODELET_INFO_STREAM("Setting camera FPS to: " << new_config.set_camera_fps);
  NODELET_INFO_STREAM("Throttling to fps: " << new_config.fps);
  NODELET_INFO_STREAM("Using MJPG fourcc: " << (new_config.use_mjpg ? "true" : "false"));
  NODELET_INFO_STREAM("Setting buffer size for capturing frames to: " << new_config.buffer_queue_size);
  NODELET_INFO_STREAM("Flip horizontal image is: " << ((new_config.flip_horizontal)?"true":"false"));
  NODELET_INFO_STREAM("Flip vertical image is: " << ((new_config.flip_vertical)?"true":"false"));
  NODELET_INFO_STREAM("Brightness is: " << new_config.brightness);
  NODELET_INFO_STREAM("Contrast is: " << new_config.contrast);
  NODELET_INFO_STREAM("Saturation is: " << new_config.saturation);
  NODELET_INFO_STREAM("Gain is: " << new_config.gain);
  NODELET_INFO_STREAM("Sharpness is: " << new_config.sharpness);
  NODELET_INFO_STREAM("White balance temperature auto is: " << ((new_config.white_balance_temperature_auto)?"true":"false"));
  NODELET_INFO_STREAM("White balance temperature is: " << new_config.white_balance_temperature);
  NODELET_INFO_STREAM("Exposure auto is: " << new_config.exposure_auto);
  NODELET_INFO_STREAM("Exposure absolute is: " << new_config.exposure_absolute);
  NODELET_INFO_STREAM("Backlight compensation is: " << ((new_config.backlight_compensation)?"true":"false"));

  if (new_config.width != 0 && new_config.height != 0)
  {
    NODELET_INFO_STREAM("Forced image width is: " << new_config.width);
    NODELET_INFO_STREAM("Forced image height is: " << new_config.height);
  }

  NODELET_DEBUG_STREAM("subscriber_num: " << subscriber_num << " and level: " << level);
  if (subscriber_num > 0 && (level & 0x1))
  {
    NODELET_DEBUG("New dynamic_reconfigure config received on a parameter with configure level 1, unsubscribing and subscribing");
    unsubscribe();
    subscribe();
  }
}

virtual void onInit() {
    nh.reset(new ros::NodeHandle(getNodeHandle()));
    pnh.reset(new ros::NodeHandle(getPrivateNodeHandle()));
    subscriber_num = 0;

    // provider can be an url (e.g.: rtsp://10.0.0.1:554) or a number of device, (e.g.: 0 would be /dev/video0)
    pnh->param<std::string>("video_stream_provider", video_stream_provider, "0");
    // check file type
    try {
      int device_num = std::stoi(video_stream_provider);
      video_stream_provider_type ="videodevice";
    } catch (std::invalid_argument &ex) {
      if (video_stream_provider.find("http://") != std::string::npos ||
          video_stream_provider.find("https://") != std::string::npos){
        video_stream_provider_type = "http_stream";
      }
      else if(video_stream_provider.find("rtsp://") != std::string::npos){
        video_stream_provider_type = "rtsp_stream";
      }
      else{
        fs::file_type file_type = fs::status(fs::canonical(fs::path(video_stream_provider))).type();
        if(fs::path(video_stream_provider) != fs::canonical(fs::path(video_stream_provider)))
          NODELET_INFO_STREAM("Video stream provider path converted from: '" << fs::path(video_stream_provider) <<
        "' to its canonical path: '" << fs::canonical(fs::path(video_stream_provider)) << "'" );
        switch (file_type) {
        case fs::file_type::character_file:
        case fs::file_type::block_file:
          video_stream_provider_type = "videodevice";
          break;
        case fs::file_type::regular_file:
          video_stream_provider_type = "videofile";
          break;
        default:
          video_stream_provider_type = "unknown";
        }
      }
    }

    // set parameters from dynamic reconfigure server
    dyn_srv = boost::make_shared<dynamic_reconfigure::Server<VideoStreamConfig> >(*pnh);
    auto f = boost::bind(&VideoStreamNodelet::configCallback, this, _1, _2);
    dyn_srv->setCallback(f);

    image_transport::SubscriberStatusCallback connect_cb =
      boost::bind(&VideoStreamNodelet::connectionCallback, this, _1);
    ros::SubscriberStatusCallback info_connect_cb =
      boost::bind(&VideoStreamNodelet::infoConnectionCallback, this, _1);
    image_transport::SubscriberStatusCallback disconnect_cb =
      boost::bind(&VideoStreamNodelet::disconnectionCallback, this, _1);
    ros::SubscriberStatusCallback info_disconnect_cb =
      boost::bind(&VideoStreamNodelet::infoDisconnectionCallback, this, _1);

    bool always_capture = false;
    pnh->getParam("always_capture", always_capture);
    if(always_capture) {
        NODELET_INFO_STREAM("Always capture is enabled");
        connectionCallbackImpl();
    }

    pub = image_transport::ImageTransport(*nh).advertiseCamera(
      "image_raw", 1,
      connect_cb, disconnect_cb,
      info_connect_cb, info_disconnect_cb,
      ros::VoidPtr(), false);
}

virtual ~VideoStreamNodelet() {
  if (subscriber_num > 0)
    subscriber_num = 0;
    unsubscribe();
}
};
} // namespace

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(nicol_rh8d::VideoStreamNodelet, nodelet::Nodelet)
