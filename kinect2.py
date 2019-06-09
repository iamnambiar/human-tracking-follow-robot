#!/usr/bin/env python


import rospy
import sys
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
import numpy as np
import serial

class cvBridgeDemo():
    
    
    def __init__(self):
        self.node_name = "cv_bridge_demo"
        
        rospy.init_node(self.node_name)
        
        # What we do during shutdown
        rospy.on_shutdown(self.cleanup)
        
        # Create the OpenCV display window for the RGB image
        self.cv_window_name = self.node_name

        # Create the cv_bridge object
        self.bridge = CvBridge()
        
        self.pub = rospy.Publisher('arduino', String, queue_size=10)
        #self.rate = rospy.Rate(10)

        # Subscribe to the camera image and depth topics and set
        # the appropriate callbacks

        rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
        rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)

        self.ser = serial.Serial('/dev/ttyACM0', 115200)
        
        rospy.Timer(rospy.Duration(0.03), self.show_img_cb)
        rospy.loginfo("Waiting for image topics...")


    def show_img_cb(self,event):
    	try: 


		cv2.namedWindow("RGB_Image", cv2.WINDOW_NORMAL)
		cv2.moveWindow("RGB_Image", 25, 75)
		
		cv2.namedWindow("Processed_Image", cv2.WINDOW_NORMAL)
		cv2.moveWindow("Processed_Image", 500, 75)

        	# And one for the depth image
		cv2.moveWindow("Depth_Image", 950, 75)
		cv2.namedWindow("Depth_Image", cv2.WINDOW_NORMAL)


        	cv2.imshow("RGB_Image",self.frame)
        	cv2.imshow("Processed_Image",self.display_image)
        	cv2.imshow("Depth_Image",self.depth_display_image)
      		cv2.waitKey(3)
    	except:
		pass


    def image_callback(self, ros_image):
        # Use cv_bridge() to convert the ROS image to OpenCV format
        try:
            self.frame = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError, e:
            print e
	    pass

        # Convert the image to a Numpy array since most cv2 functions
        # require Numpy arrays.
        frame = np.array(self.frame, dtype=np.uint8)
        
        # Process the frame using the process_image() function
        self.display_image = self.process_image(frame)
                       
    def depth_callback(self, ros_image):
        # Use cv_bridge() to convert the ROS image to OpenCV format
        try:
            # The depth image is a single-channel float32 image
            depth_image = self.bridge.imgmsg_to_cv2(ros_image, "32FC1")
        except CvBridgeError, e:
            print e
	    pass
        # Convert the depth image to a Numpy array since most cv2 functions
        # require Numpy arrays.
        depth_array = np.array(depth_image, dtype=np.float32)
                
        # Normalize the depth image to fall between 0 (black) and 1 (white)
        cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
        
        # Process the depth image
        self.depth_display_image = self.process_depth_image(depth_array)
    
        # Display the result
        #cv2.imshow("Depth Image", self.depth_display_image)
          
    def process_image(self, frame):
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w, d = frame.shape
        
        # define range of color in hsv
        lower = np.array([65,60,60])    #green
        upper = np.array([80,255,255])
        mask1 = cv2.inRange(hsv, lower, upper)

        lower = np.array([105,75,0])    #blue
        upper = np.array([110,100,190])
        mask2 = cv2.inRange(hsv, lower, upper)

        mask = mask2
        masked = cv2.bitwise_and(frame, frame, mask=mask)
        #mask = cv2.blur(mask, (5,5))
        M = cv2.moments(mask)
        if M['m00'] > 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.circle(frame, (cx, cy), 20, (0,0,255), -1)
        str = ''
        if cx < (w/3):
            str = 'L'
        elif cx < (2*w/3):
            str = 'S'
        else:
            str = 'R'
        rospy.loginfo(str)
        self.pub.publish(str)
        #self.rate.sleep()
        self.ser.write(str)
        return frame
    
    def process_depth_image(self, frame):
        # Just return the raw image for this demo
        return frame
    
    def cleanup(self):
        print "Shutting down vision node."
        cv2.destroyAllWindows()   
    
def main(args):       
    try:
        cvBridgeDemo()
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down vision node."
        cv.DestroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

