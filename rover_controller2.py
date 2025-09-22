import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
import os
import sys
from datetime import datetime
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
import time
import math
import json
import numpy as np
import struct
from sensor_msgs_py import point_cloud2 as pc2


try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
    print("PIL available - will save as JPG")
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available - will save as PPM")

class RoversCollisionRecreator(Node):
    def __init__(self, mode = 'collision'):
        super().__init__('rovers_collision_recreator')
        
        self.mode = mode
        # Publishers for robot control
        self.husky1_pub = self.create_publisher(Twist, '/husky1/cmd_vel', 10)
        self.husky2_pub = self.create_publisher(Twist, '/husky2/cmd_vel', 10)
        
        # Subscribers for robot positions
        self.husky1_odom_sub = self.create_subscription(Odometry, '/husky1/odom', self.husky1_odom_callback, 10)
        self.husky2_odom_sub = self.create_subscription(Odometry, '/husky2/odom', self.husky2_odom_callback, 10)
        
        # Image subscriber for data collection
        self.image_sub = self.create_subscription(Image, '/lander/rgb', self.image_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera_info', self.camera_info_callback, 10)
        # Point cloud subscriber
        self.pointcloud_sub = self.create_subscription(PointCloud2, '/lander/point_cloud', self.pointcloud_callback, 10)
        
        # Robot states
        self.husky1_pos = Point()
        self.husky2_pos = Point()
        self.husky1_yaw = 0.0
        self.husky2_yaw = 0.0

        # Control parameters
        self.max_linear_speed = 0.5  # m/s
        self.max_angular_speed = 1.0  # rad/s
        self.collision_distance = 0  # Distance to consider collision
        self.goal_distance = 0  # Distance to start slowing down

        # Data collection setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_folder = f"/workspace/omnilrs/scripts/nhi/rover_data_{timestamp}"
        if self.mode != 'stop':
            os.makedirs(self.data_folder, exist_ok=True)
            os.makedirs(os.path.join(self.data_folder, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.data_folder, "pointclouds"), exist_ok=True)
            os.makedirs(os.path.join(self.data_folder, "parameters"), exist_ok=True)
        self.image_count = 0
        self.pointcloud_count = 0
        self.data_log = []

        # sensor params
        self.camera_params = None
        self.lidar_params = {
            "horizontal_fov": 360.0,
            "vertical_fov": 50.0,
            "horizontal_resolution": 0.4,
            "vertical_resolution": 4.0,
            "min_range": 0.4,
            "max_range": 100.0,
            "rotation_rate": 20.0,
            "high_lod": True,  # RTX enabled
            "frame_id": "sim_lidar"
        }

        # Data collection 
        self.start_time = time.time()
        self.data_collection_active = True

        # Control timer
        self.timer = self.create_timer(0.1, self.control_loop) 

        self.get_logger().info("Collision Controller initialized!")
        self.get_logger().info("Starting collision scenario in 3 seconds...")
        self.get_logger().info(f"Data will be saved to: {self.data_folder}")

        self.target = None
    def camera_info_callback(self, msg):
        """Save camera parameters once"""
        if self.camera_params is None and self.mode != 'stop':
            self.camera_params = {
                "header": {
                    "frame_id": msg.header.frame_id,
                    "timestamp": time.time()
                },
                "image_dimensions": {
                    "height": msg.height,
                    "width": msg.width
                },
                "camera_matrix": {
                    "K": list(msg.k),  # 3x3 intrinsic matrix
                    "D": list(msg.d),  # Distortion coefficients
                    "R": list(msg.r),  # Rectification matrix
                    "P": list(msg.p)   # Projection matrix
                },
                "distortion_model": msg.distortion_model,
                "binning": {
                    "x": msg.binning_x,
                    "y": msg.binning_y
                },
                "roi": {
                    "x_offset": msg.roi.x_offset,
                    "y_offset": msg.roi.y_offset,
                    "height": msg.roi.height,
                    "width": msg.roi.width,
                    "do_rectify": msg.roi.do_rectify
                }
            }
            
            # Save camera parameters immediately
            camera_params_file = os.path.join(self.data_folder, "parameters", "camera_params.json")
            with open(camera_params_file, 'w') as f:
                json.dump(self.camera_params, f, indent=2)
            
            self.get_logger().info("Camera parameters saved!")
    def image_callback(self, msg):
        """Save images as JPG/PNG (if PIL available) or PPM (fallback)"""
        try:
            if PIL_AVAILABLE:
                # Save as JPG using PIL
                if msg.encoding == "rgb8":
                    np_arr = np.frombuffer(msg.data, dtype=np.uint8)
                    image_array = np_arr.reshape((msg.height, msg.width, 3))
                    pil_image = PILImage.fromarray(image_array, 'RGB')
                    
                elif msg.encoding == "bgr8":
                    np_arr = np.frombuffer(msg.data, dtype=np.uint8)
                    image_array = np_arr.reshape((msg.height, msg.width, 3))
                    rgb_array = image_array[:, :, ::-1]
                    pil_image = PILImage.fromarray(rgb_array, 'RGB')
                    
                elif msg.encoding == "mono8":
                    np_arr = np.frombuffer(msg.data, dtype=np.uint8)
                    image_array = np_arr.reshape((msg.height, msg.width))
                    pil_image = PILImage.fromarray(image_array, 'L')
                    
                else:
                    self.get_logger().warn(f"Unsupported encoding: {msg.encoding}")
                    return
                
                # Save as JPG in images subfolder
                if self.mode != 'stop':
                    image_filename = f"image_{self.image_count:04d}.jpg"
                    image_path = os.path.join(self.data_folder, "images", image_filename)
                    pil_image.save(image_path, "JPEG", quality=90)
                    
                    self.image_count += 1
                    if self.image_count % 50 == 0:
                        print(f"Saved {self.image_count} images")
                        self.get_logger().info(f"Saved {self.image_count} images")
                        
            # ... (keep your existing PPM fallback code but update paths)
                    
        except Exception as e:
            self.get_logger().error(f"Error saving image: {e}")
    def pointcloud_callback(self, msg):
        """Save point cloud data as NPY files"""
        try:
            if self.mode != 'stop':
                # Extract point cloud data
                points_list = []
                
                # Read point cloud data
                for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                    points_list.append([point[0], point[1], point[2]])
                
                if len(points_list) > 0:
                    # Convert to numpy array
                    points_array = np.array(points_list)
                    
                    # Save as NPY file (efficient for numpy arrays)
                    pointcloud_filename = f"pointcloud_{self.pointcloud_count:04d}.npy"
                    pointcloud_path = os.path.join(self.data_folder, "pointclouds", pointcloud_filename)
                    np.save(pointcloud_path, points_array)
                    
                    # Also save metadata
                    metadata = {
                        "timestamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                        "frame_id": msg.header.frame_id,
                        "num_points": len(points_list),
                        "point_step": msg.point_step,
                        "row_step": msg.row_step,
                        "is_dense": msg.is_dense,
                        "height": msg.height,
                        "width": msg.width,
                        "fields": [{"name": field.name, "offset": field.offset, 
                                "datatype": field.datatype, "count": field.count} 
                                for field in msg.fields]
                    }
                    
                    metadata_filename = f"pointcloud_{self.pointcloud_count:04d}_meta.json"
                    metadata_path = os.path.join(self.data_folder, "pointclouds", metadata_filename)
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    self.pointcloud_count += 1
                    
                    if self.pointcloud_count % 10 == 0:
                        self.get_logger().info(f"Saved {self.pointcloud_count} point clouds")
                        print(f"Point cloud {self.pointcloud_count}: {len(points_list)} points")
                else:
                    print(f"Point cloud {self.pointcloud_count}: No points detected")
                    
        except Exception as e:
            self.get_logger().error(f"Error saving point cloud: {e}")

    def save_data(self, distance):
        """Save distance data"""
        data_point = {
            "time": time.time() - self.start_time,
            "distance": distance,
            "photo_number": self.image_count,
            "husky1_pos": [self.husky1_pos.x, self.husky1_pos.y],
            "husky2_pos": [self.husky2_pos.x, self.husky2_pos.y]
        }
        self.data_log.append(data_point)

    def calculate_distance(self, pos1, pos2):
        """Calculate distance between two points"""
        dx = pos1.x - pos2.x
        dy = pos1.y - pos2.y
        return math.sqrt(dx*dx + dy*dy)

    def husky1_odom_callback(self, msg):
        self.husky1_pos = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        self.husky1_yaw = self.quaternion_to_yaw(orientation)

    def husky2_odom_callback(self, msg):
        self.husky2_pos = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        self.husky2_yaw = self.quaternion_to_yaw(orientation)

    def move_towards_target(self, current_pos, current_yaw, target_pos):
        twist = Twist()
        
        distance = self.calculate_distance(current_pos, target_pos)
        angle_to_target = self.calculate_angle_to_target(current_pos, current_yaw, target_pos)
        
        # Angular control - turn towards target
        if abs(angle_to_target) > 0.05:  # Reduced threshold
            twist.angular.z = max(-self.max_angular_speed, 
                                min(self.max_angular_speed, angle_to_target * 1.5))
        
        # Linear control - move forward even while turning
        if distance > self.goal_distance:
            twist.linear.x = self.max_linear_speed * 0.8
        elif distance > self.collision_distance:
            speed_factor = (distance - self.collision_distance) / (self.goal_distance - self.collision_distance)
            twist.linear.x = self.max_linear_speed * speed_factor * 0.5
        elif distance > 0.2:
            twist.linear.x = 0.1
        else:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        
        # Reduce linear speed when turning sharply
        if abs(angle_to_target) > 0.5:
            twist.linear.x *= 0.3
                
        return twist
    
    def calculate_angle_to_target(self, current_pos, current_yaw, target_pos):
        """Calculate angle to turn towards target"""
        dx = target_pos.x - current_pos.x
        dy = target_pos.y - current_pos.y
        target_yaw = math.atan2(dy, dx)
        angle_diff = target_yaw - current_yaw
        
        # Normalize angle to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
            
        return angle_diff
        
    def control_loop(self):
        # Calculate distance between robots
        distance = self.calculate_distance(self.husky1_pos, self.husky2_pos)
        
        # Save data point (distance and photo reference)
        self.save_data(distance)
        
        print('Current distance:', distance)
        husky1_twist, husky2_twist = Twist(), Twist()
        # Generate control commands
        if self.mode == 'collision':
            husky1_twist = self.move_towards_target(self.husky1_pos, self.husky1_yaw, self.husky2_pos)
            husky2_twist = self.move_towards_target(self.husky2_pos, self.husky2_yaw, self.husky1_pos)
        elif self.mode == "obscured-collision":
            target = Point()
            if self.husky1_pos.y < self.husky2_pos.y:
                #move husky2, husky1 stays at the same position
                if not self.target:
                    target.x = self.husky1_pos.x
                    target.y = self.husky1_pos.y + 0.5
                    self.target = target
                # husky1_twist = Twist()
                husky2_twist = self.move_towards_target(self.husky2_pos, self.husky2_yaw, self.target)
                
            else:
                #move husky1
                if not self.target:
                    target.x = self.husky2_pos.x
                    target.y = self.husky2_pos.y + 0.5
                    self.target =target
                husky2_twist = self.move_towards_target(self.husky1_pos, self.husky1_yaw, self.target)
                # husky1_twist = Twist()
            
        elif self.mode == 'fake-collision':
            
            target = Point()
            if self.husky1_pos.y < self.husky2_pos.y:
                if not self.target:
                    target.x = (self.husky1_pos.x + 2) if self.husky2_pos.x < self.husky1_pos.x else (self.husky1_pos.x - 2)
                    target.y = self.husky2_pos.y # di ngang thoi :D, giu nguyen y
                    self.target = target
                husky2_twist = self.move_towards_target(self.husky2_pos, self.husky2_yaw, self.target)
                # husky1_twist = Twist()
            else:
                if not self.target:
                    target.x = (self.husky2_pos.x + 2) if self.husky1_pos.x < self.husky2_pos.x else (self.husky2_pos.x - 2)
                    target.y = self.husky1_pos.y
                    self.target = target
                husky1_twist = self.move_towards_target(self.husky1_pos, self.husky1_yaw, self.target)
                # husky2_twist = Twist()
        elif self.mode == 'stop':
            husky1_twist, husky2_twist = Twist(), Twist()
            print("x:", self.husky1_pos)
            print("y:", self.husky2_pos)
        print("Positions:", self.husky1_pos, self.husky2_pos)
        print("Twists:", husky1_twist, husky2_twist)
        # Publish commands
        self.husky1_pub.publish(husky1_twist)
        self.husky2_pub.publish(husky2_twist) #Twist()
        
    def quaternion_to_yaw(self, q):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def stop_robots(self):
        """Stop both robots and save all data"""
        self.get_logger().info("Stopping robots and saving data...")
        stop_twist = Twist()
        
        # Send stop commands
        for i in range(5):
            self.husky1_pub.publish(stop_twist)
            self.husky2_pub.publish(stop_twist)
            time.sleep(0.1)
        
        if self.mode != 'stop':
            # Save LIDAR parameters
            lidar_params_file = os.path.join(self.data_folder, "parameters", "lidar_params.json")
            with open(lidar_params_file, 'w') as f:
                json.dump(self.lidar_params, f, indent=2)
            
            # Save complete data log
            log_file = os.path.join(self.data_folder, "synchronized_data_log.json")
            with open(log_file, 'w') as f:
                json.dump(self.data_log, f, indent=2)
            
            # Save enhanced summary
            summary = {
                "experiment_info": {
                    "mode": self.mode,
                    "total_duration": time.time() - self.start_time,
                    "start_time": self.start_time,
                    "end_time": time.time()
                },
                "data_collected": {
                    "total_images": self.image_count,
                    "total_pointclouds": self.pointcloud_count,
                    "total_data_points": len(self.data_log)
                },
                "file_structure": {
                    "images": "images/*.jpg",
                    "pointclouds": "pointclouds/*.npy (with *_meta.json)",
                    "parameters": "parameters/*.json",
                    "logs": "*.json"
                },
                "sensor_info": {
                    "camera_params_saved": self.camera_params is not None,
                    "lidar_params_saved": True
                }
            }
            
            summary_file = os.path.join(self.data_folder, "experiment_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.get_logger().info(f"Complete dataset saved to {self.data_folder}")
            self.get_logger().info(f"Images: {self.image_count}, Point Clouds: {self.pointcloud_count}")

def main(args=None):
    rclpy.init(args=args)
    rover_controller = RoversCollisionRecreator(mode = 'stop') #obscured-collision , fake-collision 
    try:
        rclpy.spin(rover_controller)
    except KeyboardInterrupt:
        rover_controller.get_logger().info("Keyboard interrupt received")
    finally:
        rover_controller.stop_robots()
        rover_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

# Step 1: add 2 rovers husky1 and 2 into isaac-sim
# Step1.2: add rocks, sensors and action graph, change lidar vertical fov from 30 to 50, xoay lidar (z axis) 180 do
# step2: heheh
# step 3: idk
# step 4: add camera facing those 2 rovers
# step 5: add topic with prim path to the added camera and publish a ros2 topic
# step 6: run the code with this cmd
#/isaac-sim/python.sh /workspace/omnilrs/scripts/nhi/rover_controller2.py