import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import os
import sys
from datetime import datetime
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
import time
import math
import json
import numpy as np

# Try to import PIL for PNG/JPG support
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
        self.image_sub = self.create_subscription(Image, '/lander_camera/rgb', self.image_callback, 10)
        
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
        self.image_count = 0
        self.data_log = []

        # Data collection 
        self.start_time = time.time()
        self.data_collection_active = True

        # Control timer
        self.timer = self.create_timer(0.1, self.control_loop) 

        self.get_logger().info("Collision Controller initialized!")
        self.get_logger().info("Starting collision scenario in 3 seconds...")
        self.get_logger().info(f"Data will be saved to: {self.data_folder}")

        self.target = None

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
                    # Convert BGR to RGB
                    rgb_array = image_array[:, :, ::-1]
                    pil_image = PILImage.fromarray(rgb_array, 'RGB')
                    
                elif msg.encoding == "mono8":
                    np_arr = np.frombuffer(msg.data, dtype=np.uint8)
                    image_array = np_arr.reshape((msg.height, msg.width))
                    pil_image = PILImage.fromarray(image_array, 'L')
                    
                else:
                    self.get_logger().warn(f"Unsupported encoding: {msg.encoding}")
                    return
                
                # Save as JPG
                if self.mode != 'stop':
                    image_filename = f"photo_{self.image_count:04d}.jpg"
                    image_path = os.path.join(self.data_folder, image_filename)
                    pil_image.save(image_path, "JPEG", quality=90)
                    
                    self.image_count += 1
                    
                    if self.image_count % 50 == 0:
                        print(f"Saved {self.image_count} photos as JPG files")
                        self.get_logger().info(f"Saved {self.image_count} photos as JPG files")
                    
            else:
                # Fallback to PPM format
                if msg.encoding == "rgb8":
                    header = f"P6\n{msg.width} {msg.height}\n255\n"
                    image_filename = f"photo_{self.image_count:04d}.ppm"
                    image_path = os.path.join(self.data_folder, image_filename)
                    
                    with open(image_path, 'wb') as f:
                        f.write(header.encode('ascii'))
                        f.write(msg.data)
                    
                    self.image_count += 1
                    
                elif msg.encoding == "bgr8":
                    np_arr = np.frombuffer(msg.data, dtype=np.uint8)
                    image_array = np_arr.reshape((msg.height, msg.width, 3))
                    rgb_array = image_array[:, :, ::-1]
                    
                    header = f"P6\n{msg.width} {msg.height}\n255\n"
                    image_filename = f"photo_{self.image_count:04d}.ppm"
                    image_path = os.path.join(self.data_folder, image_filename)
                    
                    with open(image_path, 'wb') as f:
                        f.write(header.encode('ascii'))
                        f.write(rgb_array.tobytes())
                    
                    self.image_count += 1
                
                if self.image_count % 50 == 0:
                    self.get_logger().info(f"Saved {self.image_count} photos as PPM files (PIL not available)")
                
        except Exception as e:
            self.get_logger().error(f"Error saving photo: {e}")

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
        elif self.mode == 'fake-collision':
            
            target = Point()
            if self.husky1_pos.y < self.husky2_pos.y:
                if not self.target:
                    target.x = (self.husky1_pos.x + 2) if self.husky2_pos.x < self.husky1_pos.x else (self.husky1_pos.x - 2)
                    target.y = self.husky2_pos.y # di ngang thoi :D, giu nguyen y
                    self.target = target
                husky2_twist = self.move_towards_target(self.husky2_pos, self.husky2_yaw, self.target)
                husky1_twist = Twist()
            else:
                if not self.target:
                    target.x = (self.husky2_pos.x + 2) if self.husky1_pos.x < self.husky2_pos.x else (self.husky2_pos.x - 2)
                    target.y = self.husky1_pos.y
                    self.target = target
                husky1_twist = self.move_towards_target(self.husky1_pos, self.husky1_yaw, self.target)
                husky2_twist = Twist()
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
        """Stop both robots and save data"""
        self.get_logger().info("Stopping robots...")
        stop_twist = Twist()
        
        # Send stop commands multiple times to ensure they're received
        for i in range(5):
            self.husky1_pub.publish(stop_twist)
            self.husky2_pub.publish(stop_twist)
            time.sleep(0.1)
        
        # Save data log
        log_file = os.path.join(self.data_folder, "distance_log.json")
        with open(log_file, 'w') as f:
            json.dump(self.data_log, f, indent=2)
        
        # Save summary
        summary = {
            "total_duration": time.time() - self.start_time,
            "total_data_points": len(self.data_log),
            "total_photos": self.image_count,
            "data_folder": self.data_folder
        }
        
        summary_file = os.path.join(self.data_folder, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        file_type = "JPG" if PIL_AVAILABLE else "PPM"
        self.get_logger().info(f"Saved {self.image_count} {file_type} photos and data to {self.data_folder}")
        self.get_logger().info("Stop commands sent to both robots")

def main(args=None):
    rclpy.init(args=args)
    rover_controller = RoversCollisionRecreator(mode = 'fake-collision')
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

#/isaac-sim/python.sh /workspace/omnilrs/scripts/nhi/rover_controller2.py