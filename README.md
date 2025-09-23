# ROS Topics Documentation

This document lists all available ROS topics in the system, organized by category and functionality.

## Table of Contents
- [OmniLRS (Omniverse Isaac Sim)](#omnilrs-omniverse-isaac-sim)
  - [Lens Flare](#lens-flare)
  - [Rendering](#rendering)
  - [Robots](#robots)
  - [Sun/Lighting](#sunlighting)
  - [Terrain](#terrain)
- [Camera Topics](#camera-topics)
- [Robot-Specific Topics](#robot-specific-topics)
  - [Husky1](#husky1)
  - [Husky2](#husky2)
  - [Lander](#lander)
- [System Topics](#system-topics)

---

## OmniLRS (Omniverse Isaac Sim)

### Lens Flare
Control camera lens flare effects and optical properties.

| Topic | Description |
|-------|-------------|
| `/OmniLRS/LensFlare/ApertureRotation` | Rotation of camera aperture |
| `/OmniLRS/LensFlare/EnableLensFlares` | Enable/disable lens flare effects |
| `/OmniLRS/LensFlare/FocalLength` | Camera focal length setting |
| `/OmniLRS/LensFlare/Fstop` | Camera f-stop (aperture) value |
| `/OmniLRS/LensFlare/NumBlades` | Number of aperture blades |
| `/OmniLRS/LensFlare/Scale` | Lens flare scale factor |
| `/OmniLRS/LensFlare/SensorAspectRatio` | Camera sensor aspect ratio |
| `/OmniLRS/LensFlare/SensorDiagonal` | Camera sensor diagonal size |

### Rendering
Real-time rendering configuration for RTX.

| Topic | Description |
|-------|-------------|
| `/OmniLRS/Render/EnableRTXInteractive` | Enable RTX interactive rendering |
| `/OmniLRS/Render/EnableRTXRealTime` | Enable RTX real-time rendering |

### Robots
Robot control and management commands.

| Topic | Description |
|-------|-------------|
| `/OmniLRS/Robots/Reset` | Reset individual robot |
| `/OmniLRS/Robots/ResetAll` | Reset all robots in simulation |
| `/OmniLRS/Robots/Spawn` | Spawn new robot instance |
| `/OmniLRS/Robots/Teleport` | Teleport robot to specified location |

### Sun/Lighting
Environmental lighting and sun parameters.

| Topic | Description |
|-------|-------------|
| `/OmniLRS/Sun/AngularSize` | Angular size of the sun |
| `/OmniLRS/Sun/Color` | Sun color settings |
| `/OmniLRS/Sun/ColorTemperature` | Sun color temperature (Kelvin) |
| `/OmniLRS/Sun/Intensity` | Sun light intensity |
| `/OmniLRS/Sun/Pose` | Sun position and orientation |

### Terrain
Terrain generation and randomization controls.

| Topic | Description |
|-------|-------------|
| `/OmniLRS/Terrain/EnableRocks` | Enable/disable rocks on terrain |
| `/OmniLRS/Terrain/RandomizeRocks` | Randomize rock placement |
| `/OmniLRS/Terrain/Switch` | Switch between terrain configurations |

---

## Camera Topics

| Topic | Description |
|-------|-------------|
| `/bird_view/depth` | Bird's eye view depth camera data |
| `/bird_view/rgb` | Bird's eye view RGB camera data |
| `/camera_info` | Camera calibration and metadata |

---

## Robot-Specific Topics

### Husky1
First Husky robot instance topics.

| Topic | Description |
|-------|-------------|
| `/husky1/clock` | Robot-specific clock/timing |
| `/husky1/cmd_vel` | Velocity commands for Husky1 |
| `/husky1/odom` | Odometry data from Husky1 |
| `/husky1/pointcloud` | Point cloud data from Husky1 sensors |
| `/husky1/tf` | Transform data for Husky1 |

### Husky2
Second Husky robot instance topics.

| Topic | Description |
|-------|-------------|
| `/husky2/clock` | Robot-specific clock/timing |
| `/husky2/cmd_vel` | Velocity commands for Husky2 |
| `/husky2/odom` | Odometry data from Husky2 |
| `/husky2/pointcloud` | Point cloud data from Husky2 sensors |
| `/husky2/tf` | Transform data for Husky2 |

### Lander
Lander/spacecraft vehicle topics.

| Topic | Description |
|-------|-------------|
| `/lander/depth_left` | Left stereo depth camera |
| `/lander/depth_right` | Right stereo depth camera |
| `/lander/point_cloud` | Generated point cloud data |
| `/lander/rgb_left` | Left stereo RGB camera |
| `/lander/rgb_right` | Right stereo RGB camera |

---

## System Topics

General ROS system and global topics.

| Topic | Description |
|-------|-------------|
| `/clock` | Global system clock |
| `/cmd_vel` | Global velocity commands |
| `/odom` | Global odometry data |
| `/parameter_events` | ROS parameter change events |
| `/pointcloud` | Global point cloud data |
| `/rosout` | ROS logging output |

---

## Usage Examples

### Subscribing to Topics
```bash
# Listen to bird view RGB camera
rostopic echo /bird_view/rgb

# Monitor Husky1 odometry
rostopic echo /husky1/odom

# Check lens flare settings
rostopic echo /OmniLRS/LensFlare/EnableLensFlares
```

### Publishing Commands
```bash
# Send velocity command to Husky1
rostopic pub /husky1/cmd_vel geometry_msgs/Twist "linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.5}"

# Enable lens flares
rostopic pub /OmniLRS/LensFlare/EnableLensFlares std_msgs/Bool "data: true"
```

### Topic Information
```bash
# Get topic type and details
rostopic info /lander/rgb_left

# List all active topics
rostopic list

# Show message type
rostopic type /husky1/odom
```

---

## Notes

- **OmniLRS topics** are specific to NVIDIA Omniverse Isaac Sim integration
- **Stereo camera setup** is available on the lander with left/right RGB and depth cameras
- **Multiple robot support** with separate namespaces for each Husky instance
- **Real-time rendering** can be toggled via RTX rendering topics

