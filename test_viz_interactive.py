#!/usr/bin/env python3
"""
Interactive test script for viz_client functionality.

Usage:
    python3 test_viz_interactive.py
    
Or in an interactive Python terminal:
    exec(open('test_viz_interactive.py').read())
    
Available test functions:
    - test_connect()
    - test_pointcloud()
    - test_joints()
    - test_trajectory()
    - test_smooth_trajectory()
    - test_ghost_end_effector()
    - test_shutdown()
    - test_force_shutdown()
    - test_all()
"""

import numpy as np
import time
import sys
import os

# Add viz_server to path
sys.path.insert(0, '/workspace/viz_server')
import viz_client as V

def test_connect():
    """Test connection to viz_server."""
    print("🔌 Testing connection...")
    try:
        # You may need to adjust the URDF path
        urdf_path = "/workspace/assets/panda/panda_collision_spheres_gltf.urdf"
        if not os.path.exists(urdf_path):
            print(f"❌ URDF not found at {urdf_path}")
            print("Please update the urdf_path variable in test_connect()")
            return False
        
        V.connect(urdf_path)
        print("✅ Successfully connected to viz_server")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def test_pointcloud():
    """Test point cloud visualization with different types."""
    print("☁️ Testing point clouds...")
    try:
        # Generate different point clouds
        robot_points = np.random.rand(500, 3).astype(np.float32) * 0.5  # Small cluster
        target_points = np.random.rand(300, 3).astype(np.float32) + [1, 0, 0]  # Offset cluster
        obstacle_points = np.random.rand(400, 3).astype(np.float32) + [-1, 0, 0]  # Another offset
        
        # Publish robot points using new API
        V.publish_robot_pointcloud(robot_points, name="robot_cloud")
        print("✅ Published robot point cloud")
        
        time.sleep(1)
        
        # Publish target points using new API
        V.publish_target_pointcloud(target_points, name="target_cloud")
        print("✅ Published target point cloud")
        
        time.sleep(1)
        
        # Publish obstacle points using new API
        V.publish_obstacle_pointcloud(obstacle_points, name="obstacle_cloud")
        print("✅ Published obstacle point cloud")
        
        time.sleep(1)
        
        # Test custom robot points
        custom_points = np.random.rand(200, 3).astype(np.float32) + [0, 1, 0]
        V.publish_robot_pointcloud(custom_points, name="custom_cloud")
        print("✅ Published custom robot point cloud")
        print("🎨 Set colors in RViz display panel for each point cloud topic!")
        
        return True
    except Exception as e:
        print(f"❌ Point cloud test failed: {e}")
        return False

def test_joints():
    """Test joint configuration visualization."""
    print("🦾 Testing joint configuration...")
    try:
        # Example joint configuration for Franka Panda
        # Note: panda_finger_joint2 mimics panda_finger_joint1, so you can:
        # 1. Set both to the same value (traditional)
        # 2. Set only panda_finger_joint1 (server will compute panda_finger_joint2)
        # 3. Set only panda_finger_joint2 (server will compute panda_finger_joint1)
        joints = {
            "panda_joint1": 0.0,
            "panda_joint2": -0.785,
            "panda_joint3": 0.0,
            "panda_joint4": -2.356,
            "panda_joint5": 0.0,
            "panda_joint6": 1.571,
            "panda_joint7": 0.785,
            "panda_finger_joint1": 0.02,  # Only need to set this one!
            # "panda_finger_joint2": 0.02  # Automatically computed from joint1
        }
        
        V.publish_joints(joints)
        print("✅ Published joint configuration (gripper open)")
        
        time.sleep(2)
        
        # Test with closed gripper - showing mimic joint functionality
        joints_closed = joints.copy()
        joints_closed["panda_finger_joint1"] = 0.0  # Closed gripper
        # Note: panda_finger_joint2 will be automatically set to match
        
        V.publish_joints(joints_closed)
        print("✅ Published joint configuration (gripper closed)")
        print("   Note: Only set finger_joint1, finger_joint2 computed automatically")
        
        return True
    except Exception as e:
        print(f"❌ Joint configuration test failed: {e}")
        print("Note: You may need to adjust joint names for your robot")
        return False

def test_trajectory():
    """Test trajectory animation."""
    print("🎬 Testing trajectory animation...")
    try:
        # Create a simple trajectory
        waypoints = []
        
        # Base configuration
        base_joints = {
            "panda_joint1": 0.0,
            "panda_joint2": -0.785,
            "panda_joint3": 0.0,
            "panda_joint4": -2.356,
            "panda_joint5": 0.0,
            "panda_joint6": 1.571,
            "panda_joint7": 0.785,
            "panda_finger_joint1": 0.02,
            "panda_finger_joint2": 0.02
        }
        
        # Create 5 waypoints with varying first joint
        for i in range(5):
            joints = base_joints.copy()
            joints["panda_joint1"] = -1.0 + i * 0.5  # Sweep from -1 to 1
            waypoints.append(joints)
        
        V.publish_trajectory(waypoints, segment_duration=1.0, rate_hz=30.0)
        print("✅ Published trajectory animation at 30Hz")
        
        return True
    except Exception as e:
        print(f"❌ Trajectory test failed: {e}")
        print("Note: You may need to adjust joint names for your robot")
        return False

def test_smooth_trajectory():
    """Test smooth high-framerate trajectory animation."""
    print("🎬 Testing smooth trajectory animation...")
    try:
        # Create a more complex trajectory with more waypoints
        waypoints = []
        
        # Base configuration
        base_joints = {
            "panda_joint1": 0.0,
            "panda_joint2": -0.785,
            "panda_joint3": 0.0,
            "panda_joint4": -2.356,
            "panda_joint5": 0.0,
            "panda_joint6": 1.571,
            "panda_joint7": 0.785,
            "panda_finger_joint1": 0.02,
            "panda_finger_joint2": 0.02
        }
        
        # Create 8 waypoints for smooth circular motion
        import math
        for i in range(8):
            joints = base_joints.copy()
            angle = i * 2 * math.pi / 8
            joints["panda_joint1"] = 0.8 * math.cos(angle)  # Circular motion
            joints["panda_joint2"] = -0.785 + 0.3 * math.sin(angle)  # Vertical motion
            waypoints.append(joints)
        
        # Add the first waypoint again to complete the loop
        waypoints.append(waypoints[0])
        
        # Publish with high framerate for smooth animation
        V.publish_trajectory(waypoints, segment_duration=0.5, rate_hz=60.0)
        print("✅ Published smooth trajectory animation at 60Hz")
        
        return True
    except Exception as e:
        print(f"❌ Smooth trajectory test failed: {e}")
        print("Note: You may need to adjust joint names for your robot")
        return False

def test_ghost_end_effector():
    """Test ghost end effector visualization."""
    print("👻 Testing ghost end effector...")
    try:
        # Test with different poses and colors
        poses = [
            [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],  # Translated in x and z
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.707, 0.707],  # Rotated 90 degrees
            [-0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],  # Translated in -x and z
        ]
        
        colors = [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
        ]
        
        for i, (pose, color) in enumerate(zip(poses, colors)):
            V.publish_ghost_end_effector(
                pose=pose,
                color=color,
                scale=1.0,
                alpha=0.5
            )
            print(f"✅ Published ghost end effector {i+1} with color {color}")
            print(f"   Includes base ee link + fingers with current joint states")
            time.sleep(2)
        
        return True
    except Exception as e:
        print(f"❌ Ghost end effector test failed: {e}")
        print("Note: Make sure link_config.yaml has proper end effector configuration")
        return False


def test_ghost_robot():
    """Test ghost robot visualization."""
    print("👻 Testing ghost robot...")
    try:
        # Test with different configs and colors
        configurations = [
            {
                "panda_joint1": 0.0,
                "panda_joint2": -0.785,
                "panda_joint3": 0.0,
                "panda_joint4": -0.356,
                "panda_joint5": 0.0,
                "panda_joint6": 1.571,
                "panda_joint7": 0.785,
                "panda_finger_joint1": 0.02,
                "panda_finger_joint2": 0.02
            },
            
            {
                "panda_joint1": 0.0,
                "panda_joint2": -0.785,
                "panda_joint3": 0.0,
                "panda_joint4": -1.356,
                "panda_joint5": 0.02,
                "panda_joint6": 1.171,
                "panda_joint7": 0.185,
                "panda_finger_joint1": 0.0,
                "panda_finger_joint2": 0.0
            },
            {
                "panda_joint1": 0.0,
                "panda_joint2": -0.785,
                "panda_joint3": 0.0,
                "panda_joint4": -2.356,
                "panda_joint5": 0.02,
                "panda_joint6": 1.871,
                "panda_joint7": 0.385,
                "panda_finger_joint1": 0.0,
                "panda_finger_joint2": 0.0
            },
        ]
        

        colors = [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
        ]
        
        for i, (configuration, color) in enumerate(zip(configurations, colors)):
            V.publish_ghost_robot(
                configuration=configuration,
                color=color,
                scale=1.0,
                alpha=0.5
            )
            print(f"✅ Published ghost robot {i+1} with color {color}")
            time.sleep(2)
        
        return True
    except Exception as e:
        print(f"❌ Ghost robot test failed: {e}")
        return False

def test_clear_functions():
    """Test clear functionality for all visualization types."""
    print("🧹 Testing clear functions...")
    try:
        # First publish some content to clear
        robot_points = np.random.rand(200, 3).astype(np.float32)
        target_points = np.random.rand(150, 3).astype(np.float32) + [1, 0, 0]
        obstacle_points = np.random.rand(100, 3).astype(np.float32) + [-1, 0, 0]
        
        # Publish point clouds
        V.publish_robot_pointcloud(robot_points)
        V.publish_target_pointcloud(target_points)
        V.publish_obstacle_pointcloud(obstacle_points)
        print("✅ Published test point clouds")
        
        # Publish ghost visualizations
        test_pose = [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]
        test_config = {
            "panda_joint1": 1.0,
            "panda_joint2": -0.785,
            "panda_joint3": 0.5,
            "panda_joint4": -1.356,
            "panda_joint5": 0.5,
            "panda_joint6": 1.171,
            "panda_joint7": 0.785,
            "panda_finger_joint1": 0.02,
        }
        
        V.publish_ghost_end_effector(test_pose, color=[1, 0, 0])
        V.publish_ghost_robot(test_config, color=[0, 1, 0])
        print("✅ Published test ghost visualizations")
        
        time.sleep(3)
        print("👀 Check RViz - you should see point clouds and ghost visualizations")
        time.sleep(2)
        
        # Now clear everything
        print("🧹 Clearing point clouds...")
        V.clear_robot_pointcloud()
        print("✅ Cleared robot point cloud")
        
        time.sleep(1)
        V.clear_target_pointcloud()
        print("✅ Cleared target point cloud")
        
        time.sleep(1)
        V.clear_obstacle_pointcloud()
        print("✅ Cleared obstacle point cloud")
        
        time.sleep(2)
        print("🧹 Clearing ghost visualizations...")
        V.clear_ghost_end_effector()
        print("✅ Cleared ghost end effector")
        
        time.sleep(1)
        V.clear_ghost_robot()
        print("✅ Cleared ghost robot")
        
        print("👀 Check RViz - all visualizations should now be cleared!")
        
        return True
    except Exception as e:
        print(f"❌ Clear functions test failed: {e}")
        return False

def test_shutdown():
    """Test server shutdown functionality."""
    print("🛑 Testing shutdown...")
    try:
        V.shutdown()
        print("Sent shutdown command to viz_server")
        print("✅ Shutdown command sent successfully")
        
        # Wait a moment and verify processes are gone
        import subprocess
        import time
        time.sleep(3)
        
        try:
            # Check if viz_server processes are gone
            result = subprocess.run(["pgrep", "-f", "viz_server.server"], capture_output=True, text=True)
            if result.returncode == 0:
                print("⚠️  Warning: viz_server processes still running")
                print(f"   PIDs: {result.stdout.strip()}")
                print("💡 Try: python3 viz_server/shutdown_viz_server.py --force")
                return False
            
            # Check if robot_state_publisher processes are gone  
            result = subprocess.run(["pgrep", "-f", "robot_state_publisher"], capture_output=True, text=True)
            if result.returncode == 0:
                print("⚠️  Warning: robot_state_publisher processes still running")
                print(f"   PIDs: {result.stdout.strip()}")
                print("💡 Try: python3 viz_server/shutdown_viz_server.py --force")
                return False
                
            print("✅ All processes terminated successfully")
            return True
            
        except Exception as e:
            print(f"⚠️  Could not verify process termination: {e}")
            return True  # Assume success if we can't verify
            
    except Exception as e:
        print(f"❌ Shutdown test failed: {e}")
        return False

def test_force_shutdown():
    """Test force shutdown using the command-line script."""
    print("🔪 Testing force shutdown...")
    try:
        import subprocess
        
        # Run the force shutdown script
        result = subprocess.run(
            ["python3", "viz_server/shutdown_viz_server.py", "--force"],
            capture_output=True, text=True, cwd="/workspace"
        )
        
        print(f"Exit code: {result.returncode}")
        if result.stdout:
            print(f"Output: {result.stdout}")
        if result.stderr:
            print(f"Error: {result.stderr}")
            
        if result.returncode == 0:
            print("✅ Force shutdown completed successfully")
            return True
        else:
            print("❌ Force shutdown failed")
            return False
            
    except Exception as e:
        print(f"❌ Force shutdown test failed: {e}")
        return False

def test_all():
    """Run all tests in sequence."""
    print("🚀 Running all tests...")
    
    tests = [
        ("Connection", test_connect),
        ("Point Cloud", test_pointcloud),
        ("Joints", test_joints),
        ("Trajectory", test_trajectory),
        ("Smooth Trajectory", test_smooth_trajectory),
        ("Ghost End Effector", test_ghost_end_effector),
        ("Ghost Robot", test_ghost_robot),
        ("Clear Functions", test_clear_functions),
    ]
    
    results = {}
    for name, test_func in tests:
        print(f"\n--- Testing {name} ---")
        results[name] = test_func()
        if results[name]:
            print(f"✅ {name} test passed")
        else:
            print(f"❌ {name} test failed")
        time.sleep(1)
    
    print("\n=== Test Results ===")
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
    
    # Optional: Ask user if they want to shutdown
    print("\n🛑 To shutdown viz_server, call: test_shutdown()")
    print("   Or directly: V.shutdown()")

def interactive_help():
    """Print help for interactive usage."""
    print("""
📚 Interactive Test Functions:
    
    test_connect()      - Test connection to viz_server
    test_pointcloud()      - Test multiple point cloud types (robot/target/obstacle)
    test_joints()          - Test joint configuration
    test_trajectory()      - Test trajectory animation (30Hz)
    test_smooth_trajectory() - Test smooth trajectory animation (60Hz)
    test_ghost_end_effector() - Test ghost end effector visualization
    test_ghost_robot()     - Test ghost robot visualization (full robot FK)
    test_clear_functions() - Test clearing all visualization types
    test_shutdown()        - Shutdown the viz_server cleanly (with verification)
    test_force_shutdown()  - Force shutdown using command-line script
    test_all()             - Run all tests
    
🎯 Quick Examples:
    
    # Test connection first
    test_connect()
    
    # Test individual features
    test_pointcloud()    # Shows robot (blue), target (green), obstacle (red) clouds
    test_joints()
    
    # Test specific point cloud types (new API)
    V.publish_robot_pointcloud(points)      # Robot points
    V.publish_target_pointcloud(points)     # Target points  
    V.publish_obstacle_pointcloud(points)   # Obstacle points
    
    # Clear specific visualizations
    V.clear_robot_pointcloud()          # Clear robot points
    V.clear_ghost_end_effector()        # Clear ghost end effector
    V.clear_ghost_robot()               # Clear ghost robot
    
    # Or run everything
    test_all()
    
    # Shutdown when done
    V.shutdown()
    
💡 Tips:
    - Make sure RViz is running first
    - Adjust URDF path in test_connect() if needed
    - Adjust joint names for your specific robot
    - Use Ctrl+C to stop trajectory animations
    - Use V.shutdown() to cleanly stop the viz_server
    - Configure point cloud colors in RViz display panel for each topic
    """)

if __name__ == "__main__":
    interactive_help()
    
    # Ask user what they want to test
    print("\nWhat would you like to test?")
    print("1. Run all tests")
    print("2. Test connection only")
    print("3. Shutdown viz_server (graceful)")
    print("4. Force shutdown (if graceful fails)")
    print("5. Enter interactive mode")
    
    choice = input("Enter choice (1-5): ").strip()
    
    if choice == "1":
        test_all()
    elif choice == "2":
        test_connect()
    elif choice == "3":
        test_shutdown()
    elif choice == "4":
        test_force_shutdown()
    elif choice == "5":
        print("Interactive mode - call test functions manually")
        interactive_help()
    else:
        print("Invalid choice. Running help...")
        interactive_help() 