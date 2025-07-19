#!/usr/bin/env python3
"""
Simple test to compare end effector sampling between original and generic samplers.
Start with just verifying the forward kinematics match.
"""

import numpy as np
from robofin.samplers_original import NumpyFrankaSampler
from robofin.samplers import NumpyRobotSampler
from robofin.robots import Robot
from robofin.kinematics.numba import franka_eef_visual_fk, eef_pose_to_link8
from robofin.robot_constants import FrankaConstants
import urchin


def test_fk_equivalence():
    """Test that our URDF-based FK produces the same results as franka_eef_visual_fk."""
    
    print("=" * 60)
    print("Testing FK Equivalence for End Effector")
    print("=" * 60)
    
    # Create robot instance
    robot = Robot("assets/panda/")
    robot_urdf = urchin.URDF.load(str(robot.urdf_path), lazy_load_meshes=True)
    
    # Test parameters - use NEUTRAL configuration for the 7 revolute joints
    neutral_config = FrankaConstants.NEUTRAL  # 7 revolute joints
    prismatic_joint = 0.02
    test_pose = np.eye(4)  # Identity pose
    frame = "right_gripper"
    
    print(f"Testing with neutral_config={neutral_config}")
    print(f"Testing with prismatic_joint={prismatic_joint}, frame={frame}")
    
    # Original FK computation
    print("\n1. Original FK computation using franka_eef_visual_fk:")
    
    # Convert pose to link8 coordinates (like the original does)
    pose_link8 = eef_pose_to_link8(test_pose, frame)
    print(f"   Pose after eef_pose_to_link8:\n{pose_link8}")
    
    # Get FK from original function
    original_fk = franka_eef_visual_fk(prismatic_joint, pose_link8)
    print(f"   Original FK shapes: {[fk.shape for fk in original_fk]}")
    print(f"   panda_hand transform:\n{original_fk[0]}")
    print(f"   panda_leftfinger transform:\n{original_fk[1]}")
    print(f"   panda_rightfinger transform:\n{original_fk[2]}")
    
    # Generic FK computation
    print("\n2. Generic FK computation using robot_urdf.link_fk:")
    
    # Create configuration: 7 revolute joints (NEUTRAL) + 1 prismatic joint
    # The URDF has 8 actuated joints: 7 revolute + 1 prismatic
    full_config = np.zeros(len(robot_urdf.actuated_joints))
    
    # Set the 7 revolute joints to NEUTRAL
    full_config[:7] = neutral_config
    
    # Set the prismatic joint 
    # Find the prismatic joint and set it
    for i, joint in enumerate(robot_urdf.actuated_joints):
        if joint.joint_type not in ['revolute', 'continuous']:
            full_config[i] = prismatic_joint
            print(f"   Set prismatic joint {joint.name} at index {i} to {prismatic_joint}")
    
    print(f"   Full config: {full_config}")
    
    # Get FK from URDF
    urdf_fk = robot_urdf.link_fk(full_config, use_names=True)
    
    # Extract the same links as the original
    eef_links = ["panda_hand", "panda_leftfinger", "panda_rightfinger"]
    
    for i, link_name in enumerate(eef_links):
        if link_name in urdf_fk:
            print(f"   {link_name} transform from URDF:\n{urdf_fk[link_name]}")
            
            # Apply visual transform for rightfinger
            if link_name == "panda_rightfinger":
                visual_transform = robot.get_visual_transform(link_name)
                print(f"   {link_name} visual transform:\n{visual_transform}")
                combined = urdf_fk[link_name] @ visual_transform
                print(f"   {link_name} with visual transform:\n{combined}")
        else:
            print(f"   {link_name} NOT FOUND in URDF FK")
    
    # Compare results
    print("\n3. Comparison:")
    
    for i, link_name in enumerate(eef_links):
        if link_name in urdf_fk:
            urdf_transform = urdf_fk[link_name]
            
            # Apply visual transform for rightfinger
            if link_name == "panda_rightfinger":
                visual_transform = robot.get_visual_transform(link_name)
                urdf_transform = urdf_transform @ visual_transform
            
            # Apply the pose transformation (pose_link8)
            final_urdf_transform = pose_link8 @ urdf_transform
            
            original_transform = original_fk[i]
            
            diff = np.max(np.abs(final_urdf_transform - original_transform))
            print(f"   {link_name}: Max difference = {diff:.10f}")
            
            if diff > 1e-6:
                print(f"   MISMATCH for {link_name}!")
                print(f"   Original:\n{original_transform}")
                print(f"   URDF:\n{final_urdf_transform}")
            else:
                print(f"   ✓ {link_name} matches")


def test_end_effector_sampling():
    """Test end effector sampling comparison."""
    
    print("\n" + "=" * 60)
    print("Testing End Effector Sampling")
    print("=" * 60)
    
    # Initialize samplers
    print("Initializing samplers...")
    original_sampler = NumpyFrankaSampler(
        num_robot_points=1000, num_eef_points=100, use_cache=True
    )
    
    robot = Robot("assets/panda/")
    generic_sampler = NumpyRobotSampler(
        robot=robot,
        num_robot_points=1000, num_eef_points=100, use_cache=True
    )
    
    # Test parameters
    test_pose = np.eye(4)
    auxiliary_joint_value = 0.02
    frame = "right_gripper"
    num_points = 50
    
    print(f"Test parameters:")
    print(f"  pose: identity")
    print(f"  auxiliary_joint_value: {auxiliary_joint_value}")
    print(f"  frame: {frame}")
    print(f"  num_points: {num_points}")
    
    # Sample from both
    np.random.seed(42)
    original_points = original_sampler.sample_end_effector(
        test_pose, auxiliary_joint_value, num_points=num_points, frame=frame
    )
    
    np.random.seed(42)  # Same seed
    generic_points = generic_sampler.sample_end_effector(
        test_pose, auxiliary_joint_value, num_points=num_points, frame=frame
    )
    
    # Compare results
    print(f"\nResults:")
    print(f"  Original shape: {original_points.shape}")
    print(f"  Generic shape: {generic_points.shape}")
    
    if original_points.shape == generic_points.shape:
        max_diff = np.max(np.abs(original_points - generic_points))
        mean_diff = np.mean(np.abs(original_points - generic_points))
        
        print(f"  Max difference: {max_diff:.10f}")
        print(f"  Mean difference: {mean_diff:.10f}")
        
        if max_diff < 1e-10:
            print("  ✓ EXACT MATCH!")
        elif max_diff < 1e-6:
            print("  ✓ Very close match")
        else:
            print("  ✗ Significant difference")
            
            print("\nFirst 5 points comparison:")
            for i in range(min(5, len(original_points))):
                print(f"  {i}: orig={original_points[i]} gen={generic_points[i]}")
    else:
        print("  ✗ Shape mismatch!")


if __name__ == "__main__":
    test_fk_equivalence()
    test_end_effector_sampling() 