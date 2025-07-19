#!/usr/bin/env python3
"""
Direct test of FK functions without random sampling.
"""

import numpy as np
from robofin.samplers_original import NumpyFrankaSampler
from robofin.samplers import NumpyRobotSampler
from robofin.robots import Robot
from robofin.kinematics.numba import get_points_on_franka_eef, eef_pose_to_link8, franka_eef_visual_fk
from robofin.samplers import get_points_on_robot_eef
from geometrout.maths import transform_in_place


def test_fk_direct():
    """Test FK functions directly without sampling."""
    
    print("=" * 60)
    print("Testing FK Functions Directly")
    print("=" * 60)
    
    # Initialize samplers to get point clouds
    original_sampler = NumpyFrankaSampler(
        num_robot_points=1000, num_eef_points=100, use_cache=True
    )
    
    robot = Robot("assets/panda/")
    generic_sampler = NumpyRobotSampler(
        robot=robot,
        num_robot_points=1000, num_eef_points=100, use_cache=True
    )
    
    # Test parameters
    pose = np.eye(4)
    prismatic_joint = 0.02
    frame = "right_gripper"
    
    print(f"Testing with pose=identity, prismatic_joint={prismatic_joint}, frame={frame}")
    
    # Test 1: Direct FK comparison
    print("\n1. FK Transform comparison:")
    
    base_pose = eef_pose_to_link8(pose, frame)
    fk_transforms = franka_eef_visual_fk(prismatic_joint, base_pose)
    
    print(f"   base_pose from eef_pose_to_link8:\n{base_pose}")
    print(f"   fk_transforms[0] (hand):\n{fk_transforms[0]}")
    print(f"   fk_transforms[1] (leftfinger):\n{fk_transforms[1]}")
    print(f"   fk_transforms[2] (rightfinger):\n{fk_transforms[2]}")
    
    # Test 2: Transform specific point clouds
    print("\n2. Transform specific point clouds:")
    
    # Get the exact point clouds
    hand_points = original_sampler.points["eef_panda_hand"]
    leftfinger_points = original_sampler.points["eef_panda_leftfinger"]
    rightfinger_points = original_sampler.points["eef_panda_rightfinger"]
    
    print(f"   hand_points shape: {hand_points.shape}")
    print(f"   leftfinger_points shape: {leftfinger_points.shape}")
    print(f"   rightfinger_points shape: {rightfinger_points.shape}")
    
    # Transform them individually
    hand_transformed = transform_in_place(np.copy(hand_points), fk_transforms[0])
    leftfinger_transformed = transform_in_place(np.copy(leftfinger_points), fk_transforms[1])
    rightfinger_transformed = transform_in_place(np.copy(rightfinger_points), fk_transforms[2])
    
    print(f"   hand_transformed first 3: {hand_transformed[:3]}")
    print(f"   leftfinger_transformed first 3: {leftfinger_transformed[:3]}")
    print(f"   rightfinger_transformed first 3: {rightfinger_transformed[:3]}")
    
    # Test 3: Full original function without sampling
    print("\n3. Full original function (no sampling):")
    
    original_all = get_points_on_franka_eef(
        pose, prismatic_joint, 0,  # sample=0 means no sampling
        hand_points, leftfinger_points, rightfinger_points, frame
    )
    
    print(f"   Original all points shape: {original_all.shape}")
    print(f"   Original first 3: {original_all[:3]}")
    print(f"   Original last 3: {original_all[-3:]}")
    
    # Test 4: My function without sampling
    print("\n4. My function (no sampling):")
    
    generic_all = get_points_on_robot_eef(
        None, pose, prismatic_joint, 0,  # sample=0 means no sampling
        generic_sampler.points,
        ["panda_hand", "panda_leftfinger", "panda_rightfinger"],
        "right_gripper", frame, robot
    )
    
    print(f"   Generic all points shape: {generic_all.shape}")
    print(f"   Generic first 3: {generic_all[:3]}")
    print(f"   Generic last 3: {generic_all[-3:]}")
    
    # Test 5: Direct comparison
    print("\n5. Direct comparison:")
    
    if original_all.shape == generic_all.shape:
        max_diff = np.max(np.abs(original_all - generic_all))
        print(f"   Max difference: {max_diff:.15f}")
        
        if max_diff < 1e-15:
            print("   ✓ EXACT MATCH!")
        else:
            print("   ✗ Different results")
            
            # Find where they differ
            diff_mask = np.abs(original_all - generic_all) > 1e-10
            diff_indices = np.where(diff_mask.any(axis=1))[0]
            print(f"   First 5 differing indices: {diff_indices[:5]}")
            
            for i in diff_indices[:5]:
                print(f"   Index {i}: orig={original_all[i]} gen={generic_all[i]}")
    else:
        print("   ✗ Shape mismatch!")


if __name__ == "__main__":
    test_fk_direct() 