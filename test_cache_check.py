#!/usr/bin/env python3
"""
Check if both samplers are using the same cached point clouds.
"""

import numpy as np
from robofin.samplers_original import NumpyFrankaSampler
from robofin.samplers import NumpyRobotSampler
from robofin.robots import Robot


def test_cache_equivalence():
    """Test if both samplers load the same cached point clouds."""
    
    print("=" * 60)
    print("Testing Cache Equivalence")
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
    
    print("Comparing cached point clouds...")
    
    # Check end effector points
    eef_links = ["panda_hand", "panda_leftfinger", "panda_rightfinger"]
    
    for link_name in eef_links:
        orig_key = f"eef_{link_name}"
        
        if orig_key in original_sampler.points and orig_key in generic_sampler.points:
            orig_points = original_sampler.points[orig_key]
            gen_points = generic_sampler.points[orig_key]
            
            print(f"  {link_name}:")
            print(f"    Original shape: {orig_points.shape}")
            print(f"    Generic shape: {gen_points.shape}")
            
            if orig_points.shape == gen_points.shape:
                max_diff = np.max(np.abs(orig_points - gen_points))
                print(f"    Max difference: {max_diff:.15f}")
                
                if max_diff < 1e-15:
                    print(f"    ✓ EXACT MATCH")
                else:
                    print(f"    ✗ Different point clouds")
                    print(f"    First 3 orig: {orig_points[:3]}")
                    print(f"    First 3 gen:  {gen_points[:3]}")
            else:
                print(f"    ✗ Shape mismatch")
        else:
            print(f"  {link_name}: Missing in one of the samplers")
    
    # Test the actual FK functions directly
    print("\nTesting FK functions directly:")
    
    pose = np.eye(4)
    prismatic_joint = 0.02
    frame = "right_gripper"
    
    # Direct comparison with same inputs
    from robofin.kinematics.numba import get_points_on_franka_eef
    from robofin.samplers import get_points_on_robot_eef
    
    # Same parameters for both
    np.random.seed(42)
    orig_direct = get_points_on_franka_eef(
        pose, prismatic_joint, 20,
        original_sampler.points["eef_panda_hand"],
        original_sampler.points["eef_panda_leftfinger"], 
        original_sampler.points["eef_panda_rightfinger"],
        frame
    )
    
    np.random.seed(42)
    gen_direct = get_points_on_robot_eef(
        None, pose, prismatic_joint, 20,
        generic_sampler.points,
        ["panda_hand", "panda_leftfinger", "panda_rightfinger"],
        "right_gripper", frame, robot
    )
    
    print(f"  Direct function comparison:")
    print(f"    Original shape: {orig_direct.shape}")
    print(f"    Generic shape: {gen_direct.shape}")
    
    if orig_direct.shape == gen_direct.shape:
        max_diff = np.max(np.abs(orig_direct - gen_direct))
        print(f"    Max difference: {max_diff:.15f}")
        
        if max_diff < 1e-15:
            print(f"    ✓ EXACT MATCH")
        else:
            print(f"    ✗ Different results")
            print(f"    First 3 orig: {orig_direct[:3]}")
            print(f"    First 3 gen:  {gen_direct[:3]}")


if __name__ == "__main__":
    test_cache_equivalence() 