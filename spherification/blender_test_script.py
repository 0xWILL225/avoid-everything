#!/usr/bin/env python3
"""
Updated test script for Blender sphere editing workflow.
Tests coordinate system fixes for .obj files.
Run this script inside Blender's Python console.
"""

import sys
sys.path.append('/workspace/spherification')
import spherification_utils

def test_blender_functions():
    """Test all Blender-specific functions with coordinate system fixes."""
    
    print("🔧 TESTING BLENDER FUNCTIONS (WITH COORDINATE FIX)")
    print("=" * 50)
    
    # Test 1: Analyze robot (now with fixed table formatting)
    print("\n1. Testing robot analysis (fixed table)...")
    result = spherification_utils.analyze_robot_spheres('assets/panda/panda.urdf')
    
    test_link: str = 'panda_link1'
    
    # Test 2: Load a link with collision spheres AND coordinate fix
    print(f"\n2. Testing collision sphere loading with coordinate fix...")
    print(f"   Loading {test_link} with automatic .obj coordinate system fix")
    success = spherification_utils.load_link_for_editing(
        'assets/panda/panda.urdf', 
        test_link, 
        'collision',
        apply_coordinate_fix=True  # This should fix the 90° rotation issue
    )
    
    if success:
        print("✅ Collision spheres loaded successfully!")
        print("\n🎯 CHECK ALIGNMENT:")
        print("   - Look at the 3D viewport")
        print("   - Green spheres should now align with the gray mesh")
        print("   - If mesh appears, it should match sphere positions")
        
        # Test 3: Save the spheres (without any changes)
        print("\n3. Testing sphere saving...")
        spherification_utils.save_edited_spheres(
            'assets/panda/panda.urdf', 
            test_link, 
            'collision'
        )
        print("✅ Spheres saved successfully!")
    
    print("\n" + "=" * 50)
    print("✅ ALL TESTS COMPLETED")
    print("\n🎯 COORDINATE SYSTEM FIX:")
    print("- .obj files: Automatic Y-up to Z-up conversion applied")
    print("- Should fix the 90° rotation mismatch you observed")
    print("- Set apply_coordinate_fix=False to disable if needed")
    print("\n🎮 MANUAL TESTING:")
    print("- Select spheres and press 'G' to move them")
    print("- Press 'S' to scale them")
    print("- Unlock mesh: Outliner > click lock icons next to mesh object")
    print("- If mesh loaded: spheres should align properly now!")

def test_coordinate_fixes():
    """Test different coordinate system transformations."""
    print("\n🔄 TESTING COORDINATE TRANSFORMATIONS")
    print("=" * 40)
    
    test_link = 'panda_link1'
    
    # Test different coordinate fix options
    tests = [
        (True, "Y-up to Z-up fix (default)"),
        (False, "No coordinate fix (original)")
    ]
    
    for i, (apply_fix, description) in enumerate(tests):
        print(f"\nTest {i+1}: {description}")
        spherification_utils.load_link_for_editing(
            'assets/panda/panda.urdf',
            test_link,
            'collision' if apply_fix else 'self_collision',
            apply_coordinate_fix=apply_fix
        )
        input(f"Press Enter to continue to next test...")

# Automatically run test when script is executed
if __name__ == "__main__":
    test_blender_functions()

# Additional functions for manual testing:
# test_coordinate_fixes()  # Uncomment to test coordinate transformations step by step

# For manual execution in Blender console:
# exec(open('/workspace/spherification/blender_test_script.py').read()) 