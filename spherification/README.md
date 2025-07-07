# Spherification

This folder contains utilities and scripts for visualizing and manipulating collision spheres in Blender, as well as converting between collision sphere representation formats.

## Standard JSON format for collision spheres

```json
{
    "first_link_name": [
        {
            "origin": [
                <x>,
                <y>,
                <z>
            ],
            "radius": <radius>
        },
        [more spheres ...]
    ],
    "second_link_name": [
        {
            "origin": [
                <x>,
                <y>,
                <z>
            ],
            "radius": <radius>
        },
        [more spheres ...]
    ]
}
```

All units in meters. These `.json` files are placed together with the robot's urdf,
from which the link names are taken, stored in `assets/<robot_name>`.

## Contents

- `spherification_utils.py`: Python utilities for:
  - Adding spheres to a Blender scene from JSON files.
  - Printing information about spheres in the scene.
  - Converting between different sphere data formats (e.g., original robofin Franka _SPHERES, _SELF_COLLISION_SPHERES) and a JSON format organized by robot link.

# Blender Sphere Editing Workflow

This guide explains how to use the sphere editing tools in Blender for robot collision sphere generation and refinement.

## Prerequisites

1. Blender 4.2 LTS installed
2. Robot URDF file with mesh references
3. Optional: collision_spheres.json and self_collision_spheres.json files

## Quick Start

### 1. Open Blender and Import Module

In Blender's Python console (Window > Toggle System Console):

```python
import sys
sys.path.append('/workspace/spherification')
import spherification_utils
```

### 2. Analyze Your Robot

```python
# Analyze robot structure and existing spheres
spherification_utils.analyze_robot_spheres('assets/panda/panda.urdf')
```

This will show a properly formatted table like:
```
Link Name          Visual Mesh  Collision  Self-Collision  
-----------------------------------------------------------
panda_link0        Yes          1          2               
panda_link1        Yes          4          4               
...
```

### 3. Generate Initial Spheres (if needed)

If you don't have sphere files yet:

```python
# Generate spheres using foam for all meshes
spherification_utils.generate_initial_spheres('path/to/your/robot.urdf')
```

### 4. Load Link for Editing

Load a specific link with its mesh and spheres:

```python
# For collision spheres (with automatic coordinate system fix)
spherification_utils.load_link_for_editing('assets/panda/panda.urdf', 'panda_link1', 'collision')

# For self-collision spheres  
spherification_utils.load_link_for_editing('assets/panda/panda.urdf', 'panda_link1', 'self_collision')

# Disable coordinate fix if needed
spherification_utils.load_link_for_editing('assets/panda/panda.urdf', 'panda_link1', 'collision', apply_coordinate_fix=False)
```

This will:
- Clear the scene
- Load the link's mesh (locked in place, coordinate-corrected)
- Add all spheres for that link (green, editable)

### 5. Edit Spheres in Blender

Use standard Blender controls:
- **Select**: Left-click on spheres
- **Move**: Press `G`, then `X`/`Y`/`Z` for axis-constrained movement
- **Scale**: Press `S` to scale sphere size
- **Delete**: Press `X` > Delete to remove spheres
- **Add**: Use the `add_sphere()` function or duplicate existing ones

### 6. Unlock Mesh Objects (if needed)

To unlock/unfreeze the mesh in Blender GUI:

**Method 1: Outliner (recommended)**
1. Open **Outliner** (top-right panel with scene collection)
2. Find your mesh object (named like `panda_link1_mesh`)
3. Click the **lock icons** next to the object name to toggle them off:
   - 🔒➡️🔓 **Location lock** (position)
   - 🔒➡️🔓 **Rotation lock** (orientation)
   - 🔒➡️🔓 **Scale lock** (size)

**Method 2: Properties Panel**
1. Select the mesh object
2. Press **N** to open properties panel
3. Scroll to **Transform** section
4. Uncheck the lock boxes

### 7. Save Your Changes

```python
# Save collision spheres
spherification_utils.save_edited_spheres('assets/panda/panda.urdf', 'panda_link1', 'collision')

# Save self-collision spheres
spherification_utils.save_edited_spheres('assets/panda/panda.urdf', 'panda_link1', 'self_collision')
```

## Coordinate System Issues & Fixes

### Common Problem: 90° Rotation Mismatch

**Issue**: .obj files often use Y-up coordinate system while URDF/robotics uses Z-up, causing spheres and meshes to appear misaligned.

**Solution**: Automatic coordinate system fix is now enabled by default.

### Testing Coordinate Alignment

```python
# Test with coordinate fix (default)
spherification_utils.load_link_for_editing('assets/panda/panda.urdf', 'panda_link1', 'collision', apply_coordinate_fix=True)

# Test without coordinate fix (for comparison)
spherification_utils.load_link_for_editing('assets/panda/panda.urdf', 'panda_link1', 'self_collision', apply_coordinate_fix=False)
```

### Interactive Alignment Testing

For finding the correct transformation for your specific robot:

```python
# Interactive testing of different coordinate transformations
spherification_utils.test_mesh_alignment('assets/panda/panda.urdf', 'panda_link1')
```

### Manual Coordinate Fixes

```python
# Get mesh object
mesh_obj = spherification_utils.get_mesh_object('panda_link1')

# Apply custom transformation
spherification_utils.apply_custom_mesh_transform(mesh_obj, 'X', -90)  # Y-up to Z-up
spherification_utils.apply_custom_mesh_transform(mesh_obj, 'Z', 180)  # 180° flip
```

## Advanced Usage

### Testing Script

Run this in Blender to test all functionality:

```python
exec(open('/workspace/spherification/blender_test_script.py').read())
```

### Available Functions

- `analyze_robot_spheres(urdf_path)` - Show robot analysis table
- `generate_initial_spheres(urdf_path)` - Generate spheres using foam
- `load_link_for_editing(urdf_path, link_name, sphere_type, apply_coordinate_fix=True)` - Load link for editing
- `save_edited_spheres(urdf_path, link_name, sphere_type)` - Save sphere changes
- `add_sphere(x, y, z, radius, name_prefix, index)` - Add new sphere manually
- `clear_scene()` - Clear all objects from scene
- `test_mesh_alignment(urdf_path, link_name)` - Interactive coordinate system testing
- `apply_custom_mesh_transform(mesh_obj, axis, degrees)` - Manual coordinate fix
- `get_mesh_object(link_name)` - Get mesh object reference

### Sphere Types

- `"collision"` - Regular collision detection spheres
- `"self_collision"` - Self-collision detection spheres (typically larger, fewer)

## File Structure

After editing, your robot directory should contain:

```
assets/your_robot/
├── robot.urdf
├── collision_spheres.json      # Regular collision spheres
├── self_collision_spheres.json # Self-collision spheres
└── meshes/
    ├── visual/
    └── collision/
```

## Tips

1. **Start with visual meshes** - They're usually higher quality than collision meshes
2. **Coordinate system fixes** - Automatic for .obj files, disable if not needed
3. **Lock mesh objects** - The system automatically locks loaded meshes to prevent accidental movement
4. **Use sphere hierarchy** - Large spheres for coarse coverage, small ones for details
5. **Self-collision spheres** - Usually need to be larger to prevent false positives
6. **Save frequently** - Use the save function after each link editing session
7. **Test alignment** - Use the interactive testing function for new robots

## Troubleshooting

### Mesh Loading Issues
- **Mesh not loading**: Check URDF paths and file formats (.obj, .stl, .dae supported)
- **Import operator errors**: Script automatically tries multiple import methods
- **Missing addons**: Script automatically enables required import addons

### Coordinate System Issues
- **90° rotation mismatch**: Enable `apply_coordinate_fix=True` (default)
- **Wrong transformation**: Use `test_mesh_alignment()` to find correct fix
- **Custom robots**: May need custom transformation parameters

### General Issues
- **No spheres visible**: Check that JSON files exist and contain data for the link
- **Import errors**: Ensure Blender has access to the module path
- **Foam generation fails**: Check that foam is installed at `/opt/foam/`
- **Performance issues**: Work on one link at a time for large robots 