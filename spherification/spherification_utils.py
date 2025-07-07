import bpy
import json
import xml.etree.ElementTree as ET
import os
from pathlib import Path
import subprocess



def print_all_sphere_info(prefix="Sphere_"):
  """Prints the location and radius of all mesh objects with the given name prefix."""
  for obj in bpy.data.objects:
    if obj.type == 'MESH' and obj.name.startswith(prefix):
      loc = obj.location
      radius = obj.dimensions.x / 2 # assumes uniform scaling
      print(f"Sphere '{obj.name}': Location = ({loc.x:.3f}, {loc.y:.3f}, {loc.z:.3f}), Radius = {radius:.3f} m")


def print_sphere_info(obj):
  if obj.type != 'MESH':
    print(f"{obj.name} is not a mesh object.")
    return
  loc = obj.location
  radius = obj.dimensions.x / 2 # assumes uniform scale and sphere
  print(f"Sphere '{obj.name}': Location = ({loc.x:.3f}, {loc.y:.3f}, {loc.z:.3f}), Radius = {radius:.3f} m")


def add_sphere(x, y, z, radius, name_prefix="Sphere_", index=None):
  """Adds a green, opaque sphere to the scene."""
  bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=(x, y, z))
  obj = bpy.context.active_object
  obj.name = f"{name_prefix}{index}" if index is not None else name_prefix

  # Add green material
  mat = bpy.data.materials.new(name="GreenMaterial")
  mat.use_nodes = True
  bsdf = mat.node_tree.nodes.get("Principled BSDF")
  bsdf.inputs["Base Color"].default_value = (0.0, 1.0, 0.0, 1.0)
  bsdf.inputs["Alpha"].default_value = 1.0
  mat.blend_method = 'OPAQUE'
  obj.data.materials.append(mat)

  return obj


def parse_urdf(urdf_path):
  """Parse URDF file and extract link information with mesh paths."""
  tree = ET.parse(urdf_path)
  root = tree.getroot()
  
  links = {}
  urdf_dir = Path(urdf_path).parent
  
  for link in root.findall('link'):
    link_name = link.get('name')
    visual_mesh = None
    collision_mesh = None
    
    # Find visual mesh
    visual = link.find('visual')
    if visual is not None:
      geometry = visual.find('geometry')
      if geometry is not None:
        mesh = geometry.find('mesh')
        if mesh is not None:
          visual_mesh = mesh.get('filename')
    
    # Find collision mesh
    collision = link.find('collision')
    if collision is not None:
      geometry = collision.find('geometry')
      if geometry is not None:
        mesh = geometry.find('mesh')
        if mesh is not None:
          collision_mesh = mesh.get('filename')
    
    links[link_name] = {
      'visual_mesh': visual_mesh,
      'collision_mesh': collision_mesh
    }
  
  return links


def analyze_robot_spheres(urdf_path):
  """
  Step 3: Analyze robot URDF and show table of links with sphere counts.
  """
  print(f"\n=== ROBOT ANALYSIS: {urdf_path} ===")
  
  # Parse URDF
  try:
    links = parse_urdf(urdf_path)
    print(f"SUCCESS: URDF parsed successfully - Found {len(links)} links")
  except Exception as e:
    print(f"ERROR: Error parsing URDF: {e}")
    return None
  
  # Check for sphere files in collision_spheres subdirectory
  urdf_dir = Path(urdf_path).parent
  collision_spheres_dir = urdf_dir / "collision_spheres"
  collision_spheres_path = collision_spheres_dir / "collision_spheres.json"
  self_collision_spheres_path = collision_spheres_dir / "self_collision_spheres.json"
  
  collision_spheres = {}
  self_collision_spheres = {}
  
  if collision_spheres_path.exists():
    with open(collision_spheres_path, 'r') as f:
      collision_spheres = json.load(f)
    print(f"SUCCESS: Found collision_spheres.json")
  else:
    print(f"WARNING: No collision_spheres.json found in {collision_spheres_dir}")
  
  if self_collision_spheres_path.exists():
    with open(self_collision_spheres_path, 'r') as f:
      self_collision_spheres = json.load(f)
    print(f"SUCCESS: Found self_collision_spheres.json")
  else:
    print(f"WARNING: No self_collision_spheres.json found in {collision_spheres_dir}")
  
  # Calculate column widths for consistent formatting
  rows = []
  for link_name, link_info in links.items():
    collision_count = len(collision_spheres.get(link_name, []))
    self_collision_count = len(self_collision_spheres.get(link_name, []))
    has_visual = "Yes" if link_info['visual_mesh'] else "No"
    rows.append([link_name, has_visual, str(collision_count), str(self_collision_count)])
  
  # Calculate maximum width for each column
  headers = ["Link Name", "Visual Mesh", "Collision", "Self-Collision"]
  all_rows = [headers] + rows
  col_widths = []
  for i in range(len(headers)):
    max_width = max(len(row[i]) for row in all_rows)
    col_widths.append(max_width + 2) # Add 2 for padding
  
  # Print formatted table
  header_format = "".join(f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers)))
  print(f"\n{header_format}")
  print("-" * sum(col_widths))
  
  for row in rows:
    row_format = "".join(f"{row[i]:<{col_widths[i]}}" for i in range(len(row)))
    print(row_format)
  
  return {
    'links': links,
    'collision_spheres': collision_spheres,
    'self_collision_spheres': self_collision_spheres,
    'urdf_dir': urdf_dir
  }


def generate_initial_spheres(urdf_path):
  """
  Step 4: Generate initial spheres using foam for all visual meshes in the URDF.
  """
  print(f"\n=== GENERATING INITIAL SPHERES ===")
  
  # Parse URDF to get mesh files
  try:
    links = parse_urdf(urdf_path)
  except Exception as e:
    print(f"ERROR: Error parsing URDF: {e}")
    return False
  
  urdf_dir = Path(urdf_path).parent
  collision_spheres = {}
  self_collision_spheres = {}
  
  # Process each link with a visual mesh
  for link_name, link_info in links.items():
    if link_info['visual_mesh']:
      mesh_path = urdf_dir / link_info['visual_mesh']
      
      if mesh_path.exists():
        print(f"Generating spheres for {link_name}: {mesh_path}")
        
        try:
          # Run foam sphere generation
          result = subprocess.run([
            'python3', '/opt/foam/scripts/generate_spheres.py',
            str(mesh_path)
          ], capture_output=True, text=True, timeout=30)
          
          if result.returncode == 0:
            # Parse foam output to extract sphere data
            # This is a placeholder - you'd need to adapt based on foam's actual output
            spheres = parse_foam_output(result.stdout)
            collision_spheres[link_name] = spheres
            self_collision_spheres[link_name] = spheres # Same for now
            print(f" SUCCESS: Generated {len(spheres)} spheres")
          else:
            print(f" ERROR: Foam failed: {result.stderr}")
            # Generate fallback sphere
            collision_spheres[link_name] = [{"origin": [0.0, 0.0, 0.0], "radius": 0.05}]
            self_collision_spheres[link_name] = [{"origin": [0.0, 0.0, 0.0], "radius": 0.05}]
            
        except Exception as e:
          print(f" ERROR: Error running foam: {e}")
          # Generate fallback sphere
          collision_spheres[link_name] = [{"origin": [0.0, 0.0, 0.0], "radius": 0.05}]
          self_collision_spheres[link_name] = [{"origin": [0.0, 0.0, 0.0], "radius": 0.05}]
      else:
        print(f" WARNING: Mesh file not found: {mesh_path}")
  
  # Save generated spheres to collision_spheres subdirectory
  collision_spheres_dir = urdf_dir / "collision_spheres"
  collision_spheres_dir.mkdir(exist_ok=True)  # Create directory if it doesn't exist
  
  collision_path = collision_spheres_dir / "collision_spheres.json"
  self_collision_path = collision_spheres_dir / "self_collision_spheres.json"
  
  with open(collision_path, 'w') as f:
    json.dump(collision_spheres, f, indent=2)
  
  with open(self_collision_path, 'w') as f:
    json.dump(self_collision_spheres, f, indent=2)
  
  print(f"SUCCESS: Saved collision_spheres.json and self_collision_spheres.json to {collision_spheres_dir}")
  
  # Re-run analysis to show updated table
  analyze_robot_spheres(urdf_path)
  return True


def parse_foam_output(output):
  """Parse foam output to extract sphere data from JSON format."""
  try:
    import json
    foam_data = json.loads(output)
    
    # Foam outputs an array of results, we want the best one (lowest score)
    if not foam_data:
      print("WARNING: Empty foam output")
      return [{"origin": [0.0, 0.0, 0.0], "radius": 0.05}]
    
    # Find the result with the best (lowest) score
    best_result = min(foam_data, key=lambda x: x.get('best', float('inf')))
    
    # Extract just the spheres, ignoring mean/best/worst scores
    spheres = best_result.get('spheres', [])
    
    print(f"  Foam result: best={best_result.get('best', 'N/A')}, spheres={len(spheres)}")
    
    return spheres
    
  except json.JSONDecodeError as e:
    print(f"WARNING: Failed to parse foam JSON output: {e}")
    print(f"  Raw output: {output[:200]}...")
    return [{"origin": [0.0, 0.0, 0.0], "radius": 0.05}]
  except Exception as e:
    print(f"WARNING: Error processing foam output: {e}")
    return [{"origin": [0.0, 0.0, 0.0], "radius": 0.05}]


def load_link_for_editing(urdf_path, link_name, sphere_type="collision", apply_coordinate_fix=True):
  """
  Step 5: Load a specific link's mesh and spheres into Blender for editing.
  sphere_type: "collision" or "self_collision"
  apply_coordinate_fix: Apply coordinate system transformation for .obj files (Y-up to Z-up)
  """
  print(f"\n=== LOADING {link_name} FOR {sphere_type.upper()} EDITING ===")
  
  # Clear existing objects
  clear_scene()
  
  # Create collections for organization
  mesh_collection, sphere_collection = create_collections(link_name, sphere_type)
  print(f"Created collections: '{mesh_collection.name}' and '{sphere_collection.name}'")
  
  urdf_dir = Path(urdf_path).parent
  
  # Parse URDF to get mesh path
  try:
    links = parse_urdf(urdf_path)
    link_info = links.get(link_name)
    if not link_info:
      print(f"ERROR: Link '{link_name}' not found in URDF")
      return False
  except Exception as e:
    print(f"ERROR: Error parsing URDF: {e}")
    return False
  
  # Load mesh
  mesh_path = None
  if link_info['visual_mesh']:
    mesh_path = urdf_dir / link_info['visual_mesh']
  elif link_info['collision_mesh']:
    mesh_path = urdf_dir / link_info['collision_mesh']
  
  if mesh_path and mesh_path.exists():
    try:
      # Import mesh based on file type
      if mesh_path.suffix.lower() == '.obj':
        bpy.ops.wm.obj_import(filepath=str(mesh_path))
      elif mesh_path.suffix.lower() == '.stl':
        bpy.ops.wm.stl_import(filepath=str(mesh_path))
      elif mesh_path.suffix.lower() == '.dae':
        bpy.ops.wm.collada_import(filepath=str(mesh_path))
      elif mesh_path.suffix.lower() == '.ply':
        bpy.ops.wm.ply_import(filepath=str(mesh_path))
      else:
        print(f"ERROR: Unsupported mesh format: {mesh_path.suffix}")
        print(f"  Supported formats: .obj, .stl, .dae, .ply")
        return False
      
      # Get all imported mesh objects (some .obj files contain multiple meshes)
      imported_objects = bpy.context.selected_objects[:] if bpy.context.selected_objects else []
      
      if imported_objects:
        print(f"Imported {len(imported_objects)} objects from {mesh_path.name}")
        
        # Apply coordinate system fix to ALL imported objects
        if apply_coordinate_fix and mesh_path.suffix.lower() == '.obj':
          for obj in imported_objects:
            if obj.type == 'MESH':
              apply_obj_coordinate_fix(obj)
        
        # Rename and lock all mesh objects
        main_mesh = None
        for i, obj in enumerate(imported_objects):
          if obj.type == 'MESH':
            if i == 0:
              # First mesh gets the main name
              obj.name = f"{link_name}_mesh"
              main_mesh = obj
            else:
              # Additional meshes get numbered names
              obj.name = f"{link_name}_mesh_{i}"
            
            # Lock the mesh object
            obj.lock_location = (True, True, True)
            obj.lock_rotation = (True, True, True)
            obj.lock_scale = (True, True, True)
            
            # Add gray material to mesh
            add_mesh_material(obj)
            
            # Move to mesh collection
            move_object_to_collection(obj, mesh_collection)
            
            print(f" SUCCESS: Loaded and locked: {obj.name}")
        
        print(f"SUCCESS: Loaded and processed {len(imported_objects)} mesh objects")
      else:
        print(f"WARNING: Mesh imported but no objects selected")
      
    except Exception as e:
      print(f"ERROR: Error loading mesh: {e}")
      print(f"  Mesh path: {mesh_path}")
      print(f"  Trying alternative import methods...")
      
      # Try legacy import operators as fallback
      try:
        if mesh_path.suffix.lower() == '.obj':
          bpy.ops.import_scene.obj(filepath=str(mesh_path))
        elif mesh_path.suffix.lower() == '.stl':
          bpy.ops.import_mesh.stl(filepath=str(mesh_path))
        
        # Apply coordinate fix for legacy import too - handle multiple meshes
        imported_objects = bpy.context.selected_objects[:] if bpy.context.selected_objects else []
        
        if imported_objects:
          print(f"Legacy import: {len(imported_objects)} objects from {mesh_path.name}")
          
          # Apply coordinate fix to ALL imported objects
          if apply_coordinate_fix and mesh_path.suffix.lower() == '.obj':
            for obj in imported_objects:
              if obj.type == 'MESH':
                apply_obj_coordinate_fix(obj)
          
          # Rename and lock all mesh objects
          for i, obj in enumerate(imported_objects):
            if obj.type == 'MESH':
              if i == 0:
                obj.name = f"{link_name}_mesh"
              else:
                obj.name = f"{link_name}_mesh_{i}"
              
              # Lock the mesh object
              obj.lock_location = (True, True, True)
              obj.lock_rotation = (True, True, True)
              obj.lock_scale = (True, True, True)
              
              # Add gray material to mesh
              add_mesh_material(obj)
              
              # Move to mesh collection
              move_object_to_collection(obj, mesh_collection)
              
              print(f" SUCCESS: Legacy loaded and locked: {obj.name}")
          
          print(f"SUCCESS: Legacy import processed {len(imported_objects)} mesh objects")
        else:
          print(f"WARNING: Legacy import succeeded but no objects found")
        
      except Exception as e2:
        print(f"ERROR: Legacy import also failed: {e2}")
        print(f"  Continuing without mesh...")
  else:
    print(f"WARNING: No mesh found for link '{link_name}'")
  
  # Load spheres from collision_spheres subdirectory
  sphere_file = f"{sphere_type}_spheres.json"
  collision_spheres_dir = urdf_dir / "collision_spheres"
  sphere_path = collision_spheres_dir / sphere_file
  
  if sphere_path.exists():
    with open(sphere_path, 'r') as f:
      sphere_data = json.load(f)
    
    link_spheres = sphere_data.get(link_name, [])
    
    for i, sphere in enumerate(link_spheres):
      origin = sphere['origin']
      radius = sphere['radius']
      sphere_obj = add_sphere(origin[0], origin[1], origin[2], radius, 
                  name_prefix=f"{link_name}_{sphere_type}_", index=i)
      
      # Move sphere to sphere collection
      move_object_to_collection(sphere_obj, sphere_collection)
      
      print(f" SUCCESS: Added sphere {i}: pos=({origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}), r={radius:.3f}")
    
    print(f"SUCCESS: Loaded {len(link_spheres)} {sphere_type} spheres for {link_name}")
    return True
  else:
    print(f"ERROR: No {sphere_file} found")
    return False



def clear_scene():
  """Clear all objects from the Blender scene and remove link-related collections."""
  # Remove all objects first
  bpy.ops.object.select_all(action='SELECT')
  bpy.ops.object.delete(use_global=False)
  
  # Remove collections that match our naming patterns
  collections_to_remove = []
  
  for collection in bpy.data.collections:
    # Check if collection matches our naming patterns
    if (collection.name.endswith('_meshes') or 
      collection.name.endswith('_collision_spheres') or 
      collection.name.endswith('_self_collision_spheres')):
      collections_to_remove.append(collection)
  
  # Remove the collections
  for collection in collections_to_remove:
    # Store the name before removing (becomes invalid after removal)
    collection_name = collection.name
    
    # Remove from scene hierarchy first
    if collection_name in bpy.context.scene.collection.children:
      bpy.context.scene.collection.children.unlink(collection)
    
    # Remove from blend data
    bpy.data.collections.remove(collection)
    print(f"CLEANUP: Removed collection: {collection_name}")
  
  if collections_to_remove:
    print(f"SUCCESS: Cleaned up {len(collections_to_remove)} collections")


def add_mesh_material(mesh_obj):
  """Add a gray material to the mesh object."""
  mat = bpy.data.materials.new(name="MeshMaterial")
  mat.use_nodes = True
  bsdf = mat.node_tree.nodes.get("Principled BSDF")
  bsdf.inputs["Base Color"].default_value = (0.7, 0.7, 0.7, 1.0) # Gray
  bsdf.inputs["Alpha"].default_value = 0.7 # Semi-transparent
  mat.blend_method = 'BLEND'
  mesh_obj.data.materials.append(mat)


def save_edited_spheres(urdf_path, link_name, sphere_type="collision"):
  """
  Step 7: Save edited spheres back to JSON file.
  """
  print(f"\n=== SAVING {sphere_type.upper()} SPHERES FOR {link_name} ===")
  
  urdf_dir = Path(urdf_path).parent
  collision_spheres_dir = urdf_dir / "collision_spheres"
  collision_spheres_dir.mkdir(exist_ok=True)  # Create directory if it doesn't exist
  
  sphere_file = f"{sphere_type}_spheres.json"
  sphere_path = collision_spheres_dir / sphere_file
  
  # Extract sphere data from Blender
  spheres = []
  sphere_prefix = f"{link_name}_{sphere_type}_"
  
  for obj in bpy.data.objects:
    if obj.type == 'MESH' and obj.name.startswith(sphere_prefix):
      loc = obj.location
      radius = obj.dimensions.x / 2 # assumes uniform scaling
      spheres.append({
        "origin": [round(loc.x, 6), round(loc.y, 6), round(loc.z, 6)],
        "radius": round(radius, 6)
      })
  
  print(f"Extracted {len(spheres)} spheres from Blender scene")
  
  # Load existing data or create new
  if sphere_path.exists():
    with open(sphere_path, 'r') as f:
      sphere_data = json.load(f)
  else:
    sphere_data = {}
  
  # Update data for this link
  sphere_data[link_name] = spheres
  
  # Save back to file
  with open(sphere_path, 'w') as f:
    json.dump(sphere_data, f, indent=2)
  
  print(f"SUCCESS: Saved {len(spheres)} spheres to {sphere_file}")
  
  # Show summary
  for i, sphere in enumerate(spheres):
    origin = sphere['origin']
    radius = sphere['radius']
    print(f" Sphere {i}: pos=({origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}), r={radius:.3f}")


# Legacy functions (keeping for compatibility)
def add_spheres_from_json(path):
  """Load and add spheres from a JSON file organized per link."""
  with open(path, 'r') as f:
    data = json.load(f)

  if not isinstance(data, dict):
    raise ValueError("Expected a dictionary with link names as keys.")

  counter = 0
  for link_name, link_data in data.items():
    if not isinstance(link_data, dict) or 'spheres' not in link_data:
      continue
    for sphere in link_data['spheres']:
      origin = sphere['origin']
      radius = sphere['radius']
      add_sphere(*origin, radius, name_prefix=f"{link_name}_", index=counter)
      counter += 1


def convert_franka_spheres_to_json(spheres):
  """Convert the Franka _SPHERES format into a JSON dict organized per link."""
  out = {}
  for radius, link_dict in spheres:
    for link, arr in link_dict.items():
      if link not in out:
        out[link] = {"spheres": []}
      for coords in arr:
        out[link]["spheres"].append({"origin": list(coords), "radius": radius})
  return out


def convert_self_collision_spheres_to_json(spheres):
  """Convert the _SELF_COLLISION_SPHERES list into a JSON dict organized per link."""
  out = {}
  for link, pos, radius in spheres:
    if link not in out:
      out[link] = {"spheres": []}
    out[link]["spheres"].append({"origin": pos, "radius": radius})
  return out


def apply_obj_coordinate_fix(mesh_obj):
  """
  Apply coordinate system transformation for .obj files.
  Common issue: .obj files use Y-up, but URDF/robotics uses Z-up.
  This applies a -90° rotation around X-axis to convert Y-up to Z-up.
  """
  import mathutils
  
  print(f"TOGGLE: Applying coordinate system fix for {mesh_obj.name}")
  
  # Ensure object is selected and active
  bpy.context.view_layer.objects.active = mesh_obj
  mesh_obj.select_set(True)
  
  # Apply transformation: -90° rotation around X-axis (Y-up to Z-up)
  # This is the most common transformation needed for .obj files in robotics
  rotation_matrix = mathutils.Matrix.Rotation(-1.5707963267948966, 4, 'X') # -90° in radians
  mesh_obj.matrix_world = rotation_matrix @ mesh_obj.matrix_world
  
  # Apply the transformation to make it permanent
  bpy.context.view_layer.update()
  
  print(f"  SUCCESS: Applied Y-up to Z-up transformation")


def apply_custom_mesh_transform(mesh_obj, rotation_axis='X', rotation_degrees=-90):
  """
  Apply custom coordinate system transformation to a mesh object.
  
  Args:
    mesh_obj: Blender mesh object
    rotation_axis: 'X', 'Y', or 'Z' 
    rotation_degrees: Rotation in degrees (e.g., -90, 90, 180)
  """
  import mathutils
  import math
  
  print(f"TOGGLE: Applying custom transform: {rotation_degrees}° around {rotation_axis}-axis")
  
  # Ensure object is selected and active
  bpy.context.view_layer.objects.active = mesh_obj
  mesh_obj.select_set(True)
  
  # Convert degrees to radians
  rotation_radians = math.radians(rotation_degrees)
  
  # Apply transformation
  rotation_matrix = mathutils.Matrix.Rotation(rotation_radians, 4, rotation_axis)
  mesh_obj.matrix_world = rotation_matrix @ mesh_obj.matrix_world
  
  # Apply the transformation to make it permanent
  bpy.context.view_layer.update()
  
  print(f"  SUCCESS: Applied {rotation_degrees}° {rotation_axis}-axis rotation")


def test_mesh_alignment(urdf_path, link_name):
  """
  Interactive function to test different coordinate transformations
  and find the correct one for your robot.
  """
  print(f"\nTESTING MESH ALIGNMENT FOR {link_name}")
  print("=" * 50)
  
  # Load without coordinate fix first
  print("Loading mesh without coordinate fix...")
  success = load_link_for_editing(urdf_path, link_name, 'collision', apply_coordinate_fix=False)
  
  if not success:
    print("ERROR: Failed to load link")
    return
  
  # Find the mesh object
  mesh_obj = None
  for obj in bpy.data.objects:
    if obj.name.endswith('_mesh'):
      mesh_obj = obj
      break
  
  if not mesh_obj:
    print("ERROR: No mesh object found")
    return
  
  print(f"\nMESH ALIGNMENT TESTING")
  print("Try these common transformations:")
  print("1. -90° X-axis (Y-up to Z-up) - Most common for .obj files")
  print("2. +90° X-axis (alternative)")
  print("3. -90° Y-axis")
  print("4. +90° Y-axis") 
  print("5. 180° Z-axis")
  
  transformations = [
    ('X', -90, "Y-up to Z-up (most common)"),
    ('X', 90, "Alternative X rotation"),
    ('Y', -90, "Y-axis rotation"),
    ('Y', 90, "Alternative Y rotation"),
    ('Z', 180, "Z-axis flip")
  ]
  
  print(f"\nTesting transformations on {mesh_obj.name}...")
  print("Look at the 3D viewport to see which alignment looks correct!")
  
  # Store original transform
  original_matrix = mesh_obj.matrix_world.copy()
  
  for i, (axis, degrees, description) in enumerate(transformations):
    print(f"\n--- Test {i+1}: {description} ---")
    
    # Reset to original
    mesh_obj.matrix_world = original_matrix.copy()
    
    # Apply test transformation
    apply_custom_mesh_transform(mesh_obj, axis, degrees)
    
    # Wait for user input
    response = input("Does this look correct? (y/n/quit): ").lower()
    if response == 'y':
      print(f"SUCCESS: Found correct transformation: {degrees}° {axis}-axis")
      print(f"  Use: apply_custom_mesh_transform(mesh_obj, '{axis}', {degrees})")
      return
    elif response == 'quit':
      break
  
  # Reset to original if no good match found
  mesh_obj.matrix_world = original_matrix.copy()
  print("TOGGLE: Reset to original orientation")
  print("You may need to find a custom transformation for this robot")


def get_mesh_object(link_name):
  """Get the mesh object for a loaded link."""
  mesh_name = f"{link_name}_mesh"
  return bpy.data.objects.get(mesh_name)


def debug_import_behavior(urdf_path, link_name):
  """
  Debug function to understand how Blender imports different file formats
  and what coordinate transformations are applied by default.
  """
  print(f"\nDEBUGGING IMPORT BEHAVIOR FOR {link_name}")
  print("=" * 60)
  
  urdf_dir = Path(urdf_path).parent
  
  # Parse URDF to get mesh path
  try:
    links = parse_urdf(urdf_path)
    link_info = links.get(link_name)
    if not link_info:
      print(f"ERROR: Link '{link_name}' not found in URDF")
      return
  except Exception as e:
    print(f"ERROR: Error parsing URDF: {e}")
    return
  
  # Get mesh path
  mesh_path = None
  if link_info['visual_mesh']:
    mesh_path = urdf_dir / link_info['visual_mesh']
  elif link_info['collision_mesh']:
    mesh_path = urdf_dir / link_info['collision_mesh']
  
  if not mesh_path or not mesh_path.exists():
    print(f"ERROR: No mesh found for link '{link_name}'")
    return
  
  print(f"Mesh file: {mesh_path}")
  print(f"File format: {mesh_path.suffix}")
  
  # Test 1: Import WITHOUT any coordinate fixes
  print(f"\nTEST 1: Import {mesh_path.suffix} file WITHOUT coordinate fixes")
  clear_scene()
  
  try:
    # Import based on file type WITHOUT coordinate fixes
    if mesh_path.suffix.lower() == '.obj':
      print("  Using: bpy.ops.wm.obj_import()")
      bpy.ops.wm.obj_import(filepath=str(mesh_path))
    elif mesh_path.suffix.lower() == '.stl':
      print("  Using: bpy.ops.wm.stl_import()")
      bpy.ops.wm.stl_import(filepath=str(mesh_path))
    elif mesh_path.suffix.lower() == '.dae':
      print("  Using: bpy.ops.wm.collada_import()")
      bpy.ops.wm.collada_import(filepath=str(mesh_path))
    
    # Check what rotation was applied by the importer
    mesh_obj = bpy.context.selected_objects[0] if bpy.context.selected_objects else None
    if mesh_obj:
      rotation = mesh_obj.rotation_euler
      print(f"  Default rotation after import:")
      print(f"   X: {rotation.x:.3f} rad ({rotation.x * 57.2958:.1f}°)")
      print(f"   Y: {rotation.y:.3f} rad ({rotation.y * 57.2958:.1f}°)")
      print(f"   Z: {rotation.z:.3f} rad ({rotation.z * 57.2958:.1f}°)")
      
      # Check if rotation is close to 90° on any axis
      x_deg = rotation.x * 57.2958
      y_deg = rotation.y * 57.2958
      z_deg = rotation.z * 57.2958
      
      if abs(x_deg - 90) < 5:
        print(f"  WARNING: X rotation (~90°) detected - typical Y-up to Z-up conversion")
      elif abs(x_deg + 90) < 5:
        print(f"  WARNING: X rotation (~-90°) detected - typical Z-up to Y-up conversion")
      
      if abs(y_deg) > 5 or abs(z_deg) > 5:
        print(f"  WARNING: Y or Z rotation detected - unusual for coordinate conversion")
    else:
      print("  ERROR: No object found after import")
  
  except Exception as e:
    print(f"  ERROR: Import failed: {e}")
  
  # Test 2: Load spheres to compare positions
  print(f"\nTEST 2: Load spheres for comparison")
  sphere_file = "collision_spheres.json"
  sphere_path = urdf_dir / sphere_file
  
  if sphere_path.exists():
    with open(sphere_path, 'r') as f:
      sphere_data = json.load(f)
    
    link_spheres = sphere_data.get(link_name, [])
    
    if link_spheres:
      first_sphere = link_spheres[0]
      sphere_obj = add_sphere(
        first_sphere['origin'][0], 
        first_sphere['origin'][1], 
        first_sphere['origin'][2], 
        first_sphere['radius'],
        name_prefix=f"debug_sphere_"
      )
      print(f"  Added sphere at: {first_sphere['origin']}")
      print(f"  Sphere radius: {first_sphere['radius']}")
    else:
      print("  WARNING: No spheres found for this link")
  else:
    print("  WARNING: No collision_spheres.json found")
  
  print(f"\nANALYSIS:")
  print("1. Check the 3D viewport - do mesh and sphere align?")
  print("2. Look at Object Properties panel (bottom right)")
  print("3. Note the rotation values shown there")
  print("4. Try manually setting X rotation to 0° and see if alignment improves")


def check_blender_import_settings():
  """
  Check Blender's import addon settings to understand default behavior.
  """
  print(f"\n🔧 CHECKING BLENDER IMPORT ADDON SETTINGS")
  print("=" * 50)
  
  import addon_utils
  
  # Check which import addons are enabled
  import_addons = [
    ('io_scene_obj', 'OBJ Import/Export'),
    ('io_mesh_stl', 'STL Import/Export'), 
    ('io_scene_gltf2', 'glTF 2.0 Import/Export'),
    ('io_mesh_ply', 'PLY Import/Export'),
    ('io_scene_x3d', 'X3D Import/Export'),
  ]
  
  for addon_name, description in import_addons:
    try:
      addon_info = addon_utils.check(addon_name)
      status = "SUCCESS: ENABLED" if addon_info[0] else "ERROR: DISABLED"
      print(f"{description:<25} {status}")
    except:
      print(f"{description:<25} ❓ UNKNOWN")
  
  print(f"\nTIP: If .obj files are rotated but .stl files aren't,")
  print(f"  it suggests the .obj importer has different default settings.")


def compare_file_formats(urdf_path, link_name):
  """
  If multiple formats exist for the same link, compare their import behavior.
  """
  print(f"\nCOMPARING FILE FORMATS FOR {link_name}")
  print("=" * 50)
  
  urdf_dir = Path(urdf_path).parent
  
  # Look for different format versions of the same mesh
  possible_formats = ['.obj', '.stl', '.dae', '.ply']
  found_meshes = []
  
  # Check visual and collision mesh directories
  for mesh_dir in ['meshes/visual', 'meshes/collision']:
    full_dir = urdf_dir / mesh_dir
    if full_dir.exists():
      for format_ext in possible_formats:
        # Look for files that might be for this link
        pattern = f"*{link_name.replace('panda_', '')}*{format_ext}"
        matching_files = list(full_dir.glob(pattern))
        for file in matching_files:
          found_meshes.append(file)
  
  if found_meshes:
    print("Found alternative mesh formats:")
    for mesh_file in found_meshes:
      print(f"  - {mesh_file}")
    
    print("\nYou can test these different formats to see")
    print("  which ones import with correct orientation!")
  else:
    print("Only one mesh format found for this link")


def load_link_without_coordinate_fix(urdf_path, link_name, sphere_type="collision"):
  """
  Load link exactly as Blender imports it, without any coordinate fixes.
  Useful for debugging the raw import behavior.
  """
  print(f"\nLOADING {link_name} WITHOUT COORDINATE FIXES")
  print("(For debugging import behavior)")
  
  return load_link_for_editing(urdf_path, link_name, sphere_type, apply_coordinate_fix=False)


def print_sphere_info_for_link(link_name, sphere_type="collision"):
  """Print info for all spheres belonging to a specific link and sphere type."""
  sphere_prefix = f"{link_name}_{sphere_type}_"
  found_spheres = []
  
  for obj in bpy.data.objects:
    if obj.type == 'MESH' and obj.name.startswith(sphere_prefix):
      found_spheres.append(obj)
  
  if not found_spheres:
    print(f"ERROR: No spheres found with prefix '{sphere_prefix}'")
    return
  
  print(f"\nSPHERE INFO FOR {link_name} ({sphere_type.upper()})")
  print("=" * 50)
  for obj in found_spheres:
    loc = obj.location
    radius = obj.dimensions.x / 2 # assumes uniform scaling
    print(f" {obj.name}: pos=({loc.x:.3f}, {loc.y:.3f}, {loc.z:.3f}), r={radius:.3f}")


def print_all_current_spheres():
  """Print info for all sphere objects currently in the scene."""
  sphere_objects = []
  
  # Look for objects that might be spheres based on common naming patterns
  for obj in bpy.data.objects:
    if obj.type == 'MESH':
      # Check for sphere-like names
      obj_name_lower = obj.name.lower()
      if any(keyword in obj_name_lower for keyword in ['sphere', 'collision', 'self_collision']):
        sphere_objects.append(obj)
  
  if not sphere_objects:
    print(f"ERROR: No sphere objects found in scene")
    return
  
  print(f"\nALL SPHERE OBJECTS IN SCENE ({len(sphere_objects)} found)")
  print("=" * 60)
  for obj in sphere_objects:
    loc = obj.location
    radius = obj.dimensions.x / 2 # assumes uniform scaling
    print(f" {obj.name}: pos=({loc.x:.3f}, {loc.y:.3f}, {loc.z:.3f}), r={radius:.3f}")


def add_manual_sphere(link_name, sphere_type, x, y, z, radius, description="manual"):
  """
  Add a sphere that will be included when saving spheres for a link.
  
  Args:
    link_name: The robot link name (e.g., 'panda_link1')
    sphere_type: 'collision' or 'self_collision'
    x, y, z: Position coordinates
    radius: Sphere radius
    description: Optional description for the sphere name
  
  Returns:
    The created sphere object
  """
  # Find the next available index for this link/type
  sphere_prefix = f"{link_name}_{sphere_type}_"
  existing_indices = []
  
  for obj in bpy.data.objects:
    if obj.name.startswith(sphere_prefix):
      # Extract index from name like "panda_link1_collision_0"
      try:
        index_str = obj.name[len(sphere_prefix):]
        # Handle names with description like "panda_link1_collision_0_manual"
        if '_' in index_str:
          index_str = index_str.split('_')[0]
        index = int(index_str)
        existing_indices.append(index)
      except ValueError:
        continue
  
  # Get next available index
  next_index = max(existing_indices, default=-1) + 1
  
  # Create sphere with proper naming
  sphere_name = f"{sphere_prefix}{next_index}"
  if description and description != "manual":
    sphere_name += f"_{description}"
  
  sphere_obj = add_sphere(x, y, z, radius, name_prefix="", index=None)
  sphere_obj.name = sphere_name
  
  print(f"SUCCESS: Added manual sphere: {sphere_name}")
  print(f"  Position: ({x:.3f}, {y:.3f}, {z:.3f}), Radius: {radius:.3f}")
  print(f"  This sphere will be saved when you call save_edited_spheres()")
  
  return sphere_obj


def get_save_naming_requirements(link_name, sphere_type):
  """
  Show the naming requirements for manual spheres to be included in saves.
  """
  sphere_prefix = f"{link_name}_{sphere_type}_"
  
  print(f"\nMANUAL SPHERE NAMING REQUIREMENTS")
  print("=" * 50)
  print(f"Link: {link_name}")
  print(f"Sphere type: {sphere_type}")
  print(f"Required prefix: '{sphere_prefix}'")
  print()
  print("NAMING EXAMPLES:")
  print(f" SUCCESS: {sphere_prefix}0")
  print(f" SUCCESS: {sphere_prefix}1") 
  print(f" SUCCESS: {sphere_prefix}5_custom")
  print(f" SUCCESS: {sphere_prefix}10_manual")
  print(f" ERROR: custom_sphere (wrong prefix)")
  print(f" ERROR: {link_name}_sphere_0 (wrong format)")
  print()
  print("EASY WAY TO ADD MANUAL SPHERES:")
  print(f"  spherification_utils.add_manual_sphere('{link_name}', '{sphere_type}', x, y, z, radius)")
  print()
  print("WHAT GETS SAVED:")
  print(f"  - All objects with names starting with '{sphere_prefix}'")
  print(f"  - Object type must be 'MESH'")
  print(f"  - Radius calculated from object dimensions (X-axis)")


def count_spheres_for_link(link_name, sphere_type="collision"):
  """Count how many spheres exist for a given link and sphere type."""
  sphere_prefix = f"{link_name}_{sphere_type}_"
  count = 0
  
  for obj in bpy.data.objects:
    if obj.type == 'MESH' and obj.name.startswith(sphere_prefix):
      count += 1
  
  return count


def generate_and_load_spheres(urdf_path, link_name, sphere_type="collision", clear_existing=True):
  """
  Generate spheres for a specific link using foam and load them into Blender immediately.
  This replaces existing spheres in the scene if clear_existing=True.
  
  Args:
    urdf_path: Path to URDF file
    link_name: Name of the link to generate spheres for
    sphere_type: "collision" or "self_collision" 
    clear_existing: Whether to clear existing spheres for this link first
  """
  print(f"\n=== GENERATING AND LOADING SPHERES FOR {link_name} ===")
  
  # Parse URDF to get mesh path
  try:
    links = parse_urdf(urdf_path)
    link_info = links.get(link_name)
    if not link_info:
      print(f"ERROR: Link '{link_name}' not found in URDF")
      return False
  except Exception as e:
    print(f"ERROR: Error parsing URDF: {e}")
    return False
  
  urdf_dir = Path(urdf_path).parent
  
  # Get mesh path
  mesh_path = None
  if link_info['visual_mesh']:
    mesh_path = urdf_dir / link_info['visual_mesh']
  elif link_info['collision_mesh']:
    mesh_path = urdf_dir / link_info['collision_mesh']
  
  if not mesh_path or not mesh_path.exists():
    print(f"ERROR: No mesh found for link '{link_name}'")
    return False
  
  print(f"Generating spheres for mesh: {mesh_path}")
  
  # Clear existing spheres for this link if requested
  if clear_existing:
    sphere_prefix = f"{link_name}_{sphere_type}_"
    removed_count = 0
    
    # Get list of objects to remove (can't modify collection while iterating)
    objects_to_remove = []
    for obj in bpy.data.objects:
      if obj.type == 'MESH' and obj.name.startswith(sphere_prefix):
        objects_to_remove.append(obj)
    
    # Remove them
    for obj in objects_to_remove:
      bpy.data.objects.remove(obj, do_unlink=True)
      removed_count += 1
    
    if removed_count > 0:
      print(f"CLEANUP: Removed {removed_count} existing spheres")
  
  # Generate spheres using foam
  generated_spheres = []
  try:
    # Run foam sphere generation
    result = subprocess.run([
      'python3', '/opt/foam/scripts/generate_spheres.py',
      str(mesh_path)
    ], capture_output=True, text=True, timeout=30)
    
    if result.returncode == 0:
      # Parse foam output to extract sphere data
      generated_spheres = parse_foam_output(result.stdout)
      print(f"SUCCESS: Foam generated {len(generated_spheres)} spheres")
    else:
      print(f"ERROR: Foam failed: {result.stderr}")
      # Generate fallback spheres
      generated_spheres = [
        {"origin": [0.0, 0.0, 0.0], "radius": 0.06 if sphere_type == "collision" else 0.1}
      ]
      print(f"WARNING: Using fallback sphere")
      
  except Exception as e:
    print(f"ERROR: Error running foam: {e}")
    # Generate fallback spheres
    generated_spheres = [
      {"origin": [0.0, 0.0, 0.0], "radius": 0.06 if sphere_type == "collision" else 0.1}
    ]
    print(f"WARNING: Using fallback sphere")
  
  # Create collections for organization
  mesh_collection, sphere_collection = create_collections(link_name, sphere_type)
  
  # Load generated spheres into Blender
  for i, sphere in enumerate(generated_spheres):
    origin = sphere['origin']
    radius = sphere['radius']
    sphere_obj = add_sphere(origin[0], origin[1], origin[2], radius, 
                name_prefix=f"{link_name}_{sphere_type}_", index=i)
    
    # Move sphere to sphere collection
    move_object_to_collection(sphere_obj, sphere_collection)
    
    print(f" SUCCESS: Added sphere {i}: pos=({origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}), r={radius:.3f}")
  
  # Save to JSON file in collision_spheres subdirectory
  collision_spheres_dir = urdf_dir / "collision_spheres"
  collision_spheres_dir.mkdir(exist_ok=True)  # Create directory if it doesn't exist
  
  sphere_file = f"{sphere_type}_spheres.json"
  sphere_path = collision_spheres_dir / sphere_file
  
  # Load existing data or create new
  if sphere_path.exists():
    with open(sphere_path, 'r') as f:
      sphere_data = json.load(f)
  else:
    sphere_data = {}
  
  # Update data for this link
  sphere_data[link_name] = generated_spheres
  
  # Save back to file
  with open(sphere_path, 'w') as f:
    json.dump(sphere_data, f, indent=2)
  
  print(f"Saved {len(generated_spheres)} spheres to {collision_spheres_dir}/{sphere_file}")
  print(f"SUCCESS: Generated and loaded {len(generated_spheres)} spheres for {link_name}")
  
  return True


def create_collections(link_name, sphere_type):
  """
  Create or get collections for organizing meshes and spheres.
  
  Returns:
    tuple: (mesh_collection, sphere_collection)
  """
  # Get or create mesh collection
  mesh_collection_name = f"{link_name}_meshes"
  if mesh_collection_name in bpy.data.collections:
    mesh_collection = bpy.data.collections[mesh_collection_name]
  else:
    mesh_collection = bpy.data.collections.new(mesh_collection_name)
    bpy.context.scene.collection.children.link(mesh_collection)
  
  # Get or create sphere collection
  sphere_collection_name = f"{link_name}_{sphere_type}_spheres"
  if sphere_collection_name in bpy.data.collections:
    sphere_collection = bpy.data.collections[sphere_collection_name]
  else:
    sphere_collection = bpy.data.collections.new(sphere_collection_name)
    bpy.context.scene.collection.children.link(sphere_collection)
  
  return mesh_collection, sphere_collection


def move_object_to_collection(obj, target_collection):
  """Move an object to a specific collection, removing it from others."""
  # Remove from all collections first
  for collection in obj.users_collection:
    collection.objects.unlink(obj)
  
  # Add to target collection
  target_collection.objects.link(obj)


def hide_meshes(link_name):
  """Hide all mesh objects for a link."""
  mesh_collection_name = f"{link_name}_meshes"
  if mesh_collection_name in bpy.data.collections:
    collection = bpy.data.collections[mesh_collection_name]
    collection.hide_viewport = True
    print(f"HIDDEN: Hidden meshes for {link_name}")
  else:
    print(f"ERROR: No mesh collection found for {link_name}")


def show_meshes(link_name):
  """Show all mesh objects for a link."""
  mesh_collection_name = f"{link_name}_meshes"
  if mesh_collection_name in bpy.data.collections:
    collection = bpy.data.collections[mesh_collection_name]
    collection.hide_viewport = False
    print(f"SHOWN: Shown meshes for {link_name}")
  else:
    print(f"ERROR: No mesh collection found for {link_name}")


def hide_spheres(link_name, sphere_type="collision"):
  """Hide all sphere objects for a link."""
  sphere_collection_name = f"{link_name}_{sphere_type}_spheres"
  if sphere_collection_name in bpy.data.collections:
    collection = bpy.data.collections[sphere_collection_name]
    collection.hide_viewport = True
    print(f"HIDDEN: Hidden {sphere_type} spheres for {link_name}")
  else:
    print(f"ERROR: No sphere collection found for {link_name}")


def show_spheres(link_name, sphere_type="collision"):
  """Show all sphere objects for a link."""
  sphere_collection_name = f"{link_name}_{sphere_type}_spheres"
  if sphere_collection_name in bpy.data.collections:
    collection = bpy.data.collections[sphere_collection_name]
    collection.hide_viewport = False
    print(f"SHOWN: Shown {sphere_type} spheres for {link_name}")
  else:
    print(f"ERROR: No sphere collection found for {link_name}")


def toggle_mesh_visibility(link_name):
  """Toggle visibility of all meshes for a link."""
  mesh_collection_name = f"{link_name}_meshes"
  if mesh_collection_name in bpy.data.collections:
    collection = bpy.data.collections[mesh_collection_name]
    collection.hide_viewport = not collection.hide_viewport
    status = "hidden" if collection.hide_viewport else "shown"
    print(f"TOGGLE: Toggled meshes for {link_name}: {status}")
  else:
    print(f"ERROR: No mesh collection found for {link_name}")


def toggle_sphere_visibility(link_name, sphere_type="collision"):
  """Toggle visibility of all spheres for a link."""
  sphere_collection_name = f"{link_name}_{sphere_type}_spheres"
  if sphere_collection_name in bpy.data.collections:
    collection = bpy.data.collections[sphere_collection_name]
    collection.hide_viewport = not collection.hide_viewport
    status = "hidden" if collection.hide_viewport else "shown"
    print(f"TOGGLE: Toggled {sphere_type} spheres for {link_name}: {status}")
  else:
    print(f"ERROR: No sphere collection found for {link_name}")


def organize_scene_collections(link_name, sphere_type="collision"):
  """
  Organize current scene objects into collections.
  Useful if you loaded objects before collection management was added.
  """
  print(f"\nORGANIZING SCENE INTO COLLECTIONS")
  print("=" * 50)
  
  # Create collections
  mesh_collection, sphere_collection = create_collections(link_name, sphere_type)
  
  # Move meshes to mesh collection
  mesh_count = 0
  for obj in bpy.data.objects:
    if obj.type == 'MESH' and obj.name.startswith(f"{link_name}_mesh"):
      move_object_to_collection(obj, mesh_collection)
      mesh_count += 1
  
  # Move spheres to sphere collection
  sphere_count = 0
  sphere_prefix = f"{link_name}_{sphere_type}_"
  for obj in bpy.data.objects:
    if obj.type == 'MESH' and obj.name.startswith(sphere_prefix):
      move_object_to_collection(obj, sphere_collection)
      sphere_count += 1
  
  print(f"SUCCESS: Organized {mesh_count} meshes into '{mesh_collection.name}' collection")
  print(f"SUCCESS: Organized {sphere_count} spheres into '{sphere_collection.name}' collection")
  
  return mesh_collection, sphere_collection


def cleanup_all_link_collections():
  """
  Manually clean up all link-related collections from the scene.
  Useful if you want to clean up without loading a new link.
  """
  print(f"\nCLEANUP: CLEANING UP ALL LINK COLLECTIONS")
  print("=" * 50)
  
  collections_to_remove = []
  
  for collection in bpy.data.collections:
    # Check if collection matches our naming patterns
    if (collection.name.endswith('_meshes') or 
      collection.name.endswith('_collision_spheres') or 
      collection.name.endswith('_self_collision_spheres')):
      collections_to_remove.append(collection)
  
  if not collections_to_remove:
    print(f"SUCCESS: No link collections found to clean up")
    return
  
  # Remove the collections
  for collection in collections_to_remove:
    # Store the name before removing (becomes invalid after removal)
    collection_name = collection.name
    
    # Remove from scene hierarchy first
    if collection_name in bpy.context.scene.collection.children:
      bpy.context.scene.collection.children.unlink(collection)
    
    # Remove from blend data
    bpy.data.collections.remove(collection)
    print(f"CLEANUP: Removed collection: {collection_name}")
  
  print(f"SUCCESS: Cleaned up {len(collections_to_remove)} collections")


def list_all_collections():
  """List all collections in the current scene."""
  print(f"\nALL COLLECTIONS IN SCENE")
  print("=" * 40)
  
  if not bpy.data.collections:
    print("No collections found")
    return
  
  for collection in bpy.data.collections:
    object_count = len(collection.objects)
    hidden = f"HIDDEN:" if collection.hide_viewport else f"SHOWN:"
    print(f"{hidden} {collection.name} ({object_count} objects)")


def cleanup_empty_collections():
  """Remove all empty collections from the scene."""
  print(f"\nCLEANUP: CLEANING UP EMPTY COLLECTIONS")
  print("=" * 40)
  
  collections_to_remove = []
  
  for collection in bpy.data.collections:
    if len(collection.objects) == 0:
      collections_to_remove.append(collection)
  
  if not collections_to_remove:
    print(f"SUCCESS: No empty collections found")
    return
  
  # Remove the collections
  for collection in collections_to_remove:
    # Store the name before removing (becomes invalid after removal)
    collection_name = collection.name
    
    # Remove from scene hierarchy first
    if collection_name in bpy.context.scene.collection.children:
      bpy.context.scene.collection.children.unlink(collection)
    
    # Remove from blend data
    bpy.data.collections.remove(collection)
    print(f"CLEANUP: Removed empty collection: {collection_name}")
  
  print(f"SUCCESS: Cleaned up {len(collections_to_remove)} empty collections")
