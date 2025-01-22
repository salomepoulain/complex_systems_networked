import sys
import os
add_parent = False

current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
print(sys.path)

suffix_to_check = "complex_system_networked"

# Check if any path in sys.path ends with the specified suffix
if not any(path.endswith(suffix_to_check) for path in sys.path):
    print("you should check your system paths, add the one ending with complex_systems_netwrked to your system paths (possibly parent path if path with src is current dir)")

def add_parent(add_parent):
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    print(f'adding {parent_dir}')
    if add_parent is True:
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

print(sys.path)