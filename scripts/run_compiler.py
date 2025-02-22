import os
import shutil
import glob

# Folders
build_dir = "src/mesh/compiling"
build_temp_dir = "build"  # Temporary folder created by setuptools
extra_dir = os.path.join(build_dir, "mesh")  # Extra folder 'mesh' that needs to be removed

# run build
os.system(f"python scripts/compiler.py build_ext --build-lib {build_dir}")

# Find the compiled file (.so ou .pyd) inside the extra folder
compiled_files = glob.glob(os.path.join(build_dir, "mesh", "compiling", "*.so")) + \
                 glob.glob(os.path.join(build_dir, "mesh", "compiling", "*.pyd"))

# Move only the compiled file to the correct folder
for file in compiled_files:
    shutil.move(file, build_dir)

# Remove the extra folder 'mesh' (if it still exists)
if os.path.exists(extra_dir):
    shutil.rmtree(extra_dir)

# Remove the folder 'build' (temporary files)
if os.path.exists(build_temp_dir):
    shutil.rmtree(build_temp_dir)

# Remove any file .c inside the destination folder
for c_file in glob.glob(os.path.join(build_dir, "*.c")):
    os.remove(c_file)