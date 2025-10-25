# change cwd to containing folder

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# get all files from here rec. ending with .c, .h, .cu, .cuh
files = []
for root, dirs, filenames in os.walk("src"):
        for filename in filenames:
                if filename.endswith((".c", ".h", ".cu", ".cuh")):
                        files.append(os.path.join(root, filename))

# now make a func that takes a file and returns true if all lines are less than 160
def check_file_length(file_path):
        with open(file_path, "r") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                        if len(line) > 160:
                                print(f"Line {i+1} in {file_path} is too long ({len(line)} characters)")
                                return False
        return True

# now make a func that checks if "//" comment is used; we do not want this
def check_file_comments(file_path):
        with open(file_path, "r") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                        if "//" in line:
                                print(f"Line {i+1} in {file_path} uses '//' comment")
                                return False
        return True

# now run all checks on all files
all_passed = True
for file in files:
        if not check_file_length(file):
                all_passed = False
        if not check_file_comments(file):
                all_passed = False

if all_passed:
        print("All style checks passed!")
else:
        print("Some style checks failed.")
        exit(1)