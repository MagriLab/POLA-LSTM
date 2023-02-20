#! /bin/bash
echo "Current working directory is: $(pwd)"

cd 128dof
# Find all subfolders of D10
subfolders=$(ls -l | grep "^d" | awk '{print $9}')
echo "first folders $subfolders"
# For each subfolder
for subfolder in $subfolders
do

  # Go to the subfolder
  cd $subfolder
  echo "Current working directory is: $(pwd)"
subsubfolders=$(ls -l | grep "^d" | awk '{print $9}' | grep -v "images")
  echo $subsubfolders
  for subsubfolder in $subsubfolders
  do
        cd $subsubfolder
        echo "Current working directory is: $(pwd)"
        # Go to the model folder
        cd model

        # Find the highest numbered iteration folder
        max_iteration=$(ls -l | grep "^d" | awk '{print $9}' | sort -n | tail -1)

        # Delete all iteration folders except for the highest numbered one
        for iteration in $(ls -l | grep "^d" | awk '{print $9}')
        do
          if [[ $iteration -lt $max_iteration ]]
          then
            rm -r $iteration
          fi
        done
        cd ../..
  done
  cd ..
  # Go baack to the D10 foldercd

done