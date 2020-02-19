#!/bin/bash
DATA=$1
for subj in "$DATA"sub*/; do 
  echo $subj
  docker run -v $subj:$subj rzlim08/minimal_preprocessing "$subj"output $(find $subj/ses-forrestgump/anat/ -name "*T1w.nii.gz") $(find $subj/ses-movie/func/ -name "*bold.nii.gz")

done

