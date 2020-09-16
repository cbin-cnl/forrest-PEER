#!/bin/bash
DATA=$1
for subj in "$DATA"sub*/; do 
  echo $subj
  singularity run -B $subj:$subj min_preproc.simg "$subj"output $(find $subj/ses-movie/anat/ -name "head.nii.gz") $(find $subj/ses-movie/func/  -name "*bold.nii.gz");
done

