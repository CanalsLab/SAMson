# Data bundled with samson-mri

* `anat_template_FP.nii.gz`: SAMson's template (Mouse-X template, `anat_template_FP.nii`), losslessly compressed. 40
  coronal slices, 0.4 mm apart, slice 0 = caudal end. The header stores voxel sizes ×10.
* `example/`: one mouse of the article's dataset D1 (3 months old, subject A110123): T2w image
  (`T2Wmice_first_im_A110123_3mo_.nii`, losslessly compressed), its manual brain mask
  (`T2Wmice_mask_A110123_3mo_.nii`), and the Full-Auto and Semi-Auto SAMson masks published with the article
  (`SAMson_masks/{full_auto,semi_auto}/d1/3_months/`).

Source: SAMson's public data, https://doi.org/10.20350/digitalCSIC/17000 (Panadero Soler et al.). Please cite the
article (see `CITATION.cff`) when you use these data.
