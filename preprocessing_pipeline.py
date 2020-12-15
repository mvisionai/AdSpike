from nipype.interfaces import fsl
from nipype.testing import example_data
import os
flt = fsl.FLIRT(bins=640, cost_func='mutualinfo')
flt.inputs.in_file = "MNI152_T1_1mm_Brain.nii.gz"
flt.inputs.reference = 'MNI152_T1_1mm_Brain.nii.gz'
flt.inputs.output_type = "NIFTI"
print(flt.cmdline)
res = flt.run()