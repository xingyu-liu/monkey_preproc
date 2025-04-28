# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# %%
"""
fMRI preprocessings using FSL, SPM, JIP, and ANTS.

Steps
-----

1. Slice Timing correction (with cache).
2. Reorient images not in RAS coordinate system and reorient images to match
   the orientation of the standard MNI152 template (with cache).
3. motion correction: adjust for movement between slices (with
   cache).
4. bias correction: correct for bias field (with cache).
5. registration: warp images to fit to a standard template brain (with cache).
6. Smooth the functional time serie (with cache).
7. SNAPs: compute some snaps assessing the different processing steps (with
    cache).
8. Reporting: generate a QC reporting (with cache).
""" 

# %% System import
from __future__ import print_function
import os

import shutil
from datetime import datetime

# Module import
import pypreclin
from pypreclin.utils.reorient import guess_orientation
from pypreclin.preproc.register import jip_align
from pypreclin.preproc.register import apply_jip_align
from pypreclin.plotting.check_preprocessing import plot_fsl_motion_parameters
from pypreclin.utils.export import gzip_file, ungzip_file

# PyConnectome import
from pyconnectome.plotting.slicer import triplanar

# PyConnectomist import
from pyconnectomist.utils.pdftools import generate_pdf

# Third party import
from joblib import Memory as JoblibMemory
from nipype.interfaces import fsl
from nipype.caching import Memory as NipypeMemory
import subprocess

# %% Global parameters
STEPS = {
    "slice_timing": "1-SliceTiming",
    "motion_correction" : "2-MotionCorrection",
    "bias_correction": "3-BiasCorrection",
    "skull_stripping": "4-SkullStripping",
    "reorient": "5-Reorient",
    "registration_mean": "6-Registration_mean",
    "registration_all": "7-Registration_all",
    "smooth": "8-Smooth",
    "snaps": "9-Snaps",
    "report": "10-Report"
}

def preproc(
        funcfile,
        anatfile,
        sid,
        outdir,
        repetitiontime,
        template,
        erase,
        resample,
        interleaved,
        sliceorder,
        realign_dof,
        realign_to_vol,
        skull_stripping,
        warp,
        warp_njobs,
        warp_index,
        warp_file,
        warp_restrict,
        delta_te,
        dwell_time,
        manufacturer,
        blip_files,
        blip_enc_dirs,
        unwarp_direction,
        phase_file,
        magnitude_file,
        anatorient,
        funcorient,
        kernel_size,
        fslconfig,
        normalization_trf,
        coregistration_trf,
        recon1,
        recon2,
        auto,
        verbose):
    """ fMRI preprocessings using FSL, SPM, JIP, and ANTS.
    """

    # TODO: remove when all controls available in pypipe
    if not isinstance(erase, bool):
        erase = eval(erase)
        resample = eval(resample)
        interleaved = eval(interleaved)
        realign_to_vol = eval(realign_to_vol)
        warp = eval(warp)
        recon1 = eval(recon1)
        recon2 = eval(recon2)
        skull_stripping = eval(skull_stripping)
        auto = eval(auto)
        warp_restrict = eval(warp_restrict)
        blip_files = None if blip_files == "" else eval(blip_files)
        blip_enc_dirs = eval(blip_enc_dirs)

    # Read input parameters
    funcfile = os.path.abspath(funcfile)
    if anatfile is not None:
        anatfile = os.path.abspath(anatfile)
    template = os.path.abspath(template)
    realign_to_mean = not realign_to_vol
    subjdir = os.path.join(os.path.abspath(outdir), sid)
    cachedir = os.path.join(subjdir, "cachedir")
    outputs = {}
    if erase and os.path.isdir(subjdir):
        shutil.rmtree(subjdir)
    if not os.path.isdir(cachedir):
        os.makedirs(cachedir)
    nipype_memory = NipypeMemory(cachedir)
    joblib_memory = JoblibMemory(cachedir, verbose=verbose)

    # ------------------------------------------------------------
    # check all path needed is in the env
    # FSLDIR, FREESURFER_HOME, ANTS_HOME, JIP_HOME
    if "FSLDIR" not in os.environ:
        raise ValueError("FSLDIR is not set in the environment.")
    if "FREESURFER_HOME" not in os.environ:
        raise ValueError("FREESURFER_HOME is not set in the environment.")
    if "ANTS_HOME" not in os.environ:
        raise ValueError("ANTS_HOME is not set in the environment.")
    if "JIP_HOME" not in os.environ:
        raise ValueError("JIP_HOME is not set in the environment.")
    if skull_stripping:
        if "MACAQUE_SS_UNET" not in os.environ:
            raise ValueError("MACAQUE_SS_UNET is not set in the environment.")

    # ------------------------------------------------------------
    def display_outputs(outputs, verbose, **kwargs):
        """ Simple function to display/store step outputs.
        """
        if verbose > 0:
            print("-" * 50)
            for key, val in kwargs.items():
                print("{0}: {1}".format(key, val))
            print("-" * 50)
        outputs.update(kwargs)

    # Check input parameters
    template_axes = guess_orientation(template)
    if template_axes != "RAS":
        raise ValueError("The template orientation must be 'RAS', '{0}' "
                         "found.".format(template_axes))
    
    if sliceorder not in ("ascending", "descending"):
        raise ValueError("Supported slice order are: ascending & descending.")

    # ------------------------------------------------------------
    # Slice timing    
    dir_slice_timing = os.path.join(subjdir, STEPS["slice_timing"])
    if not os.path.isdir(dir_slice_timing):
        os.mkdir(dir_slice_timing)
    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
    interface = nipype_memory.cache(fsl.SliceTimer)

    returncode = interface(
        in_file=funcfile,
        interleaved=interleaved,
        slice_direction=3,
        time_repetition=repetitiontime,
        index_dir=False if sliceorder=="ascending" else True,
        out_file=os.path.join(
            dir_slice_timing, os.path.basename(funcfile).split(".")[0] + ".nii.gz"))
    st_outputs = returncode.outputs.get()
    funcf_slice_time_corrected = st_outputs["slice_time_corrected_file"]
    display_outputs(
        outputs, verbose, slice_time_corrected_file=funcf_slice_time_corrected)

    # ------------------------------------------------------------
    # motion correction
    dir_motion_correction = os.path.join(subjdir, STEPS["motion_correction"])
    if not os.path.isdir(dir_motion_correction):
        os.mkdir(dir_motion_correction)

    funcf_motion_corrected = os.path.join(dir_motion_correction, 
                                               os.path.basename(funcf_slice_time_corrected))
    
    command_mcflirt = (
        f'mcflirt -in {funcf_slice_time_corrected} '
        f'-out {funcf_motion_corrected.split(".")[0]} '
        f'-cost normcorr '
        f'-dof {realign_dof} '
        f'-mats '
        f'-plots'
    )
    if verbose > 0:
        command_mcflirt = command_mcflirt + ' -verbose 1'
    # print(f"Running command: {command_mcflirt}")
    _ = subprocess.run(command_mcflirt, shell=True, check=True)

    display_outputs(
        outputs, verbose, motion_correction_funcfile=funcf_motion_corrected)
    
    # ------------------------------------------------------------
    # bias correction   
    dir_bias_correction = os.path.join(subjdir, STEPS["bias_correction"])
    if not os.path.isdir(dir_bias_correction):
        os.mkdir(dir_bias_correction)

    # step 1: get biasField from the mean functional image
    funcf_tmean = os.path.join(dir_bias_correction, 'func_tmean.nii.gz')
    funcf_tmean_bias_corrected = funcf_tmean.replace('.nii.gz', '_biasCorrected.nii.gz')
    funcf_bias_field = funcf_tmean.replace('.nii.gz', '_biasField.nii.gz')
    
    # get func_tmean_file
    command_mean = f'fslmaths {funcf_motion_corrected} -Tmean {funcf_tmean}'
    # print(f"Running command: {command_mean}")
    _ = subprocess.run(command_mean, shell=True, check=True)

    # run bias correction
    command_bias_correction = (
        f'N4BiasFieldCorrection -d 3 '
        f'-i {funcf_tmean} '
        f'-o [ {funcf_tmean_bias_corrected}, {funcf_bias_field} ] '
        f'-s 2 '
        f'-b 100'
    )
    # print(f"Running command: {command_bias_correction}")
    _ = subprocess.run(command_bias_correction, shell=True, check=True)

    # step 2: apply bias correction to the functional image by dividing the bias field
    funcf_bias_corrected = os.path.join(dir_bias_correction, 
                                        os.path.basename(funcf_motion_corrected))

    command_ants_apply_transforms = f'fslmaths {funcf_motion_corrected} -div {funcf_bias_field} {funcf_bias_corrected}'   
    # print(f"Running command: {command_ants_apply_transforms}")
    _ = subprocess.run(command_ants_apply_transforms, shell=True, check=True)

    # apply bias correction
    display_outputs(
        outputs, verbose, bias_corrected_funcfile=funcf_bias_corrected)

    # ------------------------------------------------------------
    # skull stripping and only keep the cortex and subcortical structures other than cerebellum and brainstem
    if skull_stripping:
        dir_skull_stripping = os.path.join(subjdir, STEPS["skull_stripping"])
        if not os.path.isdir(dir_skull_stripping):
            os.mkdir(dir_skull_stripping) 

        # anat model: models/Site-All-T-epoch_36_update_with_Site_6_plus_7-epoch_39.model
        # func model: models/epi_retrained_model-20-epoch.model

        # run for func
        command_func_ss = (
            f'python {os.environ["MACAQUE_SS_UNET"]}/muSkullStrip.py '
            f'-in {funcf_tmean_bias_corrected} '
            f'-out {dir_skull_stripping} '
            f'-suffix brainMask ' 
            f'-model {os.path.join(os.environ["MACAQUE_SS_UNET"], "models", "epi_retrained_model-20-epoch.model")} '
        )   
        # print(f"Running command: {command_func_ss}")
        _ = subprocess.run(command_func_ss, shell=True, check=True)

        # apply the mask to the functional image and the tmean image
        funcf_brain_mask = os.path.join(dir_skull_stripping, os.path.basename(funcf_tmean_bias_corrected).split('.')[0] + '_brainMask.nii.gz')
        funcf_brain = os.path.join(dir_skull_stripping, os.path.basename(funcf_bias_corrected))
        command_apply_mask = f'fslmaths {funcf_bias_corrected} -mul {funcf_brain_mask} {funcf_brain}'
        # print(f"Running command: {command_apply_mask}")
        _ = subprocess.run(command_apply_mask, shell=True, check=True)

        funcf_tmean_brain = os.path.join(dir_skull_stripping, os.path.basename(funcf_tmean_bias_corrected).split('.')[0] + '_brain.nii.gz')
        command_apply_mask = f'fslmaths {funcf_tmean_bias_corrected} -mul {funcf_brain_mask} {funcf_tmean_brain}'
        # print(f"Running command: {command_apply_mask}")
        _ = subprocess.run(command_apply_mask, shell=True, check=True)

        display_outputs(
            outputs, verbose, brain_funcfile=funcf_brain)
    else:
        funcf_brain = funcf_bias_corrected
        funcf_tmean_brain = funcf_tmean_bias_corrected

    # ------------------------------------------------------------
    # reorient - registration to template to prepare for jip registration
    dir_reorient = os.path.join(subjdir, STEPS["reorient"])
    if not os.path.isdir(dir_reorient):
        os.mkdir(dir_reorient)

    if resample:
        # resample the template to 1mm resolution 
        template_orig = template
        template = os.path.join(dir_reorient, os.path.basename(template).split('.')[0] + '_res-1.nii.gz')
        command_resample = f'flirt -in {template_orig} -ref {template_orig} -out {template} -applyisoxfm 1 -interp trilinear'
        # print(f"Running command: {command_resample}")
        _ = subprocess.run(command_resample, shell=True, check=True)
    
    # Registration the mean image with ANTs
    funcf_reorient_prefix = os.path.join(dir_reorient, 'func_realign')
    funcf_tmean_reorient = funcf_reorient_prefix + '.nii.gz'
    funcf_reoriented = os.path.join(dir_reorient, os.path.basename(funcf_brain))

    command_mean_realign = (
        f'antsRegistration --dimensionality 3 --float 0 '
        f'-o [{funcf_reorient_prefix}, {funcf_tmean_reorient}] '
        f'--interpolation Linear '
        f'--winsorize-image-intensities [0.005,0.995] '
        f'--use-histogram-matching 0 '
        f'--initial-moving-transform [{template},{funcf_tmean_brain},1] '
        f'--transform "Rigid[0.1]" '
        f'--metric "MI[{template},{funcf_tmean_brain},1,32,Regular,0.25]" '
        f'--convergence [1000x500x250x100,1e-6,10] '
        f'--shrink-factors 8x4x2x1 '
        f'--smoothing-sigmas 3x2x1x0vox '
        f'--float'
    )
    print(f"Running command: {command_mean}")
    _ = subprocess.run(command_mean_realign, shell=True, check=True)

    # # resample the functional tmean back to func resolution
    # funcf_tmean_reorient = funcf_reorient_prefix + '.nii.gz'
    # command_resample = f'3dresample -input {funcf_tmean_reorient_resAnat} -master {funcf_tmean_brain} -prefix {funcf_tmean_reorient} -rmode Cubic'
    # print(f"Running command: {command_resample}")
    # _ = subprocess.run(command_resample, shell=True, check=True)

    # Apply the transformation to the functional image
    transform_matrix = f'{funcf_reorient_prefix}0GenericAffine.mat'
    command_final = (
        f'antsApplyTransforms -d 3 -e 3 '
        f'-i {funcf_brain} '
        f'-r {template} '
        f'-o {funcf_reoriented} '
        f'-t {transform_matrix} '
        f'--float --interpolation Linear'
    )
    print(f"Running command: {command_final}")
    _ = subprocess.run(command_final, shell=True, check=True)

    # # Step 4: Convert the output to float data type
    # # IMPORTANT: the input image should be float data type for JIP registration
    # command_float = f'fslmaths {funcf_reoriented} {funcf_reoriented} -odt float'
    # # print(f"Running command: {command_float}")
    # _ = subprocess.run(command_float, shell=True, check=True)
    # # for func_tmean
    # command_float = f'fslmaths {funcf_tmean_reorient_prefix}.nii.gz {funcf_tmean_reorient_prefix}.nii.gz -odt float'
    # # print(f"Running command: {command_float}")
    # _ = subprocess.run(command_float, shell=True, check=True)

    display_outputs(
        outputs, verbose, standard_funcfile=funcf_reoriented)

    # ------------------------------------------------------------
    # registration mean using JIP
    # Early stop detected
    if recon1:
        print("[warn] User requested a processing early stop. Remove the 'recon1' "
              "option to resume.")
        return outputs
    
    # registration mean using JIP
    dir_reg_mean = os.path.join(subjdir, STEPS["registration_mean"])
    if not os.path.isdir(dir_reg_mean):
        os.mkdir(dir_reg_mean)
    if coregistration_trf is not None:
        shutil.copy(coregistration_trf, dir_reg_mean)

    # # if resample is True, resample template to the resolution of the functional image
    # if resample:
    #     template_jip = os.path.join(dir_reg_mean, os.path.basename(template).split('.')[0] + '_res-func.nii.gz')
    #     command_resample = f'3dresample -input {template} -master {funcf_tmean_reorient} -prefix {template_jip} -rmode Cubic'
    #     # print(f"Running command: {command_resample}")
    #     _ = subprocess.run(command_resample, shell=True, check=True)
    # else:
    #     template_jip = template

    # run jip registration
    interface = joblib_memory.cache(jip_align)
    (funcf_tmean_reg, funcf_tmean_reg_maskfile,
     funcf_tmean_native_maskfile, align_coregfile) = interface(
        source_file=funcf_tmean_reorient,
        target_file=template,
        outdir=dir_reg_mean,
        postfix='_space-template',
        auto=auto,
        non_linear=True)

    display_outputs(
        outputs, verbose, register_func_tmeanfile=funcf_tmean_reg,
        register_func_tmean_maskfile=funcf_tmean_reg_maskfile,
        native_func_tmean_maskfile=funcf_tmean_native_maskfile,
        align_coregfile=align_coregfile)
    
    # ------------------------------------------------------------
    # registration all applying xfm from registration mean
    # Early stop detected
    if recon2:
        print("[warn] User requested a processing early stop. Remove the 'recon2' "
              "option to resume.")
        return outputs
    
    # registration all applying xfm from registration mean
    dir_reg = os.path.join(subjdir, STEPS["registration_all"])
    if not os.path.isdir(dir_reg):
        os.mkdir(dir_reg)

    # apply jip registration
    interface = joblib_memory.cache(apply_jip_align)
    _ = interface(
        apply_to_files=[funcf_reoriented],
        align_with=[align_coregfile],
        outdir=dir_reg,
        postfix='_space-template',
        apply_inv=False)

    # remove the .nii files if .nii.gz files exist in the registration mean directory
    for file in os.listdir(dir_reg_mean):
        if file.endswith('.nii'):
            if os.path.exists(file.replace('.nii', '.nii.gz')):
                os.remove(os.path.join(dir_reg_mean, file))
            else:
                gzip_file(os.path.join(dir_reg_mean, file), prefix="", outdir=dir_reg_mean, remove_original_file=True)

    # ============================================================
    # Compute some snaps assessing the different processing steps.
    dir_snap = os.path.join(subjdir, STEPS["snaps"])
    if not os.path.isdir(dir_snap):
        os.mkdir(dir_snap)
    interface = joblib_memory.cache(triplanar)
    # > generate coregistration plot
    coregister_fileroot = os.path.join(dir_snap, "coregister")
    coregister_file = interface(
        input_file=funcf_tmean_reg,
        output_fileroot=coregister_fileroot,
        overlays=[template],
        overlays_colors=None,
        contours=True,
        edges=False,
        overlay_opacities=[0.7],
        resolution=300)

    # > generate a motion parameter plot
    interface = joblib_memory.cache(plot_fsl_motion_parameters)
    realign_motion_file = os.path.join(dir_snap, "realign_motion_parameters.png")
    interface(funcf_motion_corrected.split(".")[0] + ".par", realign_motion_file)
    display_outputs(
        outputs, verbose,
        realign_motion_file=realign_motion_file, coregister_file=coregister_file)

    # Generate a QC reporting
    reportdir = os.path.join(subjdir, STEPS["report"])
    reportfile = os.path.join(reportdir, "QC_preproc_{0}.pdf".format(sid))
    if not os.path.isdir(reportdir):
        os.mkdir(reportdir)
    interface = joblib_memory.cache(generate_pdf)
    tic = datetime.now()
    date = "{0}-{1}-{2}".format(tic.year, tic.month, tic.day)
    interface(
        datapath=dir_snap,
        struct_file=os.path.join(
            os.path.abspath(os.path.dirname(pypreclin.__file__)), "utils",
            "resources", "pypreclin_qcpreproc.json"),
        author="NeuroSpin",
        client="-",
        poweredby="FSL-SPM-Nipype-JIP",
        project="-",
        timepoint="-",
        subject=sid,
        date=date,
        title="fMRI Preprocessing QC Reporting",
        filename=reportfile,
        pagesize=None,
        left_margin=10,
        right_margin=10,
        top_margin=20,
        bottom_margin=20,
        show_boundary=False,
        verbose=0)
    display_outputs(outputs, verbose, reportfile=reportfile)

    return outputs


# def preproc_multi(
#         bidsdir, template, jipdir, fslconfig, auto, resample, anatorient="RAS",
#         funcorient="RAS", njobs=1, simage=None, shome=None, sbinds=None,
#         verbose=0):
#     """ Perform the FMRI preprocessing on a BIDS organized directory in
#     parallel (without FUGUE or TOPUP).
#     This function can be called with a singularity image that contains all the
#     required software.

#     If a 'jip_trf' directory is available in the session directory, the code
#     will use the available JIP transformation.

#     Parameters
    
#     bidsdir: str
#         the BIDS organized directory.
#     template: str
#         the path to the template in RAS coordiante system.
#     jipdir: str
#         the jip software binary path.
#     fslconfig: str
#         the FSL configuration script.
#     auto: bool
#         control the JIP window with the script.
#     resample: bool
#         if set resample the input template to fit the anatomical image.
#     anatorient: str, default "RAS"
#         the input anatomical image orientation.
#     funcorient: str, default "RAS"
#         the input functional image orientation.
#     njobs: int, default 1
#         the number of parallel jobs.
#     simage: simg, default None
#         a singularity image.
#     shome: str, default None
#         a home directory for the singularity image.
#     sbinds: str, default None
#         shared directories for the singularity image.
#     verbose: int, default 0
#         control the verbosity level.
#     """
#     # TODO: remove when all controls available in pypipe
#     if not isinstance(auto, bool):
#         auto = eval(auto)
#         resample = eval(resample)
#         sbinds = eval(sbinds)

#     # Parse the BIDS directory
#     desc_file = os.path.join(bidsdir, "dataset_description.json")
#     if not os.path.isfile(desc_file):
#         raise ValueError(
#             "Expect '{0}' in a BIDS organized directory.".format(desc_file))
#     with open(desc_file, "rt") as of:
#         desc = json.load(of)
#     name = desc["Name"]
#     if verbose > 0:
#         print("Processing dataset '{0}'...".format(name))
#     dataset = {name: {}}
#     funcfiles, anatfiles, subjects, outputs, trs, encdirs, normfiles, coregfiles = [
#         [] for _ in range(8)]
#     outdir = os.path.join(bidsdir, "derivatives", "preproc")
#     for sesdir in glob.glob(os.path.join(bidsdir, "sub-*", "ses-*")):
#         split = sesdir.split(os.sep)
#         sid, ses = split[-2:]
#         _anatfiles = glob.glob(os.path.join(sesdir, "anat", "sub-*T1w.nii.gz"))
#         _funcfiles = glob.glob(os.path.join(sesdir, "func", "sub-*bold.nii.gz"))
#         if len(_anatfiles) != 1:
#             if verbose > 0:
#                 print("Skip session '{0}': no valid anat.".format(sesdir))
#             continue
#         if len(_funcfiles) == 0:
#             if verbose > 0:
#                 print("Skip session '{0}': no valid func.".format(sesdir))
#             continue
#         descs = []
#         for path in _funcfiles:
#             runs = re.findall(".*_(run-[0-9]*)_.*", path)
#             if len(runs) != 1:
#                 if verbose > 0:
#                     print("Skip path '{0}': not valid name.".format(path))
#                 continue
#             sesdir = os.path.join(outdir, sid, ses)
#             rundir = os.path.join(sesdir, runs[0])
#             if not os.path.isdir(rundir):
#                 os.makedirs(rundir)
#             desc_file = path.replace(".nii.gz", ".json")
#             if not os.path.isfile(desc_file):
#                 break
#             with open(desc_file, "rt") as of:
#                 _desc = json.load(of)
#             # change by zyj
#             if False:
#                 if _desc["PhaseEncodingDirection"].startswith("i"):
#                     warp_restrict = [1, 0, 0]
#                 elif _desc["PhaseEncodingDirection"].startswith("j"):
#                     warp_restrict = [0, 1, 0]
#                 elif _desc["PhaseEncodingDirection"].startswith("k"):
#                     warp_restrict = [0, 0, 1]
#                 else:
#                     raise ValueError(
#                         "Unknown encode phase direction : {0}...".format(
#                             _desc["PhaseEncodingDirection"]))
#             warp_restrict = [0, 1, 0]
#             # change end
#             jip_normalization = os.path.join(
#                 sesdir, "jip_trf", "Normalization", "align.com")
#             if not os.path.isfile(jip_normalization):
#                 if verbose > 0:
#                     print("No JIP normalization align.com file found "
#                           "in {0}.".format(sesdir))
#                 jip_normalization = None
#             jip_coregistration = os.path.join(
#                 sesdir, "jip_trf", "Coregistration", "align.com")
#             if not os.path.isfile(jip_coregistration):
#                 if verbose > 0:
#                     print("No JIP coregistration align.com file found "
#                           "in {0}.".format(sesdir))
#                 jip_coregistration = None
#             desc = {
#                 "tr": _desc["RepetitionTime"],
#                 "warp_restrict": warp_restrict,
#                 "output": rundir}
#             descs.append(desc)
#             funcfiles.append(path)
#             anatfiles.append(_anatfiles[0])
#             subjects.append(sid)
#             outputs.append(rundir)
#             trs.append(_desc["RepetitionTime"])
#             encdirs.append(warp_restrict)
#             normfiles.append(jip_normalization)
#             coregfiles.append(jip_coregistration)
#         if len(_funcfiles) != len(descs):
#             if verbose > 0:
#                 print("Skip session '{0}': no valid desc.".format(sesdir))
#             continue
#         if sid not in dataset[name]:
#             dataset[name][sid] = {}
#         dataset[name][sid][ses] = {
#             "anat": _anatfiles[0],
#             "func": _funcfiles,
#             "desc": descs}
#     if verbose > 0:
#         pprint(dataset)

#     # Preprare inputs
#     expected_size = len(subjects)
#     if expected_size == 0:
#         if verbose > 0:
#             print("no data to process.")
#         return None
#     for name, item in (("outputs", outputs), ("funcfiles", funcfiles),
#                        ("anatfiles", anatfiles), ("subjects", subjects),
#                        ("trs", trs), ("encdirs", encdirs),
#                        ("normfiles", normfiles), ("coregfiles", coregfiles)):
#         if item is None:
#             continue
#         if verbose > 0:
#             print("[{0}] {1} : {2} ... {3}".format(
#                 name.upper(), len(item), item[0], item[-1]))
#         if len(item) != expected_size:
#             raise ValueError("BIDS dataset not aligned.")

#     # Run preproc
#     scriptdir = os.path.join(os.path.dirname(pypreclin.__file__), "scripts")
#     script = os.path.join(scriptdir, "pypreclin_preproc_fmri")
#     python_cmd = "python3"
#     if not os.path.isfile(script):
#         script = "pypreclin_preproc_fmri"
#         python_cmd = None
#     if simage is not None:
#         script = "singularity run --home {0} ".format(
#             shome or tempfile.gettempdir())
#         sbinds = sbinds or []
#         for path in sbinds:
#             script += "--bind {0} ".format(path)
#         script += simage
#         python_cmd = None
#     logdir = os.path.join(bidsdir, "derivatives", "logs")
#     if not os.path.isdir(logdir):
#         os.makedirs(logdir)
#     logfile = os.path.join(
#         logdir, "pypreclin_{0}.log".format(datetime.now().isoformat()))
#     status, exitcodes = hopla(
#         script,
#         hopla_iterative_kwargs=["f", "a", "s", "o", "r", "WR", "N", "M"],
#         f=funcfiles,
#         a=anatfiles,
#         s=subjects,
#         o=outputs,
#         r=trs,
#         t=template,
#         j=jipdir,
#         resample=resample,
#         W=False,
#         WN=1,
#         WR=encdirs,
#         NA=anatorient,
#         NF=funcorient,
#         C=fslconfig,
#         N=normfiles,
#         M=coregfiles,
#         A=auto,
#         hopla_python_cmd=python_cmd,
#         hopla_use_subprocess=True,
#         hopla_verbose=1,
#         hopla_cpus=njobs,
#         hopla_logfile=logfile)

#     return {"logfile": logfile}
    
