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
fMRI preprocessing pipeline using FSL, SPM, JIP, and ANTS.

This module implements a comprehensive fMRI preprocessing pipeline that includes:
1. Slice timing correction
2. Motion correction
3. Bias field correction
4. Skull stripping
5. Spatial normalization
6. Registration to template
7. Quality control reporting

Each processing step is implemented with caching support for efficient reprocessing.
"""

# %% System import
from __future__ import print_function
import os

import shutil
from datetime import datetime

# Module import
import pypreclin
# from pypreclin.utils.reorient import guess_orientation
from pypreclin.preproc.register import jip_align
from pypreclin.preproc.register import apply_jip_align, ants_linear_register_command
from pypreclin.plotting.check_preprocessing import plot_fsl_motion_parameters
from pypreclin.utils.export import gzip_file, ungzip_file

# PyConnectome import
from pyconnectome.plotting.slicer import triplanar

# PyConnectomist import
from pyconnectomist.utils.pdftools import generate_pdf

# Third party import
from joblib import Memory as JoblibMemory
# from nipype.interfaces import fsl
# from nipype.caching import Memory as NipypeMemory
import subprocess
from typing import Dict, Optional, Union
import logging
from pathlib import Path

# %% Global parameters
PROCESSING_STEPS = {
    "slice_timing": "1-slice_timing",
    "motion_correction": "2-motion_correction",
    "bias_correction": "3-bias_correction",
    "skull_stripping": "4-skull_stripping",
    "linear_registration": "5-pre_jip_linear_registration",
    "jip_registration_tmean": "6-jip_registration_tmean",
    "jip_registration_all": "7-jip_registration_all",
    "snaps": "8-snaps",
    "report": "9-report"
}

# %% Custom exception class for preprocessing errors
class PreprocessingError(Exception):
    """Custom exception for preprocessing pipeline errors."""
    def __init__(self, message: str, logger: Optional[logging.Logger] = None):
        super().__init__(message)
        if logger:
            logger.error(message)

# %%
def setup_logging(log_dir: Union[str, Path]) -> logging.Logger:
    """Configure logging for the preprocessing pipeline.
    
    Args:
        log_dir: Directory to store log files
        
    Returns:
        Configured logger instance
    """
    # Ensure we have an absolute path and create directory if needed
    log_dir = Path(log_dir).absolute()
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup log file path
    log_file = log_dir / "preproc.log"
    
    # Get or create logger
    logger = logging.getLogger("preproc")
    
    # Remove existing handlers to avoid duplicate logging
    logger.handlers.clear()
    
    # Set logging level
    logger.setLevel(logging.INFO)
    
    # Create formatter with better date format
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(str(log_file), mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Log initialization
    logger.info(f"Logging initialized: {log_file}")
    
    return logger

# %%
def check_environment_variables(skull_stripping: bool) -> None:
    """Check if all required environment variables are set.
    
    Args:
        skull_stripping: Whether skull stripping is enabled
        
    Raises:
        ValueError: If any required environment variable is not set
    """
    # Define required environment variables
    required_vars = {
        "FSLDIR": "FSL installation directory",
        "FREESURFER_HOME": "FreeSurfer installation directory",
        "ANTS_HOME": "ANTs installation directory",
        "JIP_HOME": "JIP installation directory. e.g. /usr/local/jip-Linux-x86_64"
    }
    
    # Add skull stripping variable if needed
    if skull_stripping:
        required_vars["MACAQUE_SS_UNET"] = "Macaque skull stripping UNet model directory"
        
    # Check each required variable
    missing_vars = []
    not_in_path = []
    for var, description in required_vars.items():
        if var not in os.environ:
            missing_vars.append(f"{var} ({description})")

        if var == 'MACAQUE_SS_UNET': continue
        if not os.path.isdir(os.path.join(os.environ[var], "bin")):
            not_in_path.append(f"{var} ({description})")
            
    # Raise error if any variables are missing
    if missing_vars:
        error_msg = "The following required environment variables are not set:\n"
        error_msg += "\n".join(f"- {var}" for var in missing_vars)
        raise ValueError(error_msg)
    
    # Raise error if any variables are not in the PATH
    if not_in_path:
        error_msg = "The following required environment variables are not in the PATH:\n"
        error_msg += "\n".join(f"- {var}" for var in not_in_path)
        raise ValueError(error_msg)

# %%
def preproc(
        funcfile: str,
        sid: str,
        outdir: str,
        repetitiontime: float,
        template: str,
        erase: Union[bool, str],
        resample: Union[bool, str],
        interleaved: Union[bool, str],
        sliceorder: str,
        realign_dof: int,
        skull_stripping: Union[bool, str],
        recon1: Union[bool, str],
        recon2: Union[bool, str],
        auto: Union[bool, str],
        verbose: int,
        anatfile: Optional[str] = None,
        followdir: Optional[str] = None) -> Dict:
    """Execute fMRI preprocessing pipeline.

    Args:
        funcfile: Path to functional MRI data file
        sid: Subject ID
        outdir: Output directory path
        repetitiontime: TR in seconds
        template: Path to template image
        erase: Whether to erase existing output directory
        resample: Whether to resample template
        interleaved: Whether acquisition was interleaved
        sliceorder: Slice acquisition order ('ascending' or 'descending')
        realign_dof: Degrees of freedom for motion correction
        skull_stripping: Whether to perform skull stripping
        recon1: Early stop after first reconstruction
        recon2: Early stop after second reconstruction
        auto: Whether to use automatic registration
        verbose: Verbosity level
        anatfile: Optional path to anatomical image
        followdir: Optional directory to follow for processing

    Returns:
        Dictionary containing paths to processed files

    Raises:
        ValueError: If required environment variables are not set
        ValueError: If slice order is invalid
    """

    # ------------------------------------------------------------
    # check input parameters
    if not isinstance(erase, bool):
        erase = eval(erase)
        resample = eval(resample)
        interleaved = eval(interleaved)
        recon1 = eval(recon1)
        recon2 = eval(recon2)
        skull_stripping = eval(skull_stripping)
        auto = eval(auto)

    # Read input parameters
    funcfile = os.path.abspath(funcfile)
    if anatfile is not None:
        anatfile = os.path.abspath(anatfile)
    outdir = os.path.abspath(outdir)
    if followdir is not None:
        followdir = os.path.abspath(followdir)
    template = os.path.abspath(template)

    subjdir = os.path.join(outdir, sid)
    cachedir = os.path.join(subjdir, "cachedir")

    # convert input parameters to a dict
    params = {
        "funcfile": funcfile,
        "anatfile": anatfile,
        "sid": sid,
        "outdir": outdir,
        "repetitiontime": repetitiontime,
        "template": template,
        "erase": erase,
        "resample": resample,
        "interleaved": interleaved,
        "sliceorder": sliceorder,
        "realign_dof": realign_dof,
        "skull_stripping": skull_stripping,
        "recon1": recon1,
        "recon2": recon2,
        "auto": auto,
        "verbose": verbose,
        "followdir": followdir,
        "subjdir": subjdir,
        "cachedir": cachedir,
    }

    # ------------------------------------------------------------
    # erase existing directory
    if erase and os.path.isdir(subjdir):
        shutil.rmtree(subjdir)

    # ------------------------------------------------------------
    # create a log file
    logger = setup_logging(subjdir)
    logger.info(f"Parameters: {params}")

    # ------------------------------------------------------------
    # Check required environment variables and parameters
    check_environment_variables(params["skull_stripping"])
    logger.info("All required environment variables are set.")

    # Validate slice order
    if params['sliceorder'] not in ('ascending', 'descending'):
        raise PreprocessingError(
            f"Slice order '{params['sliceorder']}' is not supported. "
            f"Supported slice orders are: ascending & descending.",
            logger
        )

    # ------------------------------------------------------------
    # create a directory for the outputs and the cache
    if not os.path.isdir(cachedir):
        os.makedirs(cachedir)
        logger.info(f"Created cache directory: {cachedir}")
    
    joblib_memory = JoblibMemory(cachedir, verbose=params["verbose"])

    # Log pipeline start with clear visual separation
    logger.info("=" * 50)
    logger.info("Starting monkey fMRI preprocessing pipeline!")

    # initialize variables and update them as the pipeline progresses
    funcf_tmean = None
    funcf_all = params["funcfile"]
    templatef = params["template"]

    # ------------------------------------------------------------
    # Slice timing correction
    logger.info("-" * 50)
    logger.info(f"Step: {PROCESSING_STEPS['slice_timing']}")

    dir_slice_timing = os.path.join(subjdir, PROCESSING_STEPS["slice_timing"])
    if not os.path.isdir(dir_slice_timing):
        os.makedirs(dir_slice_timing)
        logger.info(f"Created directory: {dir_slice_timing}")

    funcf_slice_time_corrected = os.path.join(
        dir_slice_timing, os.path.basename(funcf_all).split(".nii")[0] + ".nii.gz")

    # Run slice timing correction using FSL's slicetimer
    command_slicetimer = (
        f'slicetimer -i {funcf_all} '
        f'-o {funcf_slice_time_corrected} '
        f'-r {params["repetitiontime"]} '
        f'-d 3 '  # slice direction (3 = z)
    )
    if params["interleaved"]:
        command_slicetimer += ' --odd'
    if params["sliceorder"] == "descending":
        command_slicetimer += ' --down'
    if params["verbose"] == 2:
        command_slicetimer += ' -v'
    
    try:
        subprocess.run(command_slicetimer, shell=True, check=True)
        logger.info("Slice timing correction completed successfully")
    except subprocess.CalledProcessError as e:
        raise PreprocessingError(f"Slice timing correction failed: {str(e)}", logger)
    
    # update the variables
    funcf_all = funcf_slice_time_corrected

    # ------------------------------------------------------------
    # Motion correction
    logger.info("-" * 50)
    logger.info(f"Step: {PROCESSING_STEPS['motion_correction']}")

    dir_motion_correction = os.path.join(subjdir, PROCESSING_STEPS["motion_correction"])
    if not os.path.isdir(dir_motion_correction):
        os.makedirs(dir_motion_correction)

    funcf_motion_corrected = os.path.join(dir_motion_correction, os.path.basename(funcf_all))
    
    # prepare the reference volume for motion correction
    funcf_mc_ref = os.path.join(dir_motion_correction, 'func_tmean.nii.gz')
    if params["followdir"] is None:
        # get the tmean for ref_vol 
        command_mc_ref = f'fslmaths {funcf_all} -Tmean {funcf_mc_ref}'
        try:
            subprocess.run(command_mc_ref, shell=True, check=True)
            logger.info(f"Created reference volume using Tmean: {funcf_mc_ref}")
        except subprocess.CalledProcessError as e:
            raise PreprocessingError(f"Failed to create reference volume: {str(e)}", logger)
    else:
        # if followdir is not None, use the alignment parameters from the followdir
        funcf_mc_ref = os.path.join(params["followdir"], PROCESSING_STEPS["motion_correction"], 'func_tmean.nii.gz')
        if not os.path.isfile(funcf_mc_ref):
            raise PreprocessingError(f"Reference volume not found in followdir: {funcf_mc_ref}", logger)
        logger.info(f"Using reference volume from followdir: {funcf_mc_ref}")
    
    command_mcflirt = (
        f'mcflirt -in {funcf_all} '
        f'-out {funcf_motion_corrected.split(".nii")[0]} '
        f'-r {funcf_mc_ref} '
        f'-cost normcorr '
        f'-dof {realign_dof} '
        f'-mats '
        f'-plots'
    )
    if params["verbose"] > 0:
        command_mcflirt += ' -verbose 1'

    try:
        subprocess.run(command_mcflirt, shell=True, check=True)
        logger.info("Motion correction completed successfully")
    except subprocess.CalledProcessError as e:
        raise PreprocessingError(f"Motion correction failed: {str(e)}", logger)
    
    # generate func tmean motion corrected image
    funcf_tmean_motion_corrected = os.path.join(dir_motion_correction, 'func_tmean_motion_corrected.nii.gz')
    command_mean = f'fslmaths {funcf_motion_corrected} -Tmean {funcf_tmean_motion_corrected}'
    try:
        subprocess.run(command_mean, shell=True, check=True)
        logger.info(f"Created motion corrected func Tmean image: {funcf_tmean_motion_corrected}")
    except subprocess.CalledProcessError as e:
        raise PreprocessingError(f"Failed to create motion corrected func Tmean image: {str(e)}", logger)
    
    # update the variables
    funcf_tmean = funcf_tmean_motion_corrected
    funcf_all = funcf_motion_corrected
    
    # ------------------------------------------------------------
    # skull stripping and only keep the cortex and subcortical structures other than cerebellum and brainstem
    if params["skull_stripping"]:
        logger.info("-" * 50)
        logger.info(f"Step: {PROCESSING_STEPS['skull_stripping']}")

        dir_skull_stripping = os.path.join(subjdir, PROCESSING_STEPS["skull_stripping"])
        if not os.path.isdir(dir_skull_stripping):
            os.makedirs(dir_skull_stripping) 
            logger.info(f"Created directory: {dir_skull_stripping}")

        # anat model: models/Site-All-T-epoch_36_update_with_Site_6_plus_7-epoch_39.model
        # func model: models/epi_retrained_model-20-epoch.model

        funcf_brain_mask = os.path.join(dir_skull_stripping, 'func_tmean_brainMask.nii.gz')
        if params["followdir"] is None:
            # run unet on tmean image to get the brain mask
            command_func_ss = (
                f'python {os.environ["MACAQUE_SS_UNET"]}/muSkullStrip.py '
                f'-in {funcf_tmean} '
                f'-out {dir_skull_stripping} '
                f'-suffix brainMask ' 
                f'-model {os.path.join(os.environ["MACAQUE_SS_UNET"], "models", "epi_retrained_model-20-epoch.model")} '
            )   
            try:
                subprocess.run(command_func_ss, shell=True, check=True)
                logger.info("Unet brain mask completed successfully")
            except subprocess.CalledProcessError as e:
                raise PreprocessingError(f"Unet brain mask failed: {str(e)}", logger)

            # rename the brain mask file
            funcf_brain_mask_temp = os.path.join(dir_skull_stripping, os.path.basename(funcf_tmean).split('.nii')[0] + '_brainMask.nii.gz')
            shutil.move(funcf_brain_mask_temp, funcf_brain_mask)
            logger.info(f"Created brain mask: {funcf_brain_mask}")

        else:
            # if followdir is not None, use the brain mask from the followdir
            funcf_brain_mask = os.path.join(params["followdir"], PROCESSING_STEPS["skull_stripping"], 'func_tmean_brainMask.nii.gz')
            if not os.path.isfile(funcf_brain_mask):
                raise PreprocessingError(f"The file to be followed {funcf_brain_mask} does not exist.", logger)
            logger.info(f"Set brain mask from followdir: {funcf_brain_mask}")

        # apply the mask to the tmean image
        funcf_tmean_skullstripped = os.path.join(dir_skull_stripping, os.path.basename(funcf_tmean).split('.')[0] + '_brain.nii.gz')
        command_apply_mask = f'fslmaths {funcf_tmean} -mul {funcf_brain_mask} {funcf_tmean_skullstripped}'
        try:
            subprocess.run(command_apply_mask, shell=True, check=True)
            logger.info("Brain mask applied successfully on func Tmean")
        except subprocess.CalledProcessError as e:
            raise PreprocessingError(f"Brain mask application failed on func Tmean: {str(e)}", logger)
        
        # apply the mask to the functional image
        funcf_skullstripped = os.path.join(dir_skull_stripping, os.path.basename(funcf_motion_corrected))
        command_apply_mask = f'fslmaths {funcf_motion_corrected} -mul {funcf_brain_mask} {funcf_skullstripped}'
        try:
            subprocess.run(command_apply_mask, shell=True, check=True)
            logger.info("Brain mask applied successfully")
        except subprocess.CalledProcessError as e:
            raise PreprocessingError(f"Brain mask application failed: {str(e)}", logger)

        # update the variables
        funcf_tmean = funcf_tmean_skullstripped
        funcf_all = funcf_skullstripped

    # ------------------------------------------------------------
    # bias correction   
    logger.info("-" * 50)
    logger.info(f"Step: {PROCESSING_STEPS['bias_correction']}")
    logger.info('# Bias correction is only applied to the func tmean image to aid the skull '
                'stripping and registration. It will not be applied to the functional image.')

    dir_bias_correction = os.path.join(subjdir, PROCESSING_STEPS["bias_correction"])
    if not os.path.isdir(dir_bias_correction):
        os.makedirs(dir_bias_correction)
        logger.info(f"Created directory: {dir_bias_correction}")

    # Get biasField from the func tmean image
    funcf_tmean_bias_corrected = os.path.join(dir_bias_correction, 'func_tmean_biasCorrected.nii.gz')
    funcf_bias_field = funcf_tmean_bias_corrected.replace('_biasCorrected.nii.gz', '_biasField.nii.gz')

    # run bias correction
    command_bias_correction = (
        f'N4BiasFieldCorrection -d 3 '
        f'-i {funcf_tmean} '
        f'-o [ {funcf_tmean_bias_corrected}, {funcf_bias_field} ] '
        f'-s 2 '
        f'-b [ 40 ]'
    )
    try:
        subprocess.run(command_bias_correction, shell=True, check=True)
        logger.info("Bias correction completed successfully for func Tmean")
    except subprocess.CalledProcessError as e:
        raise PreprocessingError(f"Bias correction failed for func Tmean: {str(e)}", logger)
    
    # update the variables
    funcf_tmean = funcf_tmean_bias_corrected

    # ------------------------------------------------------------
    # linear registration to template to prepare for jip registration
    logger.info("-" * 50)
    logger.info(f"Step: {PROCESSING_STEPS['linear_registration']}")

    dir_linear_reg = os.path.join(subjdir, PROCESSING_STEPS["linear_registration"])
    if not os.path.isdir(dir_linear_reg):
        os.makedirs(dir_linear_reg)

    if params["resample"]:
        templatef_resampled = os.path.join(dir_linear_reg, os.path.basename(templatef).split('.nii')[0] + '_res-1.nii.gz')
        
        if params["followdir"] is None:
            command_resample = (
                f'flirt -in {templatef} '
                f'-ref {templatef} '
                f'-out {templatef_resampled} '
                f'-applyisoxfm 1 '
                f'-interp trilinear'
            )
            try:
                subprocess.run(command_resample, shell=True, check=True)
                logger.info(f"Template resampled to 1mm resolution: {templatef_resampled}")
            except subprocess.CalledProcessError as e:
                raise PreprocessingError(f"Template resampling failed: {str(e)}", logger)
        else:
            templatef_resampled = os.path.join(params["followdir"], PROCESSING_STEPS["linear_registration"], os.path.basename(templatef_resampled))
            logger.info(f"Using resampled template from followdir: {templatef_resampled}")
        
        # update the variables
        templatef = templatef_resampled
    
    # 1. Registrate the mean image with ANTs
    funcf_linear_reg_prefix = os.path.join(dir_linear_reg, 'func_tmean_realign')
    funcf_tmean_linear_reg = funcf_linear_reg_prefix + '.nii.gz'

    if params["followdir"] is None:
        command_register = ants_linear_register_command(funcf_tmean, templatef, 
                            [funcf_linear_reg_prefix + '_', funcf_tmean_linear_reg], 
                            affine=True)
        try:
            subprocess.run(command_register, shell=True, check=True)
            logger.info("Registration completed successfully for func Tmean")
        except subprocess.CalledProcessError as e:
            raise PreprocessingError(f"Registration failed for func Tmean: {str(e)}", logger)
        transform_mat = f'{funcf_linear_reg_prefix}_0GenericAffine.mat'
    else:
        transform_mat = os.path.join(params["followdir"], PROCESSING_STEPS["linear_registration"], 
            f'{os.path.basename(funcf_linear_reg_prefix)}_0GenericAffine.mat')
        if not os.path.isfile(transform_mat):
            raise PreprocessingError(f"The file to be followed {transform_mat} does not exist.", logger)
        logger.info(f"Set transform matrix from followdir: {transform_mat}")

    # 2. Apply the transformation to the functional image
    funcf_linear_reg = os.path.join(dir_linear_reg, os.path.basename(funcf_all))
    command_apply_reg = (
        f'antsApplyTransforms -d 3 -e 3 '
        f'-i {funcf_all} '
        f'-r {templatef} '
        f'-o {funcf_linear_reg} '
        f'-t {transform_mat} '
        f'--float --interpolation Linear'
    )
    try:
        subprocess.run(command_apply_reg, shell=True, check=True)
        logger.info("Registration completed successfully")
    except subprocess.CalledProcessError as e:
        raise PreprocessingError(f"Registration failed: {str(e)}", logger)
    
    # update the variables
    funcf_tmean = funcf_tmean_linear_reg
    funcf_all = funcf_linear_reg

    # ------------------------------------------------------------
    # registration tmean using JIP
    if params["followdir"] is None:
        logger.info("-" * 50)
        logger.info(f"Step: {PROCESSING_STEPS['jip_registration_tmean']}")
        
        # registration mean using JIP
        dir_jip_reg_tmean = os.path.join(subjdir, PROCESSING_STEPS["jip_registration_tmean"])
        if not os.path.isdir(dir_jip_reg_tmean):
            os.makedirs(dir_jip_reg_tmean)
            logger.info(f"Created directory: {dir_jip_reg_tmean}")

        # run jip registration
        interface = joblib_memory.cache(jip_align)
        try:
            (_, _, _, align_coregfile) = interface(
                source_file=funcf_tmean,
                target_file=templatef,
                outdir=dir_jip_reg_tmean,
                postfix='_space-template',
                auto=params["auto"],
                non_linear=True)
            logger.info("JIP registration completed successfully for func Tmean")
        except subprocess.CalledProcessError as e:
            raise PreprocessingError(f"JIP registration failed for func Tmean: {str(e)}", logger)

        # remove the .nii files if .nii.gz files exist in the registration mean directory
        for file in os.listdir(dir_jip_reg_tmean):
            if file.endswith('.nii'):
                if file == 'distortions.nii':
                    continue
                if os.path.exists(file.replace('.nii', '.nii.gz')):
                    os.remove(os.path.join(dir_jip_reg_tmean, file))
                else:
                    gzip_file(os.path.join(dir_jip_reg_tmean, file), prefix="", outdir=dir_jip_reg_tmean, remove_original_file=True)

    else:
        align_coregfile = os.path.join(params["followdir"], PROCESSING_STEPS["jip_registration_tmean"], 'align.com')
        logger.info(f"Set align coreg file from followdir: {align_coregfile}")
    
    # ------------------------------------------------------------
    # registration all applying xfm from registration mean
    logger.info("-" * 50)
    logger.info(f"Step: {PROCESSING_STEPS['jip_registration_all']}")
    
    dir_jip_reg_all = os.path.join(subjdir, PROCESSING_STEPS["jip_registration_all"])
    if not os.path.isdir(dir_jip_reg_all):
        os.makedirs(dir_jip_reg_all)
        logger.info(f"Created directory: {dir_jip_reg_all}")

    # apply jip registration
    try:
        interface = joblib_memory.cache(apply_jip_align)
        _ = interface(
            apply_to_files=[funcf_all],
            align_with=[align_coregfile],
            outdir=dir_jip_reg_all,
            postfix='_space-template',          
            apply_inv=False)
        logger.info("JIP registration completed successfully")
    except subprocess.CalledProcessError as e:
        raise PreprocessingError(f"JIP registration failed: {str(e)}", logger)
    
    # generate final func tmean image for reporting
    funcf_reg_all = os.path.join(dir_jip_reg_all, os.path.basename(funcf_all).split('.nii')[0] + '_space-template.nii.gz')
    funcf_tmean_reg = os.path.join(dir_jip_reg_all, 'func_tmean_reg.nii.gz')
    command_get_tmean = f'fslmaths {funcf_reg_all} -Tmean {funcf_tmean_reg}'
    try:
        subprocess.run(command_get_tmean, shell=True, check=True)
        logger.info("Final Tmean image created successfully")
    except subprocess.CalledProcessError as e:
        raise PreprocessingError(f"Final Tmean image creation failed: {str(e)}", logger)
    
    # update the variables
    funcf_tmean = funcf_tmean_reg
    funcf_all = funcf_reg_all
    
    # ------------------------------------------------------------
    logger.info("Preprocessing completed successfully")
    logger.info("=" * 50)
    logger.info(f"Final output files: "
                f"func data: {funcf_all}, \n"
                f"func tmean: {funcf_tmean}")

    # ------------------------------------------------------------
    # Compute some snaps assessing the different processing steps.
    logger.info("-" * 50)
    logger.info(f"Step: {PROCESSING_STEPS['snaps']}")

    dir_snap = os.path.join(subjdir, PROCESSING_STEPS["snaps"])
    if not os.path.isdir(dir_snap):
        os.mkdir(dir_snap)
    interface = joblib_memory.cache(triplanar)

    # > generate coregistration plot
    coregister_fileroot = os.path.join(dir_snap, "coregister")
    interface(
        input_file=funcf_tmean,
        output_fileroot=coregister_fileroot,
        overlays=[templatef],
        overlays_colors=None,
        contours=True,
        edges=False,
        overlay_opacities=[0.7],
        resolution=300)

    # > generate a motion parameter plot
    interface = joblib_memory.cache(plot_fsl_motion_parameters)
    realign_motion_file = os.path.join(dir_snap, "realign_motion_parameters.png")
    interface(funcf_motion_corrected.split(".")[0] + ".par", realign_motion_file)

    # Generate a QC reporting
    reportfile = os.path.join(subjdir, "QC_report.pdf")

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
