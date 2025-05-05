##########################################################################
# NSAp - Copyright (C) CEA, 2013 - 2016
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Registration utilities.
"""

# System import
import os
import subprocess
import nibabel
import numpy
import shutil

# Hopla import
from hopla.converter import hopla

# Package import
from pypreclin.utils.reorient import check_orientation
from pypreclin.utils.export import ungzip_file
from pypreclin.utils.export import gzip_file
from pypreclin import preproc

# Third party import
import pyconnectome
import nibabel


def timeserie_to_reference(tfile, outdir, rindex=None,
                           restrict_deformation=(1, 1, 1), rfile=None, njobs=1,
                           clean_tmp=True):
    """ Register all the fMRI volumes to a reference volume identified by his
    index in the timeserie.

    The registration used is a non-linear regisration.

    Parameters
    ----------
    tfile: str
        the file that containes the timeserie.
    outdir: str
        the destination folder.
    rindex: int, default None
        the reference volume index in the timeserie.
    restrict_deformation: 3-uplet
        restrict the deformation in the given axis.
    rfile: str, default None
        the reference volume.
    njobs: int, default 1
        the desired number of parallel job during the registration.
    clean_tmp: bool, default True
        if set, clean the temporary results.

    Returns
    -------
    resfile: str
        the registration result.
    """
    # Check input index
    im = nibabel.load(tfile)
    array = im.get_fdata()
    if array.ndim != 4:
        raise ValueError("A timeserie (4d volume) is expected.")
    reference_image = None
    if rindex is not None:
        if rindex < 0 or rindex > array.shape[3]:
            raise ValueError(
                "Index '{0}' is out of bound considering the last dimension "
                "as the time dimension '{1}'.".format(rindex, array.shape))
    elif rfile is not None:
        reference_image = rfile
    else:
        raise ValueError("You need to specify a reference file or index.")

    # Split the timeserie
    tmpdir = os.path.join(outdir, "tmp")
    if not os.path.isdir(tmpdir):
        os.mkdir(tmpdir)
    moving_images = []
    outdirs = []
    for i in range(array.shape[3]):
        _im = nibabel.Nifti1Image(array[..., i], im.affine)
        _outfile = os.path.join(tmpdir, str(i).zfill(4) + ".nii.gz")
        outdirs.append(os.path.join(tmpdir, str(i).zfill(4)))
        if not os.path.isdir(outdirs[-1]):
            os.mkdir(outdirs[-1])
        nibabel.save(_im, _outfile)
        moving_images.append(_outfile)
        if reference_image is None and i == rindex:
            reference_image = _outfile

    # Start volume to volume non rigid registration
    scriptdir = os.path.join(os.path.dirname(pyconnectome.__file__), "scripts")
    script = os.path.join(scriptdir, "pyconnectome_ants_register")
    python_cmd = "python3"
    if not os.path.isfile(script):
        script = "pyconnectome_ants_register"
        python_cmd = None
    logfile = os.path.join(tmpdir, "log")
    if os.path.isfile(logfile):
        os.remove(logfile)
    status, exitcodes = hopla(
        script,
        b="/usr/lib/ants",
        o=outdirs,
        i=moving_images,
        r=reference_image,
        w=1,
        D=3,
        G=0.2,
        J=1,
        N=True,
        B=True,
        R=list(restrict_deformation),
        V=2,
        hopla_iterative_kwargs=["o", "i"],
        hopla_cpus=njobs,
        hopla_logfile=logfile,
        hopla_python_cmd=python_cmd,
        hopla_use_subprocess=True,
        hopla_verbose=1)
    if not (numpy.asarray(list(exitcodes.values())) == 0).all():
        raise ValueError("The registration failed, check the log "
                         "'{0}'.".format(logfile))

    # Start timeserie concatenation
    timeserie = []
    affine = None
    for path in outdirs:
        sid = os.path.basename(path)
        _im = nibabel.load(os.path.join(
            path,  "ants_2WarpToTemplate_{0}.nii.gz".format(sid)))
        if affine is None:
            affine = _im.affine
        elif not numpy.allclose(affine, im.affine, atol=1e-3):
            raise ValueError("Affine matrices must be the same: {0} - "
                             "{1}.".format(outdirs[0], path))
        data = _im.get_fdata()
        data.shape += (1, )
        timeserie.append(data)
    registered_array = numpy.concatenate(timeserie, axis=3)
    _im = nibabel.Nifti1Image(registered_array, affine)
    resfile = os.path.join(outdir, "ants_WarpToTemplate.nii.gz")
    nibabel.save(_im, resfile)

    # Clean temporary files if requested
    if clean_tmp:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return resfile


def ants_linear_register_command(moving, fixed, out_files, affine=False):
    """ Register a source image to a taget image using the 'ants'
    command.

    Parameters
    ----------
    moving: str (mandatory)
        the source Nifti image. Should be 3d, not 4d timeseries.
    fixed: str (mandatory)
        the target Nifti masked image. Should be 3d, not 4d timeseries.
    out_files: [str, str] (mandatory)
        the output files, the first is the xfm file, the second is the transformed image.
    """

    command_mean_realign = (
        f'antsRegistration --dimensionality 3 --float 0 '
        f'-o [{out_files[0]}, {out_files[1]}] '
        f'--interpolation Linear '
        f'--winsorize-image-intensities [0.005,0.995] '
        f'--use-histogram-matching 0 '
        f'--initial-moving-transform [{fixed},{moving},1] '
        f'--transform "Rigid[0.1]" '
        f'--metric "MI[{fixed},{moving},1,32,Regular,0.25]" '
        f'--convergence [1000x500x250x100,1e-6,10] '
        f'--shrink-factors 8x4x2x1 '
        f'--smoothing-sigmas 3x2x1x0vox '
        f'--float '
    )

    if affine:
        to_add = (
            f'--transform "Affine[0.1]" '
            f'--metric "MI[{fixed},{moving},1,32,Regular,0.25]" '
            f'--convergence [1000x500x250x100,1e-6,10] '
            f'--shrink-factors 8x4x2x1 '
            f'--smoothing-sigmas 3x2x1x0vox'
        )
        command_mean_realign += to_add

    return command_mean_realign


def jip_align(source_file, target_file, outdir, postfix='_space-template',
              auto=False, non_linear=False):
    """ Register a source image to a taget image using the 'jip_align'
    command.

    Parameters
    ----------
    source_file: str (mandatory)
        the source Nifti image.
    target_file: str (mandatory)
        the target Nifti masked image.
    outdir: str (mandatory)
        the destination folder.
    jipdir: str (mandatory)
        the jip binary path.
    prefix: str (optional, default 'w')
        prefix the generated file with this character.
    auto: bool (optional, default False)
        if set control the JIP window with the script.
    non_linear: bool (optional, default False)
        in the automatic mode, decide or not to compute the non-linear
        deformation field.
    fslconfig: str (optional)
        the FSL .sh configuration file.

    Returns
    -------
    register_file: str
        the registered image.
    register_mask_file: str
        the registered and masked image.
    native_masked_file: str
        the masked image in the native space.
    """

    # Check input image orientation: must be the same
    same_orient, orients = check_orientation([source_file, target_file])
    if not same_orient:
        print(
            "[WARNING] Source file '{0}' ({2}) and taget file '{1}' ({3}) "
            "must have the same orientation for JIP to work properly.".format(
                source_file, target_file, orients[0], orients[1]))

    # Change current working directory
    cwd = os.getcwd()
    os.chdir(outdir)

    # Only support uncompressed Nifti
    source_file = ungzip_file(source_file, prefix="", outdir=outdir)
    target_file = ungzip_file(target_file, prefix="", outdir=outdir)

    # # add 0.1 to the template to avoid jip crop func_mean_file by the mask of template
    # target_file_filled = target_file.split('.')[0] + '_filled.nii.gz'
    # command_temp_jip = f'fslmaths {target_file} -add 0.1 {target_file_filled}'
    # # print(f"Running command: {command_temp_jip}")
    # _ = subprocess.run(command_temp_jip, shell=True, check=True)
    # target_file_filled = ungzip_file(target_file_filled, prefix="", outdir=outdir)

    # blur template to avoid jip crop func_mean_file by the mask of template
    target_file_blurred = target_file.split('.')[0] + '_blurred.nii.gz'
    target_file_blurMask = target_file.split('.')[0] + '_blurMask.nii.gz'
    command_blur = f'fslmaths {target_file} -s 0.5 -bin {target_file_blurMask}'
    _ = subprocess.run(command_blur, shell=True, check=True)
    command_add = f'fslmaths {target_file} -add 0.1 -mas {target_file_blurMask} {target_file_blurred}'
    _ = subprocess.run(command_add, shell=True, check=True)
    target_file_blurred = ungzip_file(target_file_blurred, prefix="", outdir=outdir)
    # Copy source file
    align_file = os.path.join(outdir, "align.com")
    cmd = ["align", source_file, "-t", target_file_blurred]
    if auto:
        if non_linear:
            cmd = cmd + ["-L", "111111111111", "-W", "111", "-p 5 -j 5", "-a"]
        else:
            cmd = cmd + ["-L", "111111111111", "-W", "000", "-A"]
        if os.path.isfile(align_file):
            cmd += ["-I"]
        
    print(" ".join(cmd))
    subprocess.call(cmd)

    if not os.path.isfile(align_file):
        raise ValueError(
            "No 'align.com' file in '{0}' folder. JIP has probably failed: "
            "'{1}'".format(outdir, " ".join(cmd)))

    # Get the apply nonlinear deformation jip batches
    aplly_nonlin_batch = os.path.join(
        os.path.dirname(preproc.__file__), "resources", "apply_nonlin.com")
    aplly_inv_nonlin_batch = os.path.join(
        os.path.dirname(preproc.__file__), "resources", "apply_inv_nonlin.com")

    # apply the xfm to the source image
    register_file = os.path.join(outdir, 
                                 os.path.basename(source_file).split('.')[0] + postfix + '.nii')
    cmd = ["jip", aplly_nonlin_batch, source_file, register_file]
    subprocess.call(cmd)

    # Apply mask
    register_mask_file = os.path.join(
        outdir, os.path.basename(register_file).split(".")[0] + "_mask.nii.gz")
    command_apply_mask = f'fslmaths {register_file} -mas {target_file} {register_mask_file}'
    # print(f"Running command: {command_apply_mask}")
    _ = subprocess.run(command_apply_mask, shell=True, check=True)

    register_mask_file = ungzip_file(register_mask_file, prefix="", outdir=outdir)

    # Send back masked image to original space
    native_masked_file = os.path.join(
        outdir, os.path.basename(source_file).split(".")[0] + "_mask.nii")
    cmd = ["jip", aplly_inv_nonlin_batch, register_mask_file,
           native_masked_file]
    subprocess.call(cmd)              
 
    # Restore current working directory and gzip output
    os.chdir(cwd)
    register_file = gzip_file(
        register_file, prefix="", outdir=outdir,
        remove_original_file=True)
    register_mask_file = gzip_file(
        register_mask_file, prefix="", outdir=outdir,
        remove_original_file=True)
    native_masked_file = gzip_file(
        native_masked_file, prefix="", outdir=outdir,
        remove_original_file=True)

    return register_file, register_mask_file, native_masked_file, align_file


def apply_jip_align(apply_to_files, align_with, outdir, postfix="_space-template",
                    apply_inv=False):
    """ Apply a jip deformation.

    Parameters
    ----------
    apply_to_files: list of str (mandatory)
        a list of image path to apply the computed deformation.
    align_with: str ot list of str (mandatory)
        the alignement file containind the deformation parameters. Expect
        an 'align.com' file.
    outdir: str (mandatory)
        the destination folder.
    jipdir: str (mandatory)
        the jip binary path.
    prefix: str (optional, default 'w')
        prefix the generated file with this character.
    apply_inv: bool (optional, default False)
        if set apply the inverse deformation.

    Returns
    -------
    deformed_files: list of str
        the deformed input files.
    """
    # Check input parameters
    if not isinstance(align_with, list):
        align_with = [align_with]
    if not isinstance(apply_to_files, list):
        raise ValueError("The 'apply_to_files' function parameter must "
                         "contains a list of files.")
    for path in apply_to_files + align_with:
        if not os.path.isfile(path):
            raise ValueError("'{0}' is not a valid file.".format(path))

    # Get the apply nonlinear deformation jip batches
    if apply_inv:
        batch = os.path.join(os.path.dirname(preproc.__file__), "resources",
                             "apply_inv_nonlin.com")
    else:
        batch = os.path.join(os.path.dirname(preproc.__file__), "resources",
                             "apply_nonlin.com")
    batch = os.path.abspath(batch)

    # # Create jip environment
    # jip_envriron = os.environ
    # jip_envriron["JIP_HOME"] = os.path.dirname(jipdir)
    # if "PATH" in jip_envriron:
    #     jip_envriron["PATH"] = jip_envriron["PATH"] + ":" + jipdir
    # else:
    #     jip_envriron["PATH"] = jipdir

    # Apply the jip deformation
    deformed_files = []
    cwd = os.getcwd()
    for path in apply_to_files:
        extra_file = ungzip_file(path, prefix="", outdir=outdir)

        deformed_file = os.path.join(
            outdir, os.path.basename(extra_file).split('.')[0] + postfix + '.nii')
        
        for align_file in align_with:
            os.chdir(os.path.dirname(align_file))
            cmd = ["jip", batch, extra_file, deformed_file]
            subprocess.call(cmd)
            # extra_file = deformed_file

        deformed_file = gzip_file(
            deformed_file, prefix="", outdir=outdir, remove_original_file=True)
        deformed_files.append(deformed_file)

        # remove the .nii files
        os.remove(extra_file)

    os.chdir(cwd)

    return deformed_files


def check_jip_install(jipdir):
    """ Simple function to test if the JIP software bibaries are present.

    Parameters
    ----------
    jip_dir: str
        the JIP binaries directory.

    Returns
    -------
    status: bool
        the JIP installation status.
    """
    status = True
    for name in ("jip", "align"):
        binary_path = os.path.join(jipdir, name)
        if not os.path.isfile(binary_path):
            status = False
            break
    return status
