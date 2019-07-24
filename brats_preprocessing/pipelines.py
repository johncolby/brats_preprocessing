import os

from nipype import Node, MapNode, Workflow, SelectFiles, Function
from nipype.interfaces import fsl
from nipype.interfaces.dcm2nii import Dcm2niix
from nipype.interfaces.io import DataSink
from nipype.interfaces.utility import IdentityInterface, Select

def dcm2nii(base_dir):
    """Convert raw DICOM data to NIfTI"""
    def parse_df(df):
        return [df['class'].tolist(), df['series'].tolist()]

    inputnode = Node(Function(input_names  = ['df'],
                              output_names = ['channel', 'path'],
                              function     = parse_df),
                     name='inputnode')
    dcm2niix  = MapNode(Dcm2niix(bids_format = False, output_dir = os.path.join(base_dir, 'nii')),
                        iterfield=['source_dir', 'out_filename'], 
                        name='dcm2niix')

    wf = Workflow(name='dcm2nii', base_dir=os.path.join(base_dir, 'nipype'))
    wf.connect(inputnode , 'path'    , dcm2niix , 'source_dir')
    wf.connect(inputnode , 'channel' , dcm2niix , 'out_filename')
    return wf

def t1(reference, mni_mask = False):
    """BraTS nipype workflow for T1 data"""
    inputnode   = Node(IdentityInterface(fields=['t1_file']), name='inputnode')
    skullstrip1 = Node(fsl.BET(mask=True), name='skullstrip1')
    reg_to_mni  = Node(fsl.FLIRT(reference=reference), name='reg_to_mni')
    applyxfm    = Node(fsl.FLIRT(reference=reference, apply_xfm=True), name='applyxfm')
    skullstrip2 = Node(fsl.BET(mask=True), name='skullstrip2')
    apply_mask  = Node(fsl.maths.ApplyMask(), name='apply_mask')
    outputnode  = Node(IdentityInterface(fields=['t1_file',
                                                't1_brain', 
                                                't1_brain_mni', 
                                                't1_brain_mask_mni', 
                                                'flirt_mat']), 
                       name='outputnode')

    wf = Workflow(name='t1_workflow')
    wf.connect(inputnode   , 't1_file'         , skullstrip1 , 'in_file')
    wf.connect(inputnode   , 't1_file'         , outputnode  , 't1_file')
    wf.connect(inputnode   , 't1_file'         , applyxfm    , 'in_file')
    wf.connect(skullstrip1 , 'out_file'        , reg_to_mni  , 'in_file')
    wf.connect(reg_to_mni  , 'out_matrix_file' , outputnode  , 'flirt_mat')
    wf.connect(reg_to_mni  , 'out_matrix_file' , applyxfm    , 'in_matrix_file')

    if mni_mask:
        wf.connect(applyxfm   , 'out_file' , apply_mask , 'in_file')
        wf.connect(apply_mask , 'out_file' , outputnode , 't1_brain_mni')
        mni_mask_path = fsl.Info.standard_image('MNI152_T1_1mm_brain_mask.nii.gz')
        wf.inputs.apply_mask.mask_file = mni_mask_path
        wf.inputs.outputnode.t1_brain_mask_mni = mni_mask_path
    else:
        wf.connect(applyxfm    , 'out_file'  , skullstrip2 , 'in_file')
        wf.connect(skullstrip2 , 'out_file'  , outputnode  , 't1_brain_mni')
        wf.connect(skullstrip2 , 'mask_file' , outputnode  , 't1_brain_mask_mni')

    return wf

def non_t1(base_dir, reference, mni_mask = False):
    """BraTS nipype workflow for non-T1 data"""
    t1_wf = t1(reference, mni_mask)

    inputnode    = Node(IdentityInterface(fields=['modality']), name='inputnode')
    template = {'file_name': '{modality}.nii.gz'}
    select_files = Node(SelectFiles(template, base_directory=os.path.join(base_dir, 'nii')), name='selectfiles')
    reg_to_t1    = Node(fsl.FLIRT(dof=6, cost='mutualinfo'), name='reg_to_t1')
    concat_xfms  = Node(fsl.ConvertXFM(concat_xfm=True), name='concat_xfms')
    reg_to_mni   = Node(fsl.FLIRT(reference=reference, apply_xfm=True), name='reg_to_mni')
    apply_mask   = Node(fsl.maths.ApplyMask(), name='apply_mask')
    datasink     = Node(DataSink(parameterization=False, base_directory=base_dir), name='datasink')
    datasink.inputs.regexp_substitutions = [('(/[^_]+)[^/]*(\\.nii\\.gz)', '\\1\\2')]

    wf = Workflow(name='nipype', base_dir=base_dir)
    wf.connect(inputnode    , 'modality'                     , select_files , 'modality')
    wf.connect(select_files , 'file_name'                    , reg_to_t1    , 'in_file')
    wf.connect(t1_wf        , 'outputnode.t1_file'           , reg_to_t1    , 'reference')
    wf.connect(reg_to_t1    , 'out_matrix_file'              , concat_xfms  , 'in_file')
    wf.connect(t1_wf        , 'outputnode.flirt_mat'         , concat_xfms  , 'in_file2')
    wf.connect(concat_xfms  , 'out_file'                     , reg_to_mni   , 'in_matrix_file')
    wf.connect(select_files , 'file_name'                    , reg_to_mni   , 'in_file')
    wf.connect(reg_to_mni   , 'out_file'                     , apply_mask   , 'in_file')
    wf.connect(t1_wf        , 'outputnode.t1_brain_mask_mni' , apply_mask   , 'mask_file')
    wf.connect(apply_mask   , 'out_file'                     , datasink     , 'mni')
    wf.connect(t1_wf        , 'outputnode.t1_brain_mni'      , datasink     , 'mni.@t1')

    return wf

def merge_orient(base_dir, reference):
    """Merge and reorient workflow"""
    inputnode    = Node(IdentityInterface(fields=['in_files']), name='inputnode')
    bias_correct = Node(fsl.FAST(output_biasfield=True, output_biascorrected=True), name='bias_correct')
    merge        = Node(fsl.Merge(dimension='t'), name='merge')
    int16        = Node(fsl.maths.ChangeDataType(output_datatype='short'), name='int16')
    pad_orient   = Node(fsl.FLIRT(reference=reference, apply_xfm=True, uses_qform=True), name='pad_orient')
    datasink     = Node(DataSink(base_directory=base_dir), name='datasink')
    datasink.inputs.substitutions = [('flair_restore_1_merged_chdt_flirt', 'preprocessed')]

    wf = Workflow(name='merge_and_orient', base_dir=os.path.join(base_dir, 'nipype'))
    wf.connect(inputnode    , 'in_files'       , bias_correct , 'in_files')
    wf.connect(bias_correct , 'restored_image' , merge        , 'in_files')
    wf.connect(merge        , 'merged_file'    , int16        , 'in_file')
    wf.connect(int16        , 'out_file'       , pad_orient   , 'in_file')
    wf.connect(pad_orient   , 'out_file'       , datasink     , 'output')
    return wf