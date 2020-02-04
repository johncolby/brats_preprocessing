import os
import pkg_resources
import shutil
import requests
import argparse
import logging

from nipype.interfaces import fsl

from .pipelines import dcm2nii, non_t1, merge_orient

from rad_apps.radstudy import RadStudy

class TumorStudy(RadStudy):
    def __init__(self, acc='', zip_path='', model_path='', n_procs=4):
        super().__init__(acc, zip_path, model_path)
        self.app_name  = 'gbm'
        self.MNI_ref   = fsl.Info.standard_image('MNI152_T1_1mm_brain.nii.gz')
        self.brats_ref = pkg_resources.resource_filename(__name__, 'brats_ref_reorient.nii.gz')
        self.n_procs   = n_procs

    def preprocess(self, mni_mask = False, do_bias_correct = False):
        """Preprocess clinical data according to BraTS specs"""
        wf = dcm2nii(self.dir_tmp)
        wf.inputs.inputnode.df = self.series_picks
        wf.run('MultiProc', plugin_args={'n_procs': self.n_procs})

        wf = non_t1(self.dir_tmp, self.MNI_ref, mni_mask)
        modalities = [x for x in self.channels if x != 't1']
        wf.inputs.t1_workflow.inputnode.t1_file = os.path.join(self.dir_tmp, 'nii', 't1.nii.gz')
        wf.get_node('inputnode').iterables = [('modality', modalities)]
        wf.write_graph(graph2use='flat', format='pdf')
        wf.write_graph(graph2use='colored', format='pdf')
        wf.run('MultiProc', plugin_args={'n_procs': self.n_procs})

        wf = merge_orient(self.dir_tmp, self.brats_ref, do_bias_correct)
        in_files = [os.path.join(self.dir_tmp, 'mni', x + '.nii.gz') for x in self.channels]
        if do_bias_correct:
            in_files.reverse()
        wf.inputs.inputnode.in_files = in_files
        wf.run('MultiProc', plugin_args={'n_procs': self.n_procs})

    def segment(self, endpoint):
        """Send POST request to model server endpoint and download results"""
        preproc_path = os.path.join(self.dir_tmp, 'output', 'preprocessed.nii.gz')
        data = open(preproc_path, 'rb').read()
        download_stream = requests.post(endpoint, 
                                        files = {'data': data}, 
                                        stream = True)
        # Save output to disk
        mask_path = os.path.join(self.dir_tmp, 'output', 'mask.nii.gz')
        with open(mask_path, 'wb') as fd:
            for chunk in download_stream.iter_content(chunk_size=8192):
                if chunk:
                    _ = fd.write(chunk)
        # Save a version with matrix size matching MNI
        FLIRT = fsl.FLIRT(in_file = mask_path, 
                          reference = self.MNI_ref, 
                          apply_xfm = True,
                          uses_qform = True,
                          out_file = os.path.join(self.dir_tmp, 'output', 'mask_mni.nii.gz'),
                          out_matrix_file = os.path.join(self.dir_tmp, 'output', 'mask_mni.mat'))
        FLIRT.run()
        # Save a template itksnap workspace
        itk_file = pkg_resources.resource_filename(__name__, 'workspace.itksnap')
        shutil.copy(itk_file, os.path.join(self.dir_tmp, 'output'))

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('acc', metavar='accession', help='Accession # to process')
    parser.add_argument('air_url', help='URL for AIR API, e.g. https://air.<domain>.edu/api/')
    parser.add_argument('model_path', help='path to model.Rdata for dcmclass')
    parser.add_argument('seg_url', help='URL for segmentation API')
    parser.add_argument('-c', '--cred_path', help='Login credentials file. If not present, will look for AIR_USERNAME and AIR_PASSWORD environment variables.', default=None)
    parser.add_argument('--mni_mask', help='Use an atlas-based mask instead of subject-based', action='store_true', default=False)
    parser.add_argument('--do_bias_correct', help='Use FSL FAST for multi-channel bias field correction', action='store_true', default=False)
    parser.add_argument('--output_dir', help='Parent directory in which to save output', default='.')
    arguments = parser.parse_args()
    return arguments

def process_gbm(args):
    try:
        mri = TumorStudy(acc = args.acc, model_path = args.model_path)
        mri.setup()
        mri.download(URL = args.air_url, cred_path = args.cred_path)
        mri.setup()
        mri.classify_series()
        mri.preprocess(args.mni_mask, args.do_bias_correct)
        mri.segment(endpoint = args.seg_url)
        mri.report()
        mri.copy_results(output_dir = args.output_dir)
        mri.rm_tmp()
    except:
        logging.exception('Processing failed.')
        mri.rm_tmp()

def cli():
    process_gbm(parse_args())