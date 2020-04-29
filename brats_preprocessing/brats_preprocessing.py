import mxnet as mx
import os
import pickle
import pkg_resources
import shutil
import requests
import argparse

from nipype.interfaces import fsl

from .pipelines import dcm2nii, non_t1, merge_orient

from radstudy import RadStudy
from unet_brats.unet import nii_to_tensor, tensor_to_nii


class TumorStudy(RadStudy):

    def __init__(self, mni_mask=False, do_bias_correct=False, n_procs=4, **kwargs):
        super().__init__(**kwargs)

        self.app_name = 'gbm'
        self.MNI_ref = fsl.Info.standard_image('MNI152_T1_1mm_brain.nii.gz')
        self.brats_ref = pkg_resources.resource_filename(__name__, 'brats_ref_reorient.nii.gz')
        self.n_procs = n_procs
        self.channels = ['flair', 't1', 't1ce', 't2']
        self.mni_mask = mni_mask
        self.do_bias_correct = do_bias_correct

    def process(self):
        self.classify_series()
        self.preprocess()
        self.segment()

    def preprocess(self):
        """Preprocess clinical data according to BraTS specs"""
        wf = dcm2nii(self.dir_tmp)
        wf.inputs.inputnode.df = self.series_picks
        wf.run('MultiProc', plugin_args={'n_procs': self.n_procs})

        wf = non_t1(self.dir_tmp, self.MNI_ref, self.mni_mask)
        modalities = [x for x in self.channels if x != 't1']
        wf.inputs.t1_workflow.inputnode.t1_file = os.path.join(self.dir_tmp, 'nii', 't1.nii.gz')
        wf.get_node('inputnode').iterables = [('modality', modalities)]
        wf.write_graph(graph2use='flat', format='pdf')
        wf.write_graph(graph2use='colored', format='pdf')
        wf.run('MultiProc', plugin_args={'n_procs': self.n_procs})

        wf = merge_orient(self.dir_tmp, self.brats_ref, self.do_bias_correct)
        in_files = [os.path.join(self.dir_tmp, 'mni', x + '.nii.gz') for x in self.channels]
        if self.do_bias_correct:
            in_files.reverse()
        wf.inputs.inputnode.in_files = in_files
        wf.run('MultiProc', plugin_args={'n_procs': self.n_procs})

    def segment(self):
        """Send POST request to model server endpoint and download results"""
        # Prepare data
        preproc_path = os.path.join(self.dir_tmp, 'output', 'preprocessed.nii.gz')
        img, nii_hdr = nii_to_tensor(preproc_path)
        img_str = pickle.dumps(mx.nd.array(img))

        # Post request to inference server
        mask_str = requests.post(self.process_url,
                                 files={'data': img_str},
                                 stream=True)

        # Convert predicted mask back to nii and save to disk
        mask = pickle.loads(mask_str.content)
        mask = mx.nd.array(mask).argmax_channel().squeeze().asnumpy()
        mask_nii = tensor_to_nii(mask, nii_hdr)
        mask_path = os.path.join(self.dir_tmp, 'output', 'mask.nii.gz')
        mask_nii.to_filename(mask_path)

        # Save a version with matrix size matching MNI
        FLIRT = fsl.FLIRT(in_file=mask_path,
                          reference=self.MNI_ref,
                          apply_xfm=True,
                          uses_qform=True,
                          out_file=os.path.join(self.dir_tmp, 'output', 'mask_mni.nii.gz'),
                          out_matrix_file=os.path.join(self.dir_tmp, 'output', 'mask_mni.mat'))
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

def cli():
    process_gbm(parse_args())