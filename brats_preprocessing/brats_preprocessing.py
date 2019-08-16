import os
import pkg_resources
import shutil
import tempfile
import zipfile
import nibabel as nib
import requests
import pydicom
import glob
from datetime import datetime
import argparse

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()

from nipype.interfaces import fsl

import pandas as pd

from .pipelines import dcm2nii, non_t1, merge_orient

class tumor_study():
    def __init__(self, acc='', zip_path='', model_path='', n_procs=4):
        self.zip_path     = zip_path
        self.model_path   = model_path
        self.dir_tmp      = ''
        self.dir_study    = ''
        self.channels     = ['flair', 't1', 't1ce', 't2']
        self.series_picks = pd.DataFrame({'class': self.channels,
                                          'prob': '',
                                          'SeriesNumber': '',
                                          'series': ''})
        self.MNI_ref      = fsl.Info.standard_image('MNI152_T1_1mm_brain.nii.gz')
        self.brats_ref    = pkg_resources.resource_filename(__name__, 'brats_ref_reorient.nii.gz')
        self.n_procs      = n_procs
        self.acc          = acc
        self.hdr          = ''
        self.study_date   = ''
        assert self.acc or self.zip_path, 'No input study provided.'

    def download(self, URL, cred_path):
        """Download study via AIR API"""
        import air_download.air_download as air
        import argparse

        assert not self.zip_path, '.zip path already available.'
        assert self.dir_tmp, 'Working area not setup yet.'
        args = argparse.Namespace()
        args.URL = URL
        args.acc = self.acc
        args.cred_path = cred_path
        args.profile = -1
        args.output = os.path.join(self.dir_tmp, f'{self.acc}.zip')
        air.main(args)
        self.zip_path = args.output
        self._extract()

    def _extract(self):
        """Extract study archive"""
        assert not self.dir_study, 'dir_study already exists.'
        dir_study = os.path.join(self.dir_tmp, 'dcm')
        os.mkdir(dir_study)
        zip_ref = zipfile.ZipFile(self.zip_path, 'r')
        zip_ref.extractall(path = dir_study)
        self.dir_study = os.path.join(dir_study, os.listdir(dir_study)[0])

    def setup(self):
        """Setup study for processing"""
        # Create temporary working directory
        if not self.dir_tmp:
            self.dir_tmp = tempfile.mkdtemp()
            os.mkdir(os.path.join(self.dir_tmp, 'nii'))

        # Extract study archive
        if not self.dir_study and self.zip_path:
            self._extract()

        # Load representative DICOM header
        if self.dir_study and not self.hdr:
            dcm_path = glob.glob(f'{self.dir_study}/*/*.dcm', recursive=True)[0]
            self.hdr = pydicom.read_file(dcm_path)
            self.acc = self.hdr.AccessionNumber
            self.study_date = datetime.strptime(self.hdr.StudyDate, '%Y%m%d').strftime('%m/%d/%Y')

    def classify_series(self):
        """Classify series into modalities"""
        ro.r['library']('dcmclass')
        ro.r['load'](self.model_path)

        self.series_picks = ro.r['predict_headers'](os.path.dirname(self.dir_study), ro.r['models'], ro.r['tb_preproc'])
        paths = [os.path.abspath(os.path.join(self.dir_study, series)) for series in self.series_picks.series.tolist()]
        self.series_picks['series'] = paths

    def add_paths(self, paths):
        """Manually specify directory paths to required series"""
        self.series_picks.series = paths

    def series_to_path(self, series):
        """Convert a SeriesNumber to path"""
        ro.r['library']('dcmclass')
        ro.r['library']('dplyr')
        tb = ro.r['load_study_headers'](os.path.join(self.dir_tmp, 'dcm'), 'SeriesNumber')
        return tb.loc[tb['SeriesNumber'] == series]['path'][0]

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

    def report(self):
        """Generate PDF report"""
        ro.r['library']('ucsfreports')
        params = ro.ListVector({'input_path':   self.dir_tmp,
                                'patient_name': self.hdr.PatientName.family_comma_given(),
                                'patient_MRN':  self.hdr.PatientID,
                                'patient_acc':  self.hdr.AccessionNumber,
                                'study_date':   self.study_date})
        ro.r['ucsf_report']('gbm', output_dir = self.dir_tmp, params = params)

    def copy_results(self, output_dir = '.'):
        src = os.path.join(self.dir_tmp, 'gbm.pdf')
        dest = os.path.join(output_dir, f'{self.acc}_gbm.pdf')
        shutil.copyfile(src, dest)

    def __str__(self):
        s_picks = str(self.series_picks.iloc[:, 0:3]) if not self.series_picks.empty else ''
        s = ('Brain Tumor object\n'
            f'  Accession #: {self.acc}\n'
            f'  dir_tmp: {self.dir_tmp}\n'
            f'  Series picks:\n{s_picks}')
        return s

    def rm_tmp(self):
        """Remove temporary working area"""
        if not self.dir_tmp == '':
            shutil.rmtree(self.dir_tmp)
        else:
            print('Nothing to remove.')

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('acc', metavar='accession', help='Accession # to process')
    parser.add_argument('url_air', help='URL for AIR API, e.g. https://air.<domain>.edu/api/')
    parser.add_argument('model_path', help='path to model.Rdata for dcmclass')
    parser.add_argument('url_seg', help='URL for segmentation API')
    parser.add_argument('-c', '--cred_path', help='Login credentials file', default='./air_login.txt')
    parser.add_argument('--mni_mask', help='Use an atlas-based mask instead of subject-based', action='store_true', default=False)
    parser.add_argument('--do_bias_correct', help='Use FSL FAST for multi-channel bias field correction', action='store_true', default=False)
    arguments = parser.parse_args()
    return arguments

def cli():
    args = parse_args()

    try:
        mri = tumor_study(acc = args.acc, model_path = args.model_path)
        mri.setup()
        mri.download(URL = args.url_air, cred_path = args.cred_path)
        mri.setup()
        mri.classify_series()
        mri.preprocess(args.mni_mask, args.do_bias_correct)
        mri.segment(endpoint = args.url_seg)
        mri.report()
        mri.copy_results()
        mri.rm_tmp()
    except:
        print("Processing failed.")
        mri.rm_tmp()
