
import os
import pkg_resources
import shutil
import tempfile
import zipfile
import nibabel as nib

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
        self.acc          = os.path.splitext(os.path.basename(self.zip_path))[0] if self.zip_path else acc
        assert self.acc, 'No accession number provided.'

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
        dir_study = os.path.join(self.dir_tmp, self.acc)
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


    def classify_series(self):
        """Classify series into modalities"""
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()

        pkgs = ['oro.dicom', 'tidyverse', 'tidytext', 'tm', 'caret']
        _ = [ro.r['library'](x) for x in pkgs]
        ro.r['load'](self.model_path)

        self.series_picks = ro.r['predict_headers'](os.path.dirname(self.dir_study), ro.r['models'], ro.r['tb_preproc'])
        paths = [os.path.abspath(os.path.join(self.dir_study, series)) for series in self.series_picks.series.tolist()]
        self.series_picks['series'] = paths

    def add_paths(self, paths):
        """Manually specify directory paths to required series"""
        self.series_picks.series = paths

    def preprocess(self):
        """Preprocess clinical data according to BraTS specs"""
        wf = dcm2nii(self.dir_tmp)
        wf.inputs.inputnode.df = self.series_picks
        wf.run('MultiProc', plugin_args={'n_procs': self.n_procs})

        wf = non_t1(self.dir_tmp, self.MNI_ref)
        modalities = [x for x in self.channels if x != 't1']
        wf.inputs.t1_workflow.inputnode.t1_file = os.path.join(self.dir_tmp, 'nii', 't1.nii.gz')
        wf.get_node('inputnode').iterables = [('modality', modalities)]
        wf.write_graph(graph2use='flat', format='pdf')
        wf.write_graph(graph2use='colored', format='pdf')
        wf.run('MultiProc', plugin_args={'n_procs': self.n_procs})

        wf = merge_orient(self.dir_tmp, self.brats_ref)
        wf.inputs.inputnode.in_files = [os.path.join(self.dir_tmp, 'mni', x + '.nii.gz') for x in self.channels[::-1]]
        wf.run('MultiProc', plugin_args={'n_procs': self.n_procs})

    def __str__(self):
        s_picks = str(self.series_picks.iloc[:, 0:3]) if not self.series_picks.empty else ''
        s = ('Brain Tumor object\n'
            f'  Accession #: {self.acc}\n'
            f'  tmp_dir: {self.dir_tmp}\n'
            f'  Series picks:\n{s_picks}')
        return s

    def rm_tmp(self):
        """Remove temporary working area"""
        if not self.dir_tmp == '':
            shutil.rmtree(self.dir_tmp)
        else:
            print('Nothing to remove.')
