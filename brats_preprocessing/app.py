from flask_wtf import FlaskForm
import os
from urllib.parse import urljoin
from wtforms import BooleanField, FormField, IntegerField
from wtforms.validators import Optional

from rad_apps.appplugin import AppPlugin
from .brats_preprocessing import TumorStudy


class Inputs(FlaskForm):
    flair = IntegerField('FLAIR', validators=[Optional()])
    t1 = IntegerField('T1', validators=[Optional()])
    t1ce = IntegerField('T1CE', validators=[Optional()])
    t2 = IntegerField('T2', validators=[Optional()])

    def __init__(self, csrf_enabled=False, *args, **kwargs):
        super(Inputs, self).__init__(csrf_enabled=csrf_enabled, *args, **kwargs)


class Options(FlaskForm):
    mni_mask = BooleanField('MNI template')
    bias_correct = BooleanField('Bias field correct')
    inputs = FormField(Inputs)

    def __init__(self, csrf_enabled=False, *args, **kwargs):
        super(Options, self).__init__(csrf_enabled=csrf_enabled, *args, **kwargs)


def wrapper_fun(app, form):
    TumorStudy(acc=form['acc'],
               download_url=app.config['AIR_URL'],
               cred_path=app.config['DOTENV_FILE'],
               model_path=app.config['MODEL_RDATA'],
               process_url=urljoin(app.config['SEG_URL'], 'gbm'),
               output_dir=os.path.join(app.config['OUTPUT_DIR_NODE'], 'gbm'),
               mni_mask=form['opts']['mni_mask'],
               do_bias_correct=form['opts']['bias_correct']
               ).run()


app = AppPlugin(long_name='Glioblastoma',
                short_name='gbm',
                form_opts=Options,
                wrapper_fun=wrapper_fun)
