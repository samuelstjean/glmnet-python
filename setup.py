from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup
import os
# from numpy.distutils.system_info import get_info
# import os, sys

config = Configuration(
    'glmnet',
    parent_package=None,
    top_path=None
)

build_f2py = "cd glmnet; f2py -c --fcompiler=gnu95 --f77flags='-fdefault-real-8' --f90flags='-fdefault-real-8' glmnet.pyf glmnet.f; cd .."
os.system(build_f2py)

f_sources = ['glmnet.pyf', 'glmnet.f']
fflags = ['-fdefault-real-8', '-ffixed-form']

# config.add_extension(name='_glmnet',
#                      sources=f_sources,
#                      extra_f77_compile_args=fflags,
#                      extra_f90_compile_args=fflags)

config_dict = config.todict()

if __name__ == '__main__':

    setup(version='0.9',
          description='Python wrappers for the GLMNET package',
          # author='Matthew Drury',
          # author_email='matthew.drury.83@gmail.com',
          # url='github.com/madrury/glmnet-python',
          license='GPL2',
          requires=['NumPy (>= 1.3)'],
          **config_dict)
