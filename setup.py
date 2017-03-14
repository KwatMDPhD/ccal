from setuptools import setup
from setuptools.command.install import install


class InstallCommand(install):
    def run(self):
        install.run(self)
        from distutils import log
        log.set_verbosity(log.DEBUG)


setup(name='ccal',
      description='Computational Cancer Analysis Library',
      packages=['ccal'],
      version='0.0.1',
      author='Huwate (Kwat) Yeerna (Medetgul-Ernar)',
      author_email='kwat.medetgul.ernar@gmail.com',
      license='MIT',
      url='https://github.com/ucsd-ccal/ccal',
      classifiers=['Development Status :: 3 - Alpha',
                   'License :: OSI Approved :: MIT License',
                   'Programming Language :: Python :: 3.5'],
      keywords=['computational cancer biology genomics'],
      install_requires=[
          'rpy2',
          'biopython',
          'plotly',
      ],
      cmdclass={'install': InstallCommand},
      package_data={})
