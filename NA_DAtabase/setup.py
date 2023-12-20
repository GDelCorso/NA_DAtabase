from setuptools import setup

setup(
      name='NA_DAtabase',
      version='0.01',
      description='NADA: NotA- DAtabase is a synthetic dataset generator.',
      author='Claudia Caudai, Giulio Del Corso, Federico Volpini',
      author_email='giulio.delcorso@isti.cnr.it',
      install_requires=[
          'os',
          'shutil',
          'pandas',
          'numpy',
          'openturns',
          'random',
          'PIL',
          'cv2',
          'math'
          ]
      )

