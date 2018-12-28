from setuptools import setup,find_packages
setup(name="faceai",
      version='0.3.7',
      description='Deep Learning library for applications about face.',
      url="https://github.com/jimmy0087/faceai-master",
      author='JimmyYoung',
      license='JimmyYoung',
      packages= find_packages(),
      #packages = ['faceai','faceai/Alignment/*.py','faceai/Detection/*.py','faceai/ThrDFace/*.py'],
      zip_safe=False
      )