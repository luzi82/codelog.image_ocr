from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'cairocffi',
    'editdistance',
    'Keras',
    'h5py',
    'captcha',
    'google-cloud-storage',
]

setup(
    name='codelog_image_ocr',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Image OCR code samples'
)
