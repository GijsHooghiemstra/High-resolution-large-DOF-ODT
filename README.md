# High-resolution-large-DOF-ODT

> [!IMPORTANT]
> (14-6-2024) Code is being cleaned for readability.

Repository containing code used for my master thesis.

Contents:
 - Python scripts for Mie code: generateField.py (Mie sphere), generateFieldSingleCylinder.py (Mie cylinder)
 - Jupyter notebook for defocus ODT
 - Jupyter notebook for Zernike aberrations
 - Jupyter notebook for Born / Rytov
 - Jupyter notebook for iterative reconstructoin
 - CUDA code for filtered backpropagation algorithm

Used libraries Python:
 - (required) numpy, scipy, matplotlib.pyplot
 - (required) pyshtools, required for Mie code, package fixes ([problem](https://github.com/scipy/scipy/issues/7778))
 - (optional) tqdm, optional tool used for progress bars
 - (optional) multiprocess, allows parallel computation of Mie field for spheres, recommended for speedup
   if not used, see generateFieldSingleCylinder.py for similar implementation without parallel computation

Used libraries CUDA:
 - (required) CUDA Toolkit
 - (required) cufft
