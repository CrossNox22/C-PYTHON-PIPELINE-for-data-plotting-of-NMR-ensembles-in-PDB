# C++/PYTHON-PIPELINE for data plotting of NMR ensembles in PDB
A hybrid C++/Python pipeline for the structural validation and dynamic profiling of NMR protein ensembles. Features include RMSF, SASA, PCA, DCCM, and Ramachandran analysis.

C-PYTHON-PIPELINE is a high-performance computational framework designed for the structural validation, dynamic profiling, and multidimensional characterization of protein NMR ensembles.

Leveraging a hybrid architecture, the pipeline integrates the computational efficiency of C++ (Eigen) for heavy geometric algorithms with the flexibility of Python (NumPy/Matplotlib) for data orchestration and advanced visualization. This tool provides a comprehensive suite of metrics to explore the conformational space of biomolecules, bridging the gap between raw atomic coordinates and biological function.

The pipeline implements a multi-parametric approach to analyze protein dynamics and stability:

- Root Mean Square Fluctuation (RMSF): precise identification of rigid core regions versus flexible loops using an optimized C++ implementation.

- Principal Component Analysis (PCA): Dimensionality reduction of the coordinate space to identify essential dynamics and major conformational modes (e.g., open/closed states).

- Dynamic Cross-Correlation Matrix (DCCM): Analysis of correlated motions to detect allosteric communication pathways and domain coupling.

- Solvent Accessible Surface Area (SASA): High-speed calculation using the Shrake-Rupley algorithm (Golden Spiral point distribution) to quantify residue exposure.

- Multi-Physics Profiling: Integrated visualization correlating flexibility (RMSF), hydrophobicity (Kyte-Doolittle scale), and solvent accessibility (SASA) to identify potential interaction sites.

- Ramachandran Plot Analysis: Automated validation of backbone torsional angles ($\phi, \psi$) to assess stereochemical quality.

- Radius of Gyration (Rg): Distribution analysis to monitor structural compactness and detect unfolded models within the ensemble.



Installation and Build Instructions

Prerequisites:
- Eigen Library:
  Install via your package manager:
  Ubuntu/Debian: sudo apt-get install libeigen3-dev
  macOS: brew install eigen
  Windows (vcpkg): vcpkg install eigen3

- numpy>=1.21.0
- matplotlib>=3.4.0 

pip install -r requirements.txt

C++ Engine Compilation
g++ -I ./eigen_lib analyzer.cpp -o analyzer -O3
The -O3 optimization flag is recommended to enhance the performance of SASA calculations

Usage
Input Data: Place your PDB ensemble files (e.g., 1D3Z.pdb) in the root directory.
Configuration: Modify the PROTEIN_IDS list within analyzer.py to specify your target ensembles.
e.g PROTEIN_IDS = ["1D3Z"]
Run: Execute the main script:
python analyzer.py

1D3Z.pdb is provided as a sample file


All results are automatically generated in the results/ folder:
- multiphysics.png: The master plot overlaying Dynamics and Surface properties.
- pca.png: 2D projection of the conformational space (Evolution of models).
- dccm.png: Heatmap of residue-residue correlations.
- ramachandran.png: Stereochemical validation plot.
- analysis.csv: Comprehensive per-residue data (RMSF, SASA) for statistical analysis.

Methodology
Alignment: Uses the Kabsch Algorithm (SVD) to superimpose all models onto a mean structure, 
           minimizing RMSD.
SASA: Implements the Shrake-Rupley method using 96 equidistant points per atom generated 
      via a Golden Spiral to approximate the solvent surface.
PCA: Performs Singular Value Decomposition on the covariance matrix of 
     atomic coordinates to extract principal eigenvectors.
