# Experimental and computational protein structure methods

X-ray crystallography needs the protein to form a well-ordered crystal; diffraction patterns are turned
into an electron-density map. It gives high resolution but crystallisation can take months and fails for
many membrane proteins.

Cryo-electron microscopy (cryo-EM) images flash-frozen particles and averages many thousands of views.
It handles large complexes and membrane proteins without crystals. NMR spectroscopy works in solution and
captures dynamics, but is mostly limited to smaller proteins.

Computational methods are far cheaper and faster. Secondary-structure predictors classify each residue as
helix, strand or coil (Q3 accuracy measures this three-class accuracy). Homology modelling copies the fold
of a related protein with a known structure. Deep-learning systems such as AlphaFold2 now predict many
structures with accuracy close to experiment, and validation tools such as Ramachandran plots check
whether backbone angles are physically reasonable.
