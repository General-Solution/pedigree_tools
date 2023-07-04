# pedigree_tools
`pedigree_tools` is a python package containing useful methods for working with `msprime` pedigrees. 
For now, it is meant to be installed by placing it in your working directory and using `import pedigree_tools` in your python script.
Features include:
* `generate_pedigree(num_probands=None, population_size=None, end_time=None, num_pedigrees=None, random_seed=None)`: <br>
  Generates one or more pedigrees with the specified number of probands, out of a population with the specified size, going back `end_time` generations.
* `get_proband_genealogy(pedigree, chosen_probands)`: <br>
  Returns a pedigreee trimmed to only the ancestors of the chosen probands.
* `get_descendents_genealogy(pedigree, ancestral_individuals)`: <br>
  Finds the proband descendents of the chosen ancestors, and returns a pedigree trimmed to the ancestors of those probands
* `predict_contributions_coalescences(pedigree)`: <br>
  Predicts the expected genetic contribution of each individual to the probands, as well as the expected rate of coalescence in each individual.
  
