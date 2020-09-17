In silico Screening
===================
This tutorial covers how to perform insilico filtering of a set of molecules
represented as smiles_. Table :numref:`smiles-table` contains some smiles
examples representing molecules with different functional groups.

.. _smiles-table:

.. list-table:: smiles to filter
   :header-rows: 1		

 * - Smiles		 
 * - CN1C=NC2=C1C(=O)N(C(=O)N2C)C
   - OC(=O)C1CNC2C3C4CC2C1N34
   - C1=CC=CC=C1
   - OC(=O)C1CNC2COC1(C2)C#C
   - CCO
   - CCCCCCCCC=CCCCCCCCC(=O)O
   - CC(=O)O
   - O=C(O)Cc1ccccc1
   - CC(C(=O)O)O


The filtering process consists in excluding (or including) a set of
molecules based on structural characteristics like their functional
groups or derived properties like bulkiness.

To run the simulation, the user must provide two files: one containing the
smiles that she wants to filter and another file containing
the values of the properties used as filters. 


Simulation input
****************
The smiles input should be in csv format like ::

  ,smiles
  ,CC(=O)O
  ,CCC(=O)O


The properties specification file to perform the filtering must be a yaml
file following the subsequent schema yaml_::

 smiles_file:
   smiles.csv

  core:
    "Cd68Se55.xyz"

  anchor:
    "O(C=O)[H]"

  batch_size: 1000
    
  filters:
    include_functional_groups:
      - "C(=O)O"
    exclude_functional_groups:
      - "S(=O)(=O)"
    scscore:
      lower_than:
        3.0
    bulkiness:
      lower_than:
        200


The *smiles_file* entry contains the path to the files containing the smiles. The
other keywords will be explain in the following sections.
	
Available filters
*****************

Include and exclude function groups
-----------------------------------
The *include_functional_groups* and *exclude_functional_groups* as their names suggest
keep and drop molecules based on a list of functional groups. Notice the functional
are also provided as smiles_.

Synthesizability scores
-----------------------
The scscore_ is a measure of synthetic complexity. It is scaled from 1 to 5
to facilited human interpretation. See the scscore_ paper for further details.


Bulkiness
---------
Assuming that a given molecule can be attached to a given surface, the bulkiness
descriptor gives a measure of the volumen occupied by the molecule from the
anchoring point extending outwards as a cone. It requires the *core* keywords
specifying the surface to attach the molecule and the *anchor* functional
group used as attachment.
See the `the CAT bulkiness <https://cat.readthedocs.io/en/latest/4_optional.html?highlight=bulkiness#optional.qd.bulkiness>`_
for further information.

	
Running the filtering script
****************************
To perform the screening you just need to execute the following command ::
  smiles_screener -i path_to_yaml_input.yml


Job distributions and results
*****************************


.. _smiles: https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system
.. _yaml: https://yaml.org/
.. _scscore: https://pubs.acs.org/doi/10.1021/acs.jcim.7b00622
