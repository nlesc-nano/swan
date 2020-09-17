In silico Screening
===================
This tutorial covers how to perform insilico filtering of a set of molecules
represented as smiles_. Table :ref:`smiles-table` contains some smiles
examples representing molecules with different functional groups.

.. _smiles-table:

.. csv-table:: smiles to filter
   :header: "smiles"

   CN1C=NC2=C1C(=O)N(C(=O)N2C)C
   OC(=O)C1CNC2C3C4CC2C1N34
   C1=CC=CC=C1
   OC(=O)C1CNC2COC1(C2)C#C
   CCO
   CCCCCCCCC=CCCCCCCCC(=O)O
   CC(=O)O
   O=C(O)Cc1ccccc1
   CC(C(=O)O)O


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

.. Note:: The filters are run in sequential order, meaning that second filter is applied
   to the set of molecules remaining after applying the first filters, the third
   filter is applied after the second and so on.


1. Include and exclude function groups
--------------------------------------
The *include_functional_groups* and *exclude_functional_groups* as their names suggest
keep and drop molecules based on a list of functional groups. Notice the functional
are also provided as smiles_.

2. Synthesizability scores
--------------------------
The scscore_ is a measure of synthetic complexity. It is scaled from 1 to 5
to facilited human interpretation. See the scscore_ paper for further details.


3. Bulkiness
------------
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
For a given filter, **Swan** will try to compute the molecular properties in parallel since properties
can be computed independently for each molecule. Therefore **Swan** split the molecular set
into batches that can be computed in parallel. The `batch_size` keyword is used to control the
size of these batches.

After the computation has finished the filtered molecules are stored in the **results** folder
in the *current work directory*. In that folder you can find a `candidates.csv` file for
each batch containing the final molecules.

.. _smiles: https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system
.. _yaml: https://yaml.org/
.. _scscore: https://pubs.acs.org/doi/10.1021/acs.jcim.7b00622
