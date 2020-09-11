#!/usr/bin/env python
"""Perform a molecular optimization using a set of smiles and CP2K."""
import argparse
import logging
import os
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import pkg_resources
from qmflows import Settings, cp2k, run, templates
from qmflows.parsers.xyzParser import string_to_plams_Molecule
from rdkit import Chem
from rdkit.Chem import AllChem
from scm.plams import Molecule, finish, from_rdmol, from_smiles, init, load
from scm.plams.recipes.adf_crs import (add_solvation_block, run_adfjob,
                                       run_crsjob)

from nanoqm.workflows.templates import generate_kinds

# Starting logger
logger = logging.getLogger(__name__)

toluene = """
15

  C        22.4837131643       24.7351093186       24.9999999099
  H        22.0766706594       24.2346804798       25.8886380697
  H        22.0766703819       24.2346803284       24.1113619649
  H        22.0956814738       25.7654947159       24.9999999416
  C        23.9949068071       24.7218576593       24.9999999594
  C        24.7164411450       24.7246225601       23.7958849915
  C        26.1149865405       24.7323308427       23.7927823070
  C        26.8205417677       24.7350767573       25.0000000373
  C        26.1149864756       24.7323318816       26.2072177337
  C        24.7164411313       24.7246219788       26.2041149500
  H        24.1741661846       24.7174801801       22.8519481371
  H        26.6538061918       24.7310962960       22.8476215525
  H        27.9083820666       24.7347183659       25.0000001876
  H        26.6538058044       24.7310954875       27.1523786452
  H        24.1741657719       24.7174810599       27.1480515695
"""

doe = """
51

  C        -8.8157539199       -0.5652846174       -1.4422851129
  C        -7.3816134120       -0.2605727486       -0.9849636069
  C        -6.3829759261       -1.3685219950       -1.3563160915
  C        -4.9437788308       -1.0684964531       -0.9097055199
  C        -3.9485567020       -2.1748195739       -1.2987252596
  C        -2.4816651521       -1.8679318678       -0.9505450533
  C        -2.1907261436       -1.8009554881        0.5577708761
  C        -0.7286261101       -1.5042213558        0.8765451610
  O         0.0803557412       -2.5925335148        0.4042782244
  C         1.4830571732       -2.3928556225        0.6317691356
  C         2.1365228892       -1.4620018311       -0.3998923995
  C         3.6646519274       -1.3447921667       -0.2476796532
  C         4.1216497535       -0.6043566050        1.0226299518
  C         5.6474044805       -0.4207171014        1.1301012334
  C         6.2408571259        0.5514179379        0.0953506874
  C         7.7473809282        0.8161417238        0.2727259780
  C         8.6363956667       -0.4079660789        0.0034818943
  H        -9.1913432356       -1.4944381375       -0.9920198955
  H        -9.5134716587        0.2380794365       -1.1675474396
  H        -8.8714368146       -0.6900714272       -2.5325057106
  H        -7.0457147675        0.6912730753       -1.4211728359
  H        -7.3628125247       -0.1039137310        0.1031567509
  H        -6.4043131305       -1.5289836745       -2.4447709398
  H        -6.7145786529       -2.3216199395       -0.9176354891
  H        -4.9278533199       -0.9099374955        0.1785097698
  H        -4.6139992931       -0.1145386763       -1.3489329901
  H        -4.2494759101       -3.1211584429       -0.8243409777
  H        -4.0273495678       -2.3538858449       -2.3819777833
  H        -1.8293902709       -2.6332122207       -1.3936325168
  H        -2.1858080205       -0.9155454709       -1.4162637529
  H        -2.8036208803       -1.0200853539        1.0328120741
  H        -2.4765054274       -2.7499986435        1.0339372960
  H        -0.4239350222       -0.5575935096        0.3982904960
  H        -0.5821678411       -1.3822473372        1.9644278745
  H         1.9352636943       -3.3929928394        0.5644442548
  H         1.6490710499       -2.0289110270        1.6592750720
  H         1.8885751613       -1.8484754565       -1.3990983658
  H         1.6865538696       -0.4607651757       -0.3384135405
  H         4.1173959182       -2.3479567810       -0.2638307793
  H         4.0637982030       -0.8283591988       -1.1323985434
  H         3.6304129596        0.3798058117        1.0613366936
  H         3.7653860059       -1.1471509968        1.9104180361
  H         5.8951019148       -0.0509806253        2.1369601951
  H         6.1370557582       -1.4016730960        1.0477605724
  H         6.0599725762        0.1761782557       -0.9220258520
  H         5.6968796101        1.5066033320        0.1607874175
  H         7.9328323731        1.1907174951        1.2897625091
  H         8.0464982221        1.6328629144       -0.4003540251
  H         8.4803114699       -0.7960897941       -1.0124770605
  H         8.4286929247       -1.2297699891        0.7012155398
  H         9.7028122449       -0.1601046907        0.1012712156
"""

o_xylene = """
18

  C        -0.8186808571        1.9418110099       -0.1628229717
  C         0.0390072634        0.7039528702       -0.0606226004
  C         1.4367095496        0.8046484059       -0.1246350089
  C         2.2555553693       -0.3248865407       -0.0332288049
  C         1.6744187061       -1.5841032000        0.1259082782
  C         0.2823868617       -1.6966833542        0.1913902986
  C        -0.5492769138       -0.5707090105        0.1002219596
  C        -2.0489776279       -0.7247503671        0.1733706560
  H        -0.2036824541        2.8426761090       -0.2809990949
  H        -1.4448182230        2.0764768100        0.7324351215
  H        -1.5068475125        1.8888331045       -1.0203716742
  H         1.8861237043        1.7881135783       -0.2484202640
  H         3.3368483108       -0.2201639801       -0.0860572740
  H         2.2972385672       -2.4728088739        0.1987895973
  H        -0.1741277396       -2.6767953778        0.3159121553
  H        -2.4752633873       -0.1533055790        1.0120739337
  H        -2.5401969099       -0.3538309916       -0.7392953912
  H        -2.3341032756       -1.7760618636        0.3040737074
"""

octadecene = """
 70
 
  C        -8.5270514446        1.6768601920        1.0600332442
  C        -7.3047137887        1.2608106158        0.7047479969
  C        -6.9926511041       -0.0721798009        0.0831986920
  C        -6.1224778484       -0.9922104838        0.9727372260
  C        -4.7547268531       -0.4116272539        1.3780280873
  C        -3.8306505720       -0.0091838409        0.2134555624
  C        -3.4139938153       -1.1773495468       -0.6939025909
  C        -2.5139467707       -0.7679712966       -1.8756372372
  C        -1.1782819945       -0.0978211345       -1.5004410163
  C        -0.2415139268       -0.9821268838       -0.6614781026
  C         1.0748032805       -0.2972015999       -0.2480379649
  C         2.0167927097        0.0320797822       -1.4188674313
  C         3.3250804726        0.7290762656       -1.0018132091
  C         4.2844794797       -0.1414188041       -0.1733579641
  C         5.6042866197        0.5730046747        0.1577609317
  C         6.5698088755       -0.2918997690        0.9842539957
  C         7.8770297032        0.4162663527        1.3842080396
  C         8.7874100006        0.7737414990        0.1990521895
  H        -8.6915252686        2.6512413914        1.5162506932
  H        -9.4078281867        1.0557532935        0.9112857452
  H        -6.4562603100        1.9209453119        0.8815912328
  H        -6.4750724174        0.0899162564       -0.8763288010
  H        -7.9318937645       -0.5896676543       -0.1586806750
  H        -6.6861604996       -1.2330208825        1.8859361678
  H        -5.9788575946       -1.9469075784        0.4462472689
  H        -4.2365527982       -1.1533975711        2.0051579393
  H        -4.9118803751        0.4623076105        2.0271752081
  H        -2.9308965428        0.4634896343        0.6345767870
  H        -4.3096600885        0.7710722127       -0.3971579947
  H        -2.9143697114       -1.9447901183       -0.0845791028
  H        -4.3120356337       -1.6670877807       -1.0992582352
  H        -3.0782985487       -0.0858033591       -2.5297943323
  H        -2.3063148560       -1.6593021393       -2.4872851297
  H        -1.3677562916        0.8399358547       -0.9580566761
  H        -0.6701978830        0.2002642465       -2.4292791647
  H        -0.7694260741       -1.3063913595        0.2475407185
  H        -0.0199963114       -1.9057749393       -1.2178028716
  H         0.8467262241        0.6296348909        0.3002552277
  H         1.5998344701       -0.9431938719        0.4707017620
  H         1.4900041659        0.6748844507       -2.1391126978
  H         2.2514623823       -0.8938869700       -1.9651018489
  H         3.0882249352        1.6452206739       -0.4400036768
  H         3.8528568160        1.0695615903       -1.9059002307
  H         4.4964025190       -1.0713179503       -0.7227260726
  H         3.7988925635       -0.4578597876        0.7610027215
  H         5.3915339225        1.5035471414        0.7057191401
  H         6.0884350212        0.8895195862       -0.7773361573
  H         6.8078636489       -1.2089071217        0.4247977879
  H         6.0517440749       -0.6290090929        1.8953601305
  H         8.4314149872       -0.2278109586        2.0824742107
  H         7.6366786170        1.3264598220        1.9523455107
  H         9.7282215551        1.2306964827        0.5373796250
  H         9.0471409539       -0.1191607584       -0.3862420368
  H         8.3093952435        1.4847315020       -0.4875246083
"""

# Solvent to compute properties
names = ["toluene", "o_xylene", "doe", "octadecene"]
solvents = {name: string_to_plams_Molecule(s)
            for name, s in
            zip(names, [toluene, o_xylene, doe, octadecene])}
# Add name to molecule
for name, mol in solvents.items():
    mol.properties.name = name

Result = namedtuple("Result", ['gammas', 'deltag'])


def set_logger():
    """Set logging default behaviour."""
    file_log = 'output.log'
    logging.basicConfig(filename=file_log, level=logging.DEBUG,
                        format='%(asctime)s---%(levelname)s\n%(message)s\n',
                        datefmt='[%I:%M:%S]')
    logging.getLogger("noodles").setLevel(logging.WARNING)


def store_results_in_df(smile: str, results: namedtuple, df_results: pd.DataFrame, path_results: str):
    """Store the computed properties in the results DataFrame."""
    cols = df_results.columns
    df_results.loc[(smile, "gammas"), cols] = results.gammas
    df_results.loc[(smile, "deltag"), cols] = results.deltag
    df_results.to_csv(path_results)


def store_optimized_molecule(smile: str, optimized_geometry: Molecule, path_optimized: str):
    """Store the xyz molecular geometry."""
    path_smile = f"{path_optimized}/{smile}"
    if not os.path.exists(path_smile):
        os.mkdir(path_smile)
    with open(f"{path_smile}/geometry.xyz", 'w') as f:
        optimized_geometry.writexyz(f)

    
def compute_properties(smile: str, adf_solvents: dict, workdir: str, path_optimized: str) -> np.array:
    """Compute properties for the given smile and solvent."""
    # Create the CP2K job
    job_cp2k = create_job_cp2k(smile, smile, workdir)

    # Run the cp2k job
    optimized_geometry = run(job_cp2k.geometry, folder=workdir)
    store_optimized_molecule(smile, optimized_geometry, path_optimized)
    logger.info(f"{smile} has been optimized with CP2K")
    
    # Create the ADF JOB
    crs_dict = create_crs_job(smile, optimized_geometry, adf_solvents, workdir)

    # extract results
    deltag = pd.Series({name: try_to_readkf(results, "deltag")
                        for name, results in crs_dict.items()})
    gammas = pd.Series({name: try_to_readkf(results, "gamma")
                        for name, results in crs_dict.items()})

    return Result(gammas, deltag)


def create_job_cp2k(smile: str, job_name: str, workdir: str) -> object:
    """Create a CP2K job object."""
    # Set path for basis set
    path_basis = pkg_resources.resource_filename("nanoqm", "basis/BASIS_MOLOPT")
    path_potential = pkg_resources.resource_filename(
        "nanoqm", "basis/GTH_POTENTIALS")

    # Settings specifics
    s = Settings()
    s.basis = "DZVP-MOLOPT-SR-GTH"
    s.potential = "GTH-PBE"
    s.cell_parameters = 5
    s.specific.cp2k.force_eval.dft.basis_set_file_name = path_basis
    s.specific.cp2k.force_eval.dft.potential_file_name = path_potential

    # functional
    s.specific.cp2k.force_eval.dft.xc["xc_functional"] = {}

    # Molecular geometry
    system_name = try_to_optimize_FF(smile, workdir)
    mol = Molecule(system_name)
    
    # Generate kinds for the atom types
    elements = [x.symbol for x in mol.atoms]
    kinds = generate_kinds(elements, s.basis, s.potential)

    # Update the setting with the kinds
    sett = templates.geometry.overlay(s)
    sett.specific = sett.specific + kinds

    return cp2k(sett, mol, job_name="cp2k_opt")


def try_to_optimize_FF(smile: str, workdir: str) -> Molecule:
    """Try to optimize the molecule with a force field."""
    try:
        # Try to optimize with RDKIT
        mol = Chem.MolFromSmiles(smile)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        mol = from_rdmol(mol)
    except:
        mol = from_smiles(smile)

    file_path = f"{workdir}/tmp.xyz"
    with open(file_path, 'w') as f:
        mol.writexyz(f)

    # Pass a path otherwise there is a problem trying to copy
    return file_path


def create_setting_adf() -> Settings:
    """Create setting to run a solvation job."""
    settings_adf = Settings()
    settings_adf.input.basis.type = 'TZ2P'
    settings_adf.input.xc.gga = 'PBE'
    settings_adf.input.scf.converge = '1.0e-06'

    settings_adf = add_solvation_block(settings_adf)

    return settings_adf


def create_crs_job(smile: str, optimized_geometry: Molecule, adf_solvents: dict, workdir: str) -> object:
    """Create a Single Point Calculation with ADF on geometries optimized with CP2k."""
    # CRS job setting
    settings_crs = Settings()
    settings_crs.input.temperature = 298.15
    settings_crs.input.property._h = 'activitycoef'

    # ADF Settings
    settings_adf = create_setting_adf()

    settings_adf.runscript.pre = "export SCM_TMPDIR=$PWD"
    settings_crs.runscript.pre = "export SCM_TMPDIR=$PWD"

    # ADF CRS JOB
    init(workdir)
    solute = run_adfjob(optimized_geometry, settings_adf)
    crs_dict = {name: run_crsjob(solvent, settings_crs, solute=solute)
                for name, solvent in adf_solvents.items()}
    finish()

    return crs_dict


def try_to_readkf(results: object, property_name: str):
    """Try to read the output from a KF binary file."""
    try:
        results = results.readkf("ACTIVITYCOEF", property_name)[1]
    except (KeyError, TypeError):
        results = None

    return results


def create_dataframe(path: str) -> pd.DataFrame:
    """Read csv file if path exists otherwise create empty dataframe."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.set_index(["smile", "property"], drop=True, inplace=True)
        return df
    else:
        # Create a multiindex DataFrame to store the compute properties for each smile/solvent
        idx = pd.MultiIndex.from_tuples([], names=["smile", "property"])
        return pd.DataFrame(columns=solvents.keys(), index=idx)


def read_or_compute_solvents(workdir: str):
    """Read solvents data or compute it from scratch."""
    path_solvents = "adf_solvents"

    if not os.path.exists(path_solvents):
        os.mkdir(path_solvents)
        init(path_solvents)
        # Run the ADF job
        settings_adf = create_setting_adf()
        adf_solvents = {name: run_adfjob(mol, settings_adf)
                        for name, mol in solvents.items()}
        finish()
    else:
        init(workdir)
        logger.info(
            f"Solvents are already computed and available at: {path_solvents}")
        path = Path(path_solvents) / "plams_workdir"
        dill_files = {x: next(path.glob(f"*.{x}/*.dill")).as_posix()
                      for x in names}
        adf_solvents = {name: load(
            file_name).results for name, file_name in dill_files.items()}
        finish()

    return adf_solvents


def main(file_path: str, workdir: str):
    """Run script."""
    set_logger()
    # Read input smiles
    df_smiles = pd.read_csv(file_path)
    # Path results
    path_results = "results.csv"
    logger.info(f"Results are stored at: {path_results}")
    # Read the database file o create new db
    df_results = create_dataframe(path_results)

    # Create workdir if it doesn't exist
    if not os.path.exists(workdir):
        os.mkdir(workdir)

    # Create folder to store the cp2k optimize molecules
    path_optimized = "optimized_molecules"
    if not os.path.exists(path_optimized):
        os.mkdir(path_optimized)
        
    # Read the solvents data if available otherwise compute it from scratch
    adf_solvents = read_or_compute_solvents(workdir)

    # Compute the properties
    for smile in df_smiles["smiles"]:
        logger.info(f"computing: {smile}")
        if smile in df_results.index:
            logger.info(f"properties of {smile} are already store!")
        else:
            results = compute_properties(smile, adf_solvents, workdir, path_optimized)
            store_results_in_df(smile, results, df_results, path_results)


if __name__ == "__main__":
    """Parse the command line arguments and call the modeller class."""
    parser = argparse.ArgumentParser(
        description="smiles_calculator.py -i smiles.csv")
    # configure logger
    parser.add_argument('-i', '--input', required=True,
                        help="Input file with the smiles")
    parser.add_argument('-w', '--workdir', default="/tmp/adf")
    args = parser.parse_args()

    main(args.input, args.workdir)
