import logging

import pandas as pd
from rdkit.Chem import AllChem

# Starting logger
LOGGER = logging.getLogger(__name__)


def sanitize_data(data: pd.DataFrame) -> pd.DataFrame:
    """Check that the data in the DataFrame is valid."""
    # discard nan values
    data.dropna(inplace=True)

    # Create conformers
    data['molecules'].apply(lambda mol: AllChem.EmbedMolecule(mol))

    # Discard molecules that do not have conformer
    LOGGER.info("Removing molecules that don't have any conformer.")
    data = data[data['molecules'].apply(lambda x: x.GetNumConformers()) >= 1]

    return data
