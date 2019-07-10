from dataCAT import Database
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="cosmos -i file_smiles")
    parser.add_argument('-csv', required=True,
                        help="Input file in with the smiles")
    parser.add_argument('-uri', help="URI to access the mongodb", default='localhost')
    parser.add_argument('-a', '--anchor', help="anchor group", default='O1')

    # read cmd line options
    args = parser.parse_args()

    # Save the data
    save_data_in_mongodb(args)


def save_data_in_mongodb(args: dict) -> None:
    """
    Store the data in a csv file in a MongoDB instance
    """
    # Read the .csv file
    df = pd.read_csv(args.csv)

    # Set DataFrame.index
    df['smiles'] = df.pop('Unnamed: 0')
    df['anchor'] = args.anchor
    df.set_index(['smiles', 'anchor'], inplace=True)

    # Set DataFrame.columns
    df.columns = pd.MultiIndex.from_tuples([
        ('E_solv', 'Toluene'),
        ('gamma', 'Toluene')
    ])

    # Create a dataCAT.Database instance
    kwargs = {
        'host': args.uri,
        'port': 27017
    }
    db = Database(**kwargs)

    # Update the database
    db.update_mongodb({'ligand': df})
