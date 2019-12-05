"""Interface to the SQL databases."""
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import Column, Float, MetaData, String, Table, create_engine
from sqlalchemy.engine import Engine


class SQLCommunicator:
    """Interface to a SQL database."""

    def __init__(self, engine: Engine):
        """Create communication with a given engine."""
        self.engine = engine

    def create_ligand_table(self, data: pd.DataFrame, table_name: str) -> Table:
        """Create a table with the properties store in the DataFrame."""
        if data.index.name != 'smiles':
            data.set_index('smiles', inplace=True)
        metadata = MetaData()
        columns = (Column(name, Float) for name in data.columns)
        properties = Table(table_name, metadata,
                           Column('smiles', String, primary_key=True),
                           *columns)
        metadata.create_all(self.engine)

        return properties


def creeate_postgres_engine(
        passwd: str, database: str = "quantumdb", user: str = "postgres", host: str = "localhost") -> Engine:
    """Generate a SQLAlchemy instance using postgres."""
    url_passwd = quote_plus(passwd)
    return create_engine(f'postgresql://{user}:{url_passwd}@{host}/{database}')
