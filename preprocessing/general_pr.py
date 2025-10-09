import pandas as pd
import logging
logger = logging.getLogger(__name__)


def parse_header_to_acc_locus_and_name(df: pd.DataFrame, df_header_col: str, acc_col: str, locus_col: str = "locus_tag", name_col: str = "name") -> pd.DataFrame:
    """
    Parse the header column in the dataframe to extract accession, locus, and name.
    
    Args:
        df (pd.DataFrame): DataFrame containing the header column.
        df_header_col (str): Name of the column containing the header.
        acc_col (str): Name of the column to store the accession.
        locus_col (str): Name of the column to store the locus.
        name_col (str): Name of the column to store the name.
    
    Returns:
        pd.DataFrame: DataFrame with the new columns added.
    """
    def parse_header(header: str):
        parts = header.split('|')
        if len(parts) == 4:
            return parts[1], parts[2], parts[3]
        return None, None, None

    df[[acc_col, locus_col, name_col]] = df[df_header_col].apply(lambda x: pd.Series(parse_header(x)))
    return df


def convert_count_to_val(count: int, denominator: int, val_type: str):
        """
        Args:
            count (int):
            denominator (int): denominator
            val_type (str): 'ratio' or 'percentage'
        Returns:
            float or int: desired val
        """
        if val_type == 'ratio':
            val = float(round(count/denominator, 2))
        elif val_type == 'percentage':
            val = int(round(count/denominator, 2)*100)
        else:
            raise ValueError(f"val_type {val_type} is not supported")
        return val

