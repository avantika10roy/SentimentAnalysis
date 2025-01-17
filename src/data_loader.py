# Dependencies
import pandas as pd

def load_csv_data(filepath: str) -> pd.DataFrame:
    """
    Load the CSV dataset

    Arguments:
    ----------
        filepath { str } : 

    Errors:
    -------
        DataLoadingError : 

    Returns:
    --------
        { DataFrame }    : 
    """
    try:
        dataframe = pd.read_csv(filepath_or_buffer = filepath,
                                index_col          = None)

        return dataframe

    except Exception as DataLoadingError:
        raise RuntimeError(f"Error loading data: {repr(DataLoadingError)}")
