"""
Data loading functionality for Trimmed Match marketing experiments.

This module handles the loading of data from various sources including
Google Sheets, local CSV files, and pandas DataFrames.
"""

import pandas as pd
import gspread
from google import auth as google_auth
from gspread_dataframe import set_with_dataframe
import os


def authenticate_google():
    """Authenticate with Google services."""
    return google_auth.default()


def get_gspread_client(creds=None):
    """Get a gspread client.
    
    Args:
        creds: Google credentials. If None, authenticate first.
    
    Returns:
        gspread client
    """
    if creds is None:
        creds, _ = authenticate_google()
    return gspread.authorize(creds)


def read_trix(url, client=None):
    """Read data from a Google Sheet.
    
    Args:
        url: URL to the Google Sheet
        client: Authenticated gspread client. If None, one will be created.
    
    Returns:
        pandas DataFrame with the sheet contents
    """
    if client is None:
        client = get_gspread_client()
        
    wks = client.open_by_url(url).sheet1
    data = wks.get_all_values()
    headers = data.pop(0)
    return pd.DataFrame(data, columns=headers)


def write_trix(df, url, sheet_name="Sheet1", client=None):
    """Write a DataFrame to a Google Sheet.
    
    Args:
        df: pandas DataFrame to write
        url: URL to the Google Sheet
        sheet_name: Name of the sheet to write to
        client: Authenticated gspread client. If None, one will be created.
    """
    if client is None:
        client = get_gspread_client()
        
    workbook = client.open_by_url(url)
    try:
        worksheet = workbook.worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = workbook.add_worksheet(sheet_name, rows=df.shape[0] + 10, cols=df.shape[1] + 5)
    
    worksheet.clear()
    set_with_dataframe(worksheet, df)
    
    return worksheet


def read_csv(file_path, **kwargs):
    """Read data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        **kwargs: Additional arguments to pass to pandas.read_csv
    
    Returns:
        pandas DataFrame with the file contents
    """
    df = pd.read_csv(file_path, **kwargs)
    
    # Convert columns to appropriate types
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    if 'geo' in df.columns:
        df['geo'] = pd.to_numeric(df['geo'], errors='coerce')
    
    if 'response' in df.columns:
        df['response'] = pd.to_numeric(df['response'], errors='coerce')
    
    if 'cost' in df.columns:
        df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
        
    return df


def write_csv(df, file_path, **kwargs):
    """Write data to a CSV file.
    
    Args:
        df: pandas DataFrame to write
        file_path: Path to the CSV file
        **kwargs: Additional arguments to pass to pandas.to_csv
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, **kwargs)


def prepare_dataframe(df):
    """Prepare a DataFrame for use with Trimmed Match.
    
    Args:
        df: pandas DataFrame to prepare
    
    Returns:
        Prepared pandas DataFrame
    """
    df = df.copy()
    
    # Convert columns to appropriate types
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    if 'geo' in df.columns:
        df['geo'] = pd.to_numeric(df['geo'], errors='coerce')
    
    if 'response' in df.columns:
        df['response'] = pd.to_numeric(df['response'], errors='coerce')
    
    if 'cost' in df.columns:
        df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
        
    return df


def load_data(source, **kwargs):
    """Load data from various sources.
    
    Args:
        source: Source of the data. Can be a pandas DataFrame, a path to a CSV file,
               or a URL to a Google Sheet.
        **kwargs: Additional arguments to pass to the specific loader
    
    Returns:
        pandas DataFrame with the loaded data
    """
    if isinstance(source, pd.DataFrame):
        return prepare_dataframe(source)
    elif isinstance(source, str):
        if source.startswith('http') and ('docs.google.com' in source or 'spreadsheets.google.com' in source):
            return prepare_dataframe(read_trix(source, **kwargs))
        elif os.path.isfile(source) and source.endswith('.csv'):
            return read_csv(source, **kwargs)
        else:
            raise ValueError(f"Unable to determine data source type for: {source}")
    else:
        raise TypeError(f"Unsupported data source type: {type(source)}")
