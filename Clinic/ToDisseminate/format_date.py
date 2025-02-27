import pandas as pd

def convert_to_datetime(date_str):
  """# Custom function to convert date strings to datetime objects."""
  try:
      return pd.to_datetime(date_str, format='%m/%d/%Y')
  except (ValueError, TypeError):
      return pd.NaT  # Return NaT for invalid dates