"""Getting clean dataframe from the Stanford audiological database that is ready for transfer to other clinics or researchers."""

"""Enhance the clinical data dump by
- Replacing the MRN with a HMAC code
- Classifying the type of hearing loss (sensorineural, conductive, mixed)
- Enhance the dataset with some pure-tone averages (3-freq PTA, 4-freq, HFPTA)
- Replace NR values with a million to ensure it is not included in calculations of PTA, and to caution the user regarding no responses values at a particular frequency
- Flag the patients with multiple visits

Finally, write out the new data."""

import pandas as pd
from absl import app
from absl import flags

import replace_MRN
import hearingloss_classifier
import format_date

def classify_hearing_loss(df: pd.DataFrame):
  """Assign result to a temporary dataframe, since classifier creates lots of temporary column names."""
  orig_df = df.copy()
  new_df = hearingloss_classifier.HearingClassifier(df)
  df = orig_df

  # Specify columns related to bone conduction, hearing loss types, and PTA to be included in output file
  bc_columns = ['RBone500', 'RBone1000', 'RBone2000', 'RBone4000',
          'LBone500', 'LBone1000', 'LBone2000', 'LBone4000']

  hl_type_columns = ['R_Type_HL', 'R_Type_HL_HF', 'R_Type_HL_4freq',
             'L_Type_HL', 'L_Type_HL_HF', 'L_Type_HL_4freq']

  # Add PTA columns from the classifier also here
  pta_columns = ['R_PTA', 'R_PTA_4freq', 'R_HFPTA',
           'L_PTA', 'L_PTA_4freq', 'L_HFPTA']

  # Combine the selected columns to be transferred
  selected_columns = bc_columns + hl_type_columns + pta_columns

  # Update the original dataframe with the new values
  df[selected_columns] = new_df[selected_columns]

  return df

def label_duplicates(df: pd.DataFrame) -> pd.DataFrame:
  """Label multiple visits for each patient."""
  duplicate_name = 'MultipleVisits'
  df[duplicate_name] = False
  df.loc[df['Patients::HMAC'].duplicated(
    keep=False), 'MultipleVisits'] = True
  return df


FLAGS = flags.FLAGS
flags.DEFINE_string('input',
                    '/Users/malcolm/Downloads/3-27-2024 raw data no cleaning.csv',
                    'Input CSV filename')
flags.DEFINE_string('output',
                    'All_Clean_Audiology_Data.csv',
                    'Output CSV filename')
flags.DEFINE_string('hmac_key',
                    None,
                    'Key to use when hashing the MRN.',
                    required=True)
flags.DEFINE_string('cluster_dir', 'ClusterData_v1',
                    'Where to find the pretrained cluster json data')
flags.DEFINE_string('convert_mrns', None,
                    'Which file to read MRNs from to convert to hashes')

def main(argv):
  """Get final clean dataframe with hashed MRNs, degree, type of HL."""
  if FLAGS.convert_mrns:
    # Just convert a list of MRNs (in a file) into their hash IDs.
    with open(FLAGS.convert_mrns, 'r') as fp:
      # Replace the column header
      print(fp.readline().strip().replace('MRN', 'HMAC'))
      for line in fp:
        line = line.strip()
        print(replace_MRN.create_hmac(line, FLAGS.hmac_key))

  df = pd.read_csv(FLAGS.input,
           on_bad_lines='warn',
           na_values=['\n',
                '? * Line 1, Column 1\n  Syntax error: value, '
                'object or array expected.\n* Line 1, Column 1\n'
                '  A valid JSON document must be either an array '
                'or an object value.'],
           low_memory = False) # Added low_memory = False to get rid of DType warning

  df.replace('NR', 1000000, inplace=True)

  for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors='ignore')

  print('Column names:', df.columns.tolist())

  df = replace_MRN.replace_mrn(df, FLAGS.hmac_key)
  df = classify_hearing_loss(df)
  df = label_duplicates(df)

##### VMA -- Added a few more cases to make it cleaner
  ### Remove abstracted cases
  df = df[~df['ClinicianFullName'].str.contains('Abstracted', na=False)]
  # df = df[~df['TestLocation'].str.contains('Abstracted', na=False)]

  # Convert 'DateOfTest' column to datetime using custom function
  df['DateOfTest'] = df['DateOfTest'].apply(format_date.convert_to_datetime)
  # Drop rows with NaT values (invalid dates)
  df = df.dropna(subset=['DateOfTest'])

  # Renaming variables
  df.rename(columns={'Patients::Gender': 'Gender'}, inplace=True)
  df['Gender'] = df['Gender'].str.strip()

  # Drop undesired columns
  df = df.drop(columns = ['HMAC_code.1', 'DuplicateAudioDelete',
                          'PTA_L_AC_500_1k_2k_4k', 'PTA_R_AC_500_1k_2k_4k'])
  # Drop the first column - it's only an index
  # df = df.drop(df.columns[[0]], axis=1, inplace=True)

  # Reordering columns
  begin_columns = ['Patients::HMAC', 'DateOfTest']
  ssq_columns = sorted([col for col in df.columns if col.startswith('SSQ::')])
  hlq_columns = sorted([col for col in df.columns if col.startswith('HLQ::')])
  other_columns = [col for col in df.columns if col not in begin_columns + ssq_columns + hlq_columns]
  new_column_order = begin_columns + other_columns + ssq_columns + hlq_columns
  df = df[new_column_order]

  df.to_csv(FLAGS.output)

  num_multiples = len(set(df.loc[df.MultipleVisits]['Patients::HMAC']))
  num_patients = len(set(df['Patients::HMAC']))
  num_visits = len(df['Patients::HMAC'])
  print(f'Total # Records: {num_visits}, # Patients: {num_patients}, '
      f'# Multiple Visit Patients: {num_multiples}')

  return

if __name__ == '__main__':
  app.run(main)
