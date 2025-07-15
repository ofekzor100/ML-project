def prepare_data(training_data, new_data):
  import pandas as pd
  from sklearn.preprocessing import StandardScaler, MinMaxScaler

  new_data_copy = new_data.copy()

  #  fill the missing values
  household_income_median = training_data.household_income.median()
  PCR_02_median = training_data.PCR_02.median()
  new_data_copy = new_data_copy.fillna(value={"household_income": household_income_median, "PCR_02": PCR_02_median})

  # replace blood_type with boolean SpecialProprety
  new_data_copy["SpecialProperty"] = new_data_copy["blood_type"].isin(["O+","B+"])
  del new_data_copy["blood_type"]

  # scale PCR features
  standard_scaler = StandardScaler()
  minmax_scaler = MinMaxScaler(feature_range=(-1,1))

  for metric in ["PCR_01", "PCR_03", "PCR_04", "PCR_06", "PCR_07", "PCR_08", "PCR_09"]:
    new_data_copy[metric] = minmax_scaler.fit_transform(new_data_copy[[metric]])

  for metric in ["PCR_02", "PCR_05", "PCR_10"]:
    new_data_copy[metric] = standard_scaler.fit_transform(new_data_copy[[metric]])

  return new_data_copy