"""Read and assemble the five real datasets used by the fair GP experiments.

The read_and_treat_* functions retain their historical [X, y, S, names]
interface. load_dataset removes the sensitive feature from X and appends S
as the final input column, records categorical feature indices, and returns
read-only arrays in original reader row order.

Law retains all source rows. Its sensitive label is 1 for White and 0 for
Black, Asian, Hispanic, Mexican, Other, Puertorican, and Amerindian. Missing
or unrecognized race labels raise an error.
"""

from functools import lru_cache
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_READERS = {
    "crime": ("read_and_treat_crime_data", "sensitive"),
    "compas": ("read_and_treat_compas_data", "race"),
    "student": ("read_and_treat_student_data", "sex"),
    "law": ("read_and_treat_law_data", "race"),
    "adult": ("read_and_treat_adult_data", "Sex"),
}

CATEGORICAL_NAMES = {
    "compas": {"event"},
    "student": {"school", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason",
                "guardian", "schoolsup", "famsup", "paid", "activities", "nursery",
                "higher", "internet", "romantic"},
    "adult": {"OrigEthn", "Workclass", "Martial Status", "Occupation", "Child"},
}

DATA_FILES = {
    "crime": ("attributes.csv", "communities.data"),
    "compas": ("compas-scores-two-years.csv",),
    "student": ("student.csv",),
    "law": ("law_data.csv",),
    "adult": ("adult.data.csv", "adult.test.csv"),
}


def read_and_treat_adult_data(DirectoryName, SensitiveVarName='Sex', display=False, *,
                             categorical_codes=False):
    """
    Read and treat the adult census data as in [Besse et al., The American Statistician, 2021]
    -> DirectoryName is the directory in which the files "adult.data.csv" and "adult.test.csv" are located
    ->  SensitiveVarName is the string representing the sensitive variable in "adult.data.csv" and "adult.test.csv"
    Return [X, y, S, feature_names] with test rows before training rows.
    Set categorical_codes=True for one numeric code per nominal field.
    Missing nominal values receive code -1.
    """

    #read and merge original data
    original_data_train = pd.read_csv(
        os.path.join(DirectoryName, "adult.data.csv"),
        names=[
            "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
            "Occupation", "Relationship", "OrigEthn", "Sex", "Capital Gain", "Capital Loss",
            "Hours per week", "Country", "Target"],
            sep=r'\s*,\s*',
            engine='python',
            na_values="?")

    original_data_test = pd.read_csv(
        os.path.join(DirectoryName, "adult.test.csv"),
        names=[
            "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
            "Occupation", "Relationship", "OrigEthn", "Sex", "Capital Gain", "Capital Loss",
            "Hours per week", "Country", "Target"],
            sep=r'\s*,\s*',
            engine='python',
            na_values="?")

    original_data = pd.concat([original_data_test,original_data_train])
    original_data.reset_index(inplace = True, drop = True)
    #print(original_data.tail())

    #data preparation 1/3
    data=original_data.copy()

    data['Child'] = np.where(data['Relationship']=='Own-child', 'ChildYes', 'ChildNo')
    data['OrigEthn'] = np.where(data['OrigEthn']=='White', 'CaucYes', 'CaucNo')
    data=data.drop(columns=['fnlwgt','Relationship','Country','Education'])
    data=data.replace('<=50K.','<=50K')
    data=data.replace('>50K.','>50K')

    #print(original_data.tail())

    #data preparation 2/3
    data_ohe=data.copy()
    data_ohe['Target'] = np.where(data_ohe['Target']=='>50K', 1., 0.)
    #print(' -> In column Target: label >50K gets 1.')
    data_ohe['OrigEthn'] = np.where(data_ohe['OrigEthn']=='CaucYes', 1., 0.)
    #print(' -> In column '+str('OrigEthn')+': label '+str('CaucYes')+' gets 1.')

    data_ohe['Sex'] = np.where(data_ohe['Sex']=='Male', 1., 0.)
    #print(' -> In column '+str('Sex')+': label '+str('Male')+' gets 1.')

    for col in ['Workclass', 'Martial Status', 'Occupation', 'Child']:
        if categorical_codes:
            data_ohe[col] = pd.Categorical(data_ohe[col]).codes
        elif len(set(list(data_ohe[col])))==2:
            LabelThatGets1=data_ohe[col][0]
            data_ohe[col] = np.where(data_ohe[col]==LabelThatGets1, 1., 0.)
            #print(' -> In column '+str(col)+': label '+str(LabelThatGets1)+' gets 1.')
        else:
            #print(' -> In column '+str(col)+': one-hot encoding conversion with labels '+str(set(list(data_ohe[col]))))
            data_ohe=pd.get_dummies(data_ohe,prefix=[col],columns=[col])

    #print(data_ohe.tail())

    #data preparation 3/3
    #... extract the X and y np.arrays
    y=data_ohe['Target'].values.reshape(-1,1)

    data_ohe_wo_target=data_ohe.drop(columns=['Target'])

    X_col_names=list(data_ohe_wo_target.columns)
    X=data_ohe_wo_target.values
    S=X[:,X_col_names.index(SensitiveVarName)].ravel()
    return [X, y, S, X_col_names]

def read_and_treat_law_data(DirectoryName, SensitiveVarName="race", display=False):
    """Read Law data with White=1 and all known non-White races=0.

    Retain every source row and return [X, y, S, feature_names]. The target
    is ZFYA and the features are race, LSAT, and UGPA. Missing, empty, and
    unrecognized race labels raise ValueError. The source label "Other"
    is a known non-White category.
    """
    data = pd.read_csv(os.path.join(DirectoryName, "law_data.csv"))
    data = data.loc[:, ["race", "LSAT", "UGPA", "ZFYA"]].copy()
    race = data["race"].astype("string").str.strip().str.casefold()
    if race.isna().any() or race.eq("").any():
        raise ValueError("Law data contain missing race labels. Supply a known race for every row.")
    known_races = {"white", "black", "asian", "hispanic", "mexican", "other",
                   "puertorican", "amerindian"}
    unknown = sorted(set(race) - known_races)
    if unknown:
        raise ValueError(f"Law data contain unrecognized race labels: {unknown}.")
    data["race"] = race.eq("white").astype(np.float64)
    y = data["ZFYA"].to_numpy().reshape(-1, 1)
    features = data.drop(columns=["ZFYA"])
    feature_names = list(features.columns)
    x = features.to_numpy()
    sensitive = x[:, feature_names.index(SensitiveVarName)].ravel()
    return [x, y, sensitive, feature_names]


def read_and_treat_crime_data(DirectoryName,SensitiveVarName='sensitive', display=False):
    """
    -> DirectoryName is the directory in which the file "communities.data" is located
    ->  SensitiveVarName is the string representing the sensitive variable
    Return [X, y, S, feature_names].
    """

    #read and merge original data
    attrib = pd.read_csv(os.path.join(DirectoryName, 'attributes.csv'), sep=r"\s+")
    data = pd.read_csv(os.path.join(DirectoryName, "communities.data"), names = attrib['attributes'])
    
    #remove non-predictive features
    data = data.drop(columns=['state','county','community','communityname','fold'], axis=1)
    
    #handle missing values
    data = data.replace('?', np.nan)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')#, axis = 0
    imputer.fit(data[['OtherPerCap']])
    data[['OtherPerCap']] = imputer.transform(data[['OtherPerCap']])
    data = data.dropna(axis=1)
    #data = data[data['OtherPerCap'].notna()]
    
    #create the sensitive variable
    data.loc[data['racepctblack'] >= 0.07, "sensitive"] = 0
    data.loc[data['racepctblack'] < 0.07, "sensitive"] = 1
    #drop other variables
    data = data.drop(columns=['racepctblack','racePctAsian','racePctAsian','racePctHisp'], axis=1)
    
    #extract the target and the features
    y=data['ViolentCrimesPerPop'].values.reshape(-1,1)
    data_wo_target=data.drop(columns=['ViolentCrimesPerPop'])
    X_col_names=list(data_wo_target.columns)
    X=data_wo_target.values
    S=X[:,X_col_names.index(SensitiveVarName)].ravel()
    return [X, y, S, X_col_names]


def read_and_treat_compas_data(DirectoryName,SensitiveVarName='race',display=False):
    """
    -> DirectoryName is the directory in which the file "compas-scores-two-years.csv" is located
    ->  SensitiveVarName is the string representing the sensitive variable
    Return [X, y, S, feature_names].
    """

    #read and merge original data
    data = pd.read_csv(os.path.join(DirectoryName, "compas-scores-two-years.csv"))
    data = data.loc[:, ['race','age','juv_misd_count','juv_other_count','priors_count', 'event','decile_score','two_year_recid']]

    #keep only black and white people
    data = data.loc[data['race'].isin(['African-American','Caucasian'])]
    data.loc[data['race'] == "African-American", "race"] = 0
    data.loc[data['race'] == "Caucasian", "race"] = 1
    
    #extract the target and the features
    y=data['two_year_recid'].values.reshape(-1,1)
    data_wo_target=data.drop(columns=['two_year_recid'])
    X_col_names=list(data_wo_target.columns)
    X=data_wo_target.values
    S=X[:,X_col_names.index(SensitiveVarName)].ravel()
    return [X, y, S, X_col_names]

def read_and_treat_student_data(DirectoryName,SensitiveVarName='sex',display=False):
    """
    -> DirectoryName is the directory in which the file "student.csv" is located
    ->  SensitiveVarName is the string representing the sensitive variable
    Return [X, y, S, feature_names].
    """

    #read and merge original data
    #data = pd.read_csv(os.path.join(DirectoryName, "student.csv"))
    data = pd.read_csv(os.path.join(DirectoryName, "student.csv"),sep=';',quotechar='"')
    
    # data = data.loc[:, ['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason',
    #                     'guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities',
    #                     'nursery','higher', 'internet','romantic','famrel','freetime','goout', 'Dalc', 'Walc', 
    #                     'health', 'absences', 'G1', 'G2', 'G3']]

    # 1 # create the sensitive variable
    data.loc[data['sex'] == "F", "sex"] = 0
    data.loc[data['sex'] == "M", "sex"] = 1
    
    #2 # Change  the categorical variables into numerical ones
    data.loc[data['address'] == "U", "address"] = 0
    data.loc[data['address'] == "R", "address"] = 1
    
    data.loc[data['famsize'] == "LE3", "famsize"] = 0
    data.loc[data['famsize'] == "GT3", "famsize"] = 1
    
    data.loc[data['Pstatus'] == "T", "Pstatus"] = 1
    data.loc[data['Pstatus'] == "A", "Pstatus"] = 0
    
    data.loc[data['school'] == "GP", "school"] = 0
    data.loc[data['school'] == "MS", "school"] = 1
    
    #Guardian
    Guardians = ["mother","father","other"]
    for guardian in Guardians:
        data.loc[data['guardian'] == guardian, 'guardian'] = Guardians.index(guardian)
    
    #Jobs of mother and father
    Jobs = ["teacher","services","health","at_home","other"]
    vars = ['Mjob','Fjob']
    for var in vars:
        for job in Jobs:
            data.loc[data[var] == job, var] = Jobs.index(job)
    
    #Reason to choose this school
    Reasons = ["course","home","reputation","other"]
    for reason in Reasons:
        data.loc[data['reason'] == reason, 'reason'] = Reasons.index(reason)
     
    #Change attribute with yes/no values into binary ones
    binary_vars = ['schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']
    for var in binary_vars:
        data.loc[data[var] == "yes", var] = 1
        data.loc[data[var] == "no", var] = 0
    
    #Extract the target and the features
    y=data['G3'].values.reshape(-1,1)
    data_wo_target=data.drop(columns=['G3'])
    X_col_names=list(data_wo_target.columns)
    X=data_wo_target.values
    S=X[:,X_col_names.index(SensitiveVarName)].ravel()
    return [X, y, S, X_col_names]


def load_dataset(dataset, data_dir):
    """Return the preprocessed source rows. Treat the cached arrays as read-only."""
    directory = Path(data_dir).resolve()
    if dataset not in DATA_FILES:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose from {list(DATA_FILES)}.")
    missing = [name for name in DATA_FILES[dataset] if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {dataset} data files in {directory}: {', '.join(missing)}. "
            "Run python Datas_Scripts/download_datasets.py "
            f"--datasets {dataset} --data-dir \"{directory}\". "
            "See Datas_Scripts/README.md for source and formatting instructions."
        )
    return _load_dataset(dataset, str(directory))

@lru_cache(maxsize=1)
def _load_dataset(dataset, data_dir):
    reader_name, sensitive_name = DATASET_READERS[dataset]
    reader = globals()[reader_name]
    options = {"categorical_codes": True} if dataset == "adult" else {}
    x, y, sensitive, feature_names = reader(
        DirectoryName=data_dir, SensitiveVarName=sensitive_name, display=False, **options
    )
    x = np.asarray(x)
    y = np.asarray(y).reshape(-1)
    sensitive = np.asarray(sensitive).reshape(-1)
    row_ids = np.arange(len(y), dtype=np.int64)
    if dataset == "adult":
        # The reader concatenates the official test and training files, test first.
        # Some copies include a non-data header at the start of the test file.
        with (Path(data_dir) / "adult.test.csv").open() as test_file:
            if test_file.readline().lstrip().startswith("|"):
                x, y, sensitive, row_ids = x[1:], y[1:], sensitive[1:], row_ids[1:]
    y = np.asarray(y, dtype=np.float64)
    sensitive = np.asarray(sensitive, dtype=np.float64)
    feature_names = list(feature_names)
    sensitive_column = feature_names.index(sensitive_name)
    if not np.array_equal(np.asarray(x[:, sensitive_column], dtype=float), sensitive):
        raise ValueError(f"{dataset}: the sensitive feature does not match the group labels.")
    kept = [index for index in range(len(feature_names)) if index != sensitive_column]
    features = [feature_names[index] for index in kept]
    categorical = tuple(index for index, name in enumerate(features)
                        if name in CATEGORICAL_NAMES.get(dataset, set()))
    xs = np.column_stack((np.asarray(x[:, kept], dtype=np.float64), sensitive))
    if not np.isfinite(xs).all() or not np.isfinite(y).all() or set(sensitive) != {0.0, 1.0}:
        raise ValueError(f"{dataset}: finite rows and both binary sensitive groups are required.")
    for array in (xs, y, row_ids):
        array.setflags(write=False)
    return {"XS": xs, "Y": y, "row_ids": row_ids, "feature_names": features + ["Sensitive"],
            "categorical_dims": categorical}
