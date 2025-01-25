import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from feature_engine.outliers import OutlierTrimmer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
df = pd.read_csv('loan_data.csv')
# print(df.info())

df["person_age"] = df["person_age"].astype(int)

cat_cols = [var for var in df.columns if df[var].dtypes == 'object']
num_cols = [var for var in df.columns if df[var].dtypes != 'object']

# for i in cat_cols:
#     print(df[i].value_counts())
#     print("\n")


def plot_categorical_column(dataframe, column):
    plt.figure(figsize=(7, 7))
    ax = sns.countplot(x=dataframe[column])
    total_count = len(dataframe[column])
    threshold = 0.05 * total_count
    category_counts = dataframe[column].value_counts(normalize=True) * 100
    ax.axhline(threshold, color='red', linestyle='--', label=f'0.05% of total count ({threshold:.0f})')

    for p in ax.patches:
        height = p.get_height()
        percentage = (height / total_count) * 100
        ax.text(p.get_x() + p.get_width() / 2., height + 0.02 * total_count, f'{percentage:.2f}%', ha="center")

    plt.title(f'Label Cardinality for "{column}" Column')
    plt.ylabel('Count')
    plt.xlabel(column)
    plt.tight_layout()

    plt.legend()
    plt.show()


# for col in cat_cols:
#     plot_categorical_column(df, col)

# df[num_cols].hist(bins=30, figsize=(12,10))
# plt.show()

# label_prop = df['loan_status'].value_counts()
#
# plt.pie(label_prop.values, labels=['Rejected (0)', 'Approved (1)'], autopct='%.2f')
# plt.title('Target label proportions')
# plt.show()

# for col in num_cols:
#     sns.boxplot(df[col])
#     plt.title(f'Boxplot of {col}')
#     plt.show()
skewed_cols = ['person_age', 'person_income', 'person_emp_exp',
               'loan_amnt', 'loan_percent_income',
               'cb_person_cred_hist_length', 'credit_score']

norm_cols= ['loan_int_rate']


mms = MinMaxScaler()
ss = StandardScaler()

df[skewed_cols] = ss.fit_transform(df[skewed_cols])
df[skewed_cols] = ss.transform(df[skewed_cols])

df[norm_cols] = mms.fit_transform(df[norm_cols])
df[norm_cols] = mms.transform(df[norm_cols])

person_education = {'High School': 0,'Associate': 1,'Bachelor': 2,'Master': 3,'Doctorate':4}
gender_mapping = {'male': 0, 'female': 1}
home_ownership_mapping = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
loan_intent_mapping = {'PERSONAL': 0, 'EDUCATION': 1, 'MEDICAL': 2, 'VENTURE': 3, 'HOMEIMPROVEMENT': 4, 'DEBTCONSOLIDATION': 5}
previous_loan_defaults_mapping = {'No': 0, 'Yes': 1}

df['person_education'] = df['person_education'].map(person_education)
df['person_gender'] = df['person_gender'].map(gender_mapping)
df['person_home_ownership'] = df['person_home_ownership'].map(home_ownership_mapping)
df['loan_intent'] = df['loan_intent'].map(loan_intent_mapping)
df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map(previous_loan_defaults_mapping)
df.isnull().sum()

trimmer = OutlierTrimmer(capping_method='iqr', tail='right',
                        variables= ['person_age', 'person_gender', 'person_education', 'person_income',
       'person_emp_exp', 'person_home_ownership', 'loan_amnt',
       'loan_intent', 'loan_int_rate', 'loan_percent_income',
       'cb_person_cred_hist_length', 'credit_score',
       'previous_loan_defaults_on_file'])

df2 = trimmer.fit_transform(df)
print(df.shape)
print(df2.shape)

# plt.figure(figsize=(15, 8))
# sns.heatmap(df2.corr(), annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix')
# plt.show()

threshold = 0.1

correlation_matrix = df2.corr()
high_corr_features = correlation_matrix.index[abs(correlation_matrix["loan_status"]) > threshold].tolist()
high_corr_features.remove("loan_status")
print(high_corr_features)

X_selected = df[high_corr_features]
Y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(X_selected, Y, test_size=0.2, random_state=42)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)