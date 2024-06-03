import streamlit as st
import pandas as pd
import os


from typing import Optional, List, Tuple, Dict, Union



def validate_dataframe(df: pd.DataFrame,
                       n_cols: Optional[int] = None,
                       n_rows: Optional[Tuple[int, int]] = None,
                       columns: Optional[List[str]] = None,
                       column_types: Optional[Dict[str, type]] = None,
                       check_duplicates: bool = False,
                       check_null_values: bool = False,
                       unique_columns: Optional[List[str]] = None,
                       column_ranges: Optional[Dict[str, Tuple[Union[int, float], Union[int, float]]]] = None,
                       date_columns: Optional[List[str]] = None,
                       categorical_columns: Optional[Dict[str, List[Union[str, int, float]]]] = None) -> Tuple[
    bool, str]:
    if n_cols is not None and len(df.columns) != n_cols:
        return False, f"Error: Expected {n_cols} columns but found {len(df.columns)} columns."

    if n_rows is not None:
        min_rows, max_rows = n_rows
        if not (min_rows <= len(df) <= max_rows):
            return False, f"Error: Number of rows should be between {min_rows} and {max_rows}."

    if columns is not None and not set(columns).issubset(df.columns):
        missing_columns = set(columns) - set(df.columns)
        return False, f"Error: Missing columns: {missing_columns}."

    if column_types is not None:
        for col, expected_type in column_types.items():
            if col not in df.columns:
                return False, f"Error: Column '{col}' not found."
            if not df[col].dtype == expected_type:
                return False, f"Error: Column '{col}' should have type {expected_type}."

    if check_duplicates and df.duplicated().any():
        return False, "Duplicates found in the DataFrame."

    if check_null_values and df.isnull().any().any():
        return False, "DataFrame contains null values."

    if unique_columns is not None:
        for col in unique_columns:
            if col in df.columns and df[col].duplicated().any():
                return False, f"Column '{col}' should have only unique values."

    if column_ranges is not None:
        for col, value_range in column_ranges.items():
            if col in df.columns and not df[col].between(*value_range).all():
                return False, f"Values in '{col}' should be between {value_range[0]} and {value_range[1]}."

    if date_columns is not None:
        for col in date_columns:
            if col in df.columns:
                try:
                    pd.to_datetime(df[col], errors='raise')
                except ValueError:
                    return False, f"'{col}' should be in a valid date format."

    if categorical_columns is not None:
        for col, allowed_values in categorical_columns.items():
            if col in df.columns and not df[col].isin(allowed_values).all():
                return False, f"Values in '{col}' should be {allowed_values}."

    return True, "DataFrame has passed all validations."


# Streamlit app
st.title('Data Validation and Processing')

uploaded_file_config = st.file_uploader("Choose a model_config CSV file", type="csv")
uploaded_file_collateral = st.file_uploader("Choose a model_collateral CSV file", type="csv")
uploaded_file_authorrep = st.file_uploader("Choose a model_authorrep CSV file", type="csv")

#uploaded_files_authorrep = st.file_uploader("Choose authorrep CSV files", type="csv", accept_multiple_files=True)

# Process multiple author files
'''if uploaded_files_authorrep:
    dfs = [pd.read_csv(file) for file in uploaded_files_authorrep]

    # Concatenate all dataframes into a single dataframe
    authorrep_data = pd.concat(dfs, ignore_index=True)

    # Display the concatenated dataframe
    st.write("Concatenated Authorrep Data:", authorrep_data)'''



if uploaded_file_config and uploaded_file_collateral and uploaded_file_authorrep:
    model_config = pd.read_csv(uploaded_file_config)
    model_collateral = pd.read_csv(uploaded_file_collateral)
    model_authorrep = pd.read_csv(uploaded_file_authorrep)


    st.write("Model Config:")
    st.write(model_config)

    st.write("Model Collateral:")
    st.write(model_collateral)

    st.write("Model Authorrep:")
    st.write(model_authorrep)

    if st.button('Validate DataFrames'):
        valid_config, msg_config = validate_dataframe(model_config, n_cols=4, check_duplicates=True)
        valid_collateral, msg_collateral = validate_dataframe(model_collateral, n_cols=78, check_duplicates=True)
        valid_authorrep, msg_authorrep = validate_dataframe(model_authorrep, n_cols=14, check_duplicates=True)

        st.write(f"Model Config Validation: {valid_config}, Message: {msg_config}")
        st.write(f"Model Collateral Validation: {valid_collateral}, Message: {msg_collateral}")
        st.write(f"Model Authorrep Validation: {valid_authorrep}, Message: {msg_authorrep}")
    if st.button("ecl Reports"):
        import pandas as pd
        import streamlit as st
        join_coll_config = pd.merge(model_config, model_collateral, on='id')




        join_coll_auth = pd.merge(model_collateral, model_authorrep, on='id')
        st.write(join_coll_auth);
        stage1=join_coll_auth['EAD']* join_coll_auth['PD12']*join_coll_auth['LGD']
        stage2=join_coll_auth['EAD']*join_coll_auth['PDLT']*join_coll_auth['LGD']
        stage3=join_coll_auth['EAD'] * join_coll_auth['LGD']

        ECl_report = pd.concat([stage1, stage2, stage3, join_coll_auth[['EAD', 'PD12', 'LGD', 'PDLT']]], axis=1)

        ECl_report = ECl_report.reset_index(drop=True)

        st.write("Ecl Reports sucessfully",ECl_report)
    if st.button("ead varition"):
        # Calculate change in EAD
        join_coll_config = pd.merge(model_config, model_collateral, on='id')
        #st.write(join_coll_config);

        join_coll_auth = pd.merge(model_collateral, model_authorrep, on='id')
        change_ead = join_coll_auth['EAD'] - join_coll_auth['Previous EAD']

        # Calculate percentage change in EAD
        ead_percentage = (join_coll_auth['EAD'] - join_coll_auth['Previous EAD']) / join_coll_auth['Previous EAD'] * 100

        # Concatenate the calculated columns with the original columns
        ead_variation = pd.concat([change_ead, ead_percentage, join_coll_auth[['EAD', 'Previous EAD','Reporting Date']]], axis=1)

        # Reset index if needed
        ead_variation = ead_variation.reset_index(drop=True)
        print(join_coll_auth.columns)

        # Display the DataFrame

        st.write('EAD variation',ead_variation)


if st.button("ml"):
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix

    Ecl_computation = pd.read_csv(
        r"C:\Users\Star\Desktop\Stuffs(IMP)\dataScience expotent files\Project (datascience)\Output_data.csv")
    Ead_report = pd.read_csv(
        r"C:\Users\Star\Desktop\Stuffs(IMP)\dataScience expotent files\Project (datascience)\A.csv")

    merged_data = pd.merge(Ecl_computation, Ead_report, on='EAD')
    print(merged_data)
    print(merged_data['Reporting Date'])
    merged_data.dropna()
    # Assuming 'Target' is binary (0 or 1), you may need to adjust this based on your data
    # Convert the target variable to binary if necessary
    # merged_data['Reporting Date'] = merged_data['Reporting Date'].astype(int)
    merged_data['Reporting Date'] = pd.to_datetime(merged_data['Reporting Date'])
    # Split the data into features (X) and target variable (y)
    X = merged_data.drop(columns=['EAD', 'Reporting Date', 'Previous EAD', 'Reporting Date'])
    y = merged_data['Reporting Date']
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply logistic regression
    logistic_model = LogisticRegression()
    logistic_model.fit(X_test, y_test)

    # Predictions
    y_pred = logistic_model.predict(X_train)

    # Model evaluation
    accuracy = accuracy_score(y_train, y_pred)
    conf_matrix = confusion_matrix(y_train, y_pred)

    #print("Accuracy:", accuracy)
    #print("Confusion Matrix:\n", conf_matrix)

    # Plot scatter plot of actual vs. predicted values
    plt.scatter(y_train, y_pred, alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs. Predicted")
    plt.show()
    st.write("accuracy is",accuracy)

    st.write("congif is",conf_matrix)

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Assuming you have loaded your data into a DataFrame named merged_data
# Define features (X) and target variable (y)
Ecl_computation = pd.read_csv(
    r"C:\Users\Star\Desktop\Stuffs(IMP)\dataScience expotent files\Project (datascience)\Output_data.csv")
Ead_report = pd.read_csv(
    r"C:\Users\Star\Desktop\Stuffs(IMP)\dataScience expotent files\Project (datascience)\A.csv")

merged_data = pd.merge(Ecl_computation, Ead_report, on='EAD')
print(merged_data)
print(merged_data['Reporting Date'])
merged_data.dropna()
merged_data['Reporting Date'] = pd.to_datetime(merged_data['Reporting Date'])
    # Split the data into features (X) and
y = merged_data['Reporting Date']
X = merged_data.drop(columns=['EAD', 'Reporting Date', 'Previous EAD'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Streamlit app
st.title('Random Forest Classifier')


# Button to trigger Random Forest classification
if st.button("Random Forest"):
    # Apply Random Forest classifier
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    # Predictions
    y_pred_rf = rf_model.predict(X_test)

    # Model evaluation
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
    result_rf = classification_report(y_test, y_pred_rf)

    # Display results
    st.write("Random Forest Model Accuracy:", accuracy_rf)
    st.write("Random Forest Confusion Matrix:\n", conf_matrix_rf)
    st.write("Classification Report:", result_rf)








