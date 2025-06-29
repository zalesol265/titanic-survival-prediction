import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier


def plot_categorical(df, col, target):
    sns.countplot(data=df, x=col, hue=target)
    plt.title(f"{col} vs {target}")
    plt.show()


def plot_numerical(df, col, target):
    sns.histplot(data=df, x=col, hue=target, kde=True)
    plt.title(f"{col} distribution by {target}")
    plt.show()


def plot_features(df, cols, target):
    for col in cols:
        if col not in df.columns:
            print(f"⚠️ Skipping {col}: not found in DataFrame.")
            continue
        
        if pd.api.types.is_numeric_dtype(df[col]):
            print(f"📈 Plotting numerical feature: {col}")
            plot_numerical(df, col, target)
            plot_numerical_percentage(df, col, target)
        else:
            print(f"📊 Plotting categorical feature: {col}")
            plot_categorical(df, col, target)
            plot_categorical_percentage(df, col, target)


def plot_categorical_percentage(df, feature, target):
    # Create a cross-tab of counts
    ct = pd.crosstab(df[feature], df[target])
    # Normalize to get percentages across rows (or columns if you want)
    ct_perc = ct.div(ct.sum(axis=1), axis=0) * 100

    # Plot as stacked barplot by percentage
    ct_perc.plot(kind='bar', stacked=True, figsize=(8,6), colormap='viridis')
    plt.ylabel('Percentage (%)')
    plt.title(f'Percentage distribution of {target} by {feature}')
    plt.legend(title=target, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_numerical_percentage(df, feature, target, bins=10):
    # Bin numerical feature
    df[f'{feature}_binned'] = pd.cut(df[feature], bins=bins)

    # Crosstab counts of binned feature by target
    ct = pd.crosstab(df[f'{feature}_binned'], df[target])

    # Normalize by row to get percentages
    ct_perc = ct.div(ct.sum(axis=1), axis=0) * 100

    # Plot stacked bar chart with percentages
    ct_perc.plot(kind='bar', stacked=True, figsize=(8,6), colormap='viridis')
    plt.ylabel('Percentage (%)')
    plt.title(f'Percentage distribution of {target} by binned {feature}')
    plt.legend(title=target, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Optionally drop the temporary binned column
    df.drop(columns=[f'{feature}_binned'], inplace=True)


def show_missing(df):
    msno.matrix(df)

def plot_decision_tree(df, target_col, feature_col=None, max_depth=3, figsize=(10,8)):
    """
    Train and plot a decision tree on specified feature(s) to predict the target.

    Categorical columns are automatically one-hot encoded.

    Args:
        df (pd.DataFrame): The dataset.
        target_col (str): The target column name.
        feature_col (str or list of str or None): Feature column(s) to use. 
            If None, uses all columns except target.
        max_depth (int): Maximum depth of the tree.
        figsize (tuple): Figure size for the plot.
    """
    if feature_col is None:
        X = df.drop(columns=[target_col])
    elif isinstance(feature_col, str):
        X = df[[feature_col]]
    else:
        X = df[list(feature_col)]

    y = df[target_col]

    # One-hot encode categorical columns automatically
    X = pd.get_dummies(X, drop_first=True)

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X, y)

    tree_rules = export_text(clf, feature_names=list(X.columns), show_weights=True)
    print(tree_rules)

    plt.figure(figsize=figsize)
    plot_tree(
        clf, 
        feature_names=X.columns, 
        class_names=[str(c) for c in clf.classes_], 
        filled=True, 
        rounded=True, 
        fontsize=10
    )
    plt.title(f"Decision Tree to predict '{target_col}'")
    plt.show()


def survival_rate_by_feature(df, feature, target='Survived', bins=None, quantiles=False, plot=True):
    data = df.copy()
    
    # Drop rows where the feature is null, so binning and grouping work without errors
    data = data[data[feature].notnull()]

    # Bin numerical features if requested
    if bins:
        if quantiles:
            data[feature + '_binned'] = pd.qcut(data[feature], q=bins)
        else:
            data[feature + '_binned'] = pd.cut(data[feature], bins=bins)
        group_col = feature + '_binned'
    else:
        group_col = feature

    grouped = data.groupby(group_col)[target].agg(['count', 'sum'])
    print(grouped)
    grouped['survival_rate'] = grouped['sum'] / grouped['count']

    if plot:
        ax = grouped['survival_rate'].plot(kind='bar', figsize=(8,5), title=f'Survival Rate by {group_col} (%)')
        plt.ylabel('Survival Rate')
        plt.xlabel(group_col)
        plt.xticks(rotation=45)

        # Add percentage labels on top of bars
        for p in ax.patches:
            height = p.get_height()
            ax.text(
                p.get_x() + p.get_width() / 2,  # x position: center of the bar
                height + 0.01,                  # y position: just above the bar
                f'{height*100:.1f}%',          # label text as percentage with 1 decimal place
                ha='center'                    # horizontal alignment
            )

        plt.tight_layout()
        plt.show()

    return grouped



def encode_categoricals_for_corr(df):
    df_encoded = df.copy()
    label_enc = LabelEncoder()
    
    for col in df_encoded.select_dtypes(include='object').columns:
        try:
            df_encoded[col] = label_enc.fit_transform(df_encoded[col].astype(str))
        except:
            pass  # skip if it errors (optional handling)

    return df_encoded


def plot_correlation_matrix(df, figsize=(10, 8), annot=True, cmap='coolwarm', title='Correlation Matrix'):
  
    encoded_df = encode_categoricals_for_corr(df)
  
    # Select only numeric columns
    numeric_df = encoded_df.select_dtypes(include=['number'])

    # Compute correlation matrix
    corr = numeric_df.corr()

    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=annot, fmt=".2f", cmap=cmap, square=True, linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return corr



# def survival_rate_by_groups(df, group_cols, target='Survived', bin_col=None, bins=None, bin_labels=None, plot=True):
#     data = df.copy()

#     # Drop rows with missing group or target values
#     cols_to_check = group_cols + [target]
#     if bin_col:
#         cols_to_check.append(bin_col)
#     data = data.dropna(subset=cols_to_check)

#     # Handle optional binning
#     if bin_col:
#         if bins is None:
#             # Default binning for age
#             bins = [0, 12, 18, 30, 45, 60, 100]
#             bin_labels = ['Child', 'Teen', '20-29', '30-44', '45-59', '60+']
#         data[bin_col + '_group'] = pd.cut(data[bin_col], bins=bins, labels=bin_labels, right=False)
#         group_cols = [bin_col + '_group' if col == bin_col else col for col in group_cols]

#     # Group and calculate survival stats
#     grouped = data.groupby(group_cols)[target].agg(['count', 'sum']).reset_index()
#     grouped['survival_rate'] = grouped['sum'] / grouped['count']

#     # Show table
#     print(grouped)

#     # Optional plot: only if one or two group columns
#     if plot:
#         if len(group_cols) == 1:
#             x, hue = group_cols[0], None
#         elif len(group_cols) >= 2:
#             x, hue = group_cols[0], group_cols[1]
#         else:
#             x, hue = None, None

#         if x:
#             plt.figure(figsize=(10,6))
#             ax = sns.barplot(data=grouped, x=x, y='survival_rate', hue=hue)

#             plt.title(f"Survival Rate by {' and '.join(group_cols)}")
#             plt.ylabel('Survival Rate')
#             plt.xlabel(x)
#             plt.ylim(0, 1)

#             if hue:
#                 plt.legend(title=hue)

#             # Add percentage labels
#             for p in ax.patches:
#                 height = p.get_height()
#                 if not pd.isna(height):
#                     ax.text(
#                         x=p.get_x() + p.get_width() / 2,
#                         y=height + 0.02,  # adjust offset as needed
#                         s=f'{height:.0%}',  # convert to percentage
#                         ha='center',
#                         va='bottom',
#                         fontsize=9
#                     )

#             plt.tight_layout()
#             plt.show()

#     return grouped


def survival_rate_by_groups(
    df,
    group_cols,
    target='Survived',
    bin_col=None,
    bins=None,
    bin_labels=None,
    bin_method='cut',  # 'cut' or 'qcut'
    plot=True
):
    data = df.copy()

    # Drop rows with missing group or target values
    cols_to_check = group_cols + [target]
    if bin_col:
        cols_to_check.append(bin_col)
    data = data.dropna(subset=cols_to_check)

    if bin_col:
        if bin_method == 'qcut':
            # qcut auto labels intervals if no labels provided
            data[bin_col + '_group'] = pd.qcut(data[bin_col], q=bins, labels=bin_labels)
        else:  # cut with range labels
            if bin_labels is None:
                # Get bin edges for cut
                bin_edges = pd.cut(data[bin_col], bins=bins, retbins=True)[1]
                # Create range labels
                bin_labels = [
                    f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}"
                    for i in range(len(bin_edges)-1)
                ]
            data[bin_col + '_group'] = pd.cut(data[bin_col], bins=bins, labels=bin_labels, include_lowest=True)

        group_cols = [bin_col + '_group' if col == bin_col else col for col in group_cols]

    # Group and calculate survival stats
    grouped = data.groupby(group_cols)[target].agg(['count', 'sum']).reset_index()
    grouped['survival_rate'] = grouped['sum'] / grouped['count']

    # Show table
    print(grouped)

    # Plot
    if plot:
        if len(group_cols) == 1:
            x, hue = group_cols[0], None
        elif len(group_cols) >= 2:
            x, hue = group_cols[0], group_cols[1]
        else:
            x, hue = None, None

        if x:
            plt.figure(figsize=(10,6))
            ax = sns.barplot(data=grouped, x=x, y='survival_rate', hue=hue)
            plt.title(f"Survival Rate by {' and '.join(group_cols)}")
            plt.ylabel('Survival Rate')
            plt.xlabel(x)
            plt.ylim(0, 1)

            if hue:
                plt.legend(title=hue)

            for p in ax.patches:
                height = p.get_height()
                if not pd.isna(height):
                    ax.text(
                        x=p.get_x() + p.get_width() / 2,
                        y=height + 0.02,
                        s=f'{height:.0%}',
                        ha='center',
                        va='bottom',
                        fontsize=9
                    )

            plt.tight_layout()
            plt.show()

    return grouped



def train_catboost_and_plot_importance(df, target_col, categorical_features, numerical_features,
                                       iterations=100, learning_rate=0.1, depth=3, random_seed=42):
    """
    Trains a CatBoost model and plots feature importances.

    Parameters:
    - df (pd.DataFrame): The full dataset
    - target_col (str): The name of the target column
    - categorical_features (list): List of categorical column names
    - numerical_features (list): List of numerical column names
    - iterations (int): Number of training iterations
    - learning_rate (float): Learning rate
    - depth (int): Tree depth
    - random_seed (int): Random seed for reproducibility

    Returns:
    - model (CatBoostClassifier): Trained CatBoost model
    - importance_df (pd.DataFrame): DataFrame of feature importances
    """

    # Prepare features and target
    X = df[categorical_features + numerical_features].copy()
    y = df[target_col]

    # Ensure categorical features are string type
    X[categorical_features] = X[categorical_features].astype(str)

    # Initialize and train model
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        cat_features=categorical_features,
        verbose=0,
        random_seed=random_seed
    )
    model.fit(X, y)

    # Get importances
    importances = model.get_feature_importance()
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Print and plot
    print(importance_df)
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title("CatBoost Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    return model, importance_df

