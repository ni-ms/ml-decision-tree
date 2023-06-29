import os
from matplotlib import pyplot as plt
from progress.bar import Bar
import graphviz
import numpy as np
import pandas as pd
import pydotplus
from sklearn.metrics import accuracy_score, f1_score

from decisionTree import DecisionTree
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

# get the data from the csv file and replace the missing values with NaN
df = pd.read_csv('data/csv/census-income.csv', header=None)
df = df.replace(' ?', np.nan)
df = df.set_axis(
    ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
     'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class'], axis=1,
    copy=False)

df_test = pd.read_csv('data/csv/census-income.test.csv', header=None)

df_test = df_test.replace(' ?', np.nan)
df_test = df_test.set_axis(
    ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
     'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class'], axis=1,
    copy=False)

cat_cols = ["workclass", "occupation", "native-country"]
imp_cat = SimpleImputer(strategy="most_frequent")
df[cat_cols] = imp_cat.fit_transform(df[cat_cols])
df_test[cat_cols] = imp_cat.fit_transform(df_test[cat_cols])

num_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
imp_num = SimpleImputer(strategy="mean")
df[num_cols] = imp_num.fit_transform(df[num_cols])
df_test[num_cols] = imp_num.fit_transform(df_test[num_cols])

df.to_csv('data/csv/census-income-modified.csv', index=False)
df_test.to_csv('data/csv/census-income-test-modified.csv', index=False)
df2 = pd.read_csv('data/csv/census-income-modified.csv', header=0)
df_test2 = pd.read_csv('data/csv/census-income-test-modified.csv', header=0)

target = df2['class']
data = df2.drop(columns=['class'])

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.5, random_state=13)


def discretize_data(inp_df, continuous_features):
    discretized_df = inp_df.copy()
    # use kbnins discretizer to discretize the continuous features
    discritizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans')

    for feature in continuous_features:
        discretized_df[feature] = discritizer.fit_transform(inp_df[feature].values.reshape(-1, 1))

    # for each feature, replace 0, 1, 2, 3, or 4 with your own labels
    for feature in continuous_features:
        discretized_df[feature] = discretized_df[feature].replace([0, 1, 2, 3, 4],
                                                                  ['very-low', 'low', 'medium', 'high', 'very-high'])

    return discretized_df


def grow_tree(input_name, x_data_inp, y_data_inp, x_test_inp, y_test_inp):
    data = pd.read_csv(input_name)
    cols = data.columns
    desc_features = cols[:-1]
    target = cols[-1]

    for column in cols:
        data[column] = data[column].astype(str)

    # data_desc = data[desc_features].values
    # data_target = data[target].values
    data_desc = x_data_inp.values
    data_target = y_data_inp.values

    decisionTree = DecisionTree(data_desc.tolist(), desc_features, data_target.tolist(), 'entropy')
    decisionTree.id3(0, 0)

    """Old accuracy code"""
    #
    # def accuracy(decisionTree, x_data_inp, y_data_inp):
    #     correct_count = 0
    #
    #     bar = Bar('Processing', max=len(x_data_inp))
    #
    #     for i in range(len(x_data_inp)):
    #         x_data_v = x_data_inp.iloc[i]
    #         val = decisionTree.traverse_tree_help(x_data_v)
    #
    #         if y_data_inp.iloc[i] == val:
    #             correct_count += 1
    #         bar.next()
    #
    #     bar.finish()
    #
    #     print("Accuracy: ", correct_count / len(x_data_inp) * 100)
    #     return correct_count / len(x_data_inp) * 100
    #
    # num_nodes = []
    # accuracy_train = []
    # accuracy_test = []
    #
    # for i in range(20):
    #     threshold = 0.01 + i * 0.005
    #     decisionTree.id3(threshold, 0)
    #     decisionTree.delete_nodes()
    #     acc_train = accuracy(decisionTree, x_data_inp, y_data_inp)
    #     acc_test = accuracy(decisionTree, x_test_inp, y_test_inp)
    #     n_nodes = decisionTree.get_num_nodes()
    #
    #     num_nodes.append(n_nodes)
    #     accuracy_train.append(acc_train)
    #     accuracy_test.append(acc_test)
    #
    # # plot number of nodes vs accuracy
    # plt.plot(num_nodes, accuracy_train)
    # plt.xlabel('Number of Nodes')
    # plt.ylabel('Accuracy')
    # plt.title('Number of Nodes vs Accuracy (Training Data)')
    # plt.show()
    #
    # # plot number of nodes vs accuracy
    # plt.plot(num_nodes, accuracy_test)
    # plt.xlabel('Number of Nodes')
    # plt.ylabel('Accuracy')
    # plt.title('Number of Nodes vs Accuracy (Test Data)')
    # plt.show()

    # New function to do reduced error prunign
    decisionTree.reduced_error_pruning_helper(x_test_inp, y_test_inp)
    x_val = decisionTree.num_nodes
    x_val.sort()
    y_val = decisionTree.accuracy_train
    y_val.sort()

    decisionTree.reduced_error_pruning_helper(x_data_inp, y_data_inp)
    # plot number of nodes vs accuracy
    x_val_2 = decisionTree.num_nodes

    y_val_2 = decisionTree.accuracy_train

    plt.plot(x_val_2, y_val_2, label='Test Data')
    plt.plot(x_val, y_val, label='Training Data')

    plt.xlabel('Number of Nodes')
    plt.ylabel('Accuracy')
    plt.title('Number of Nodes vs Accuracy ')
    plt.show()

    dot = decisionTree.print_visualTree(render=True)
    # print(dot)
    print("System Entropy: ", format(decisionTree.entropy, '.5f'))

    # os.rename('output', folder_name)


def get_data_set(test_df, train_df):
    # combine test and train data
    combined_df = pd.concat([test_df, train_df])
    # randomly select 67% of the data for training and 33% for testing
    train_df = combined_df.sample(frac=0.67, random_state=13)
    test_df = combined_df.drop(train_df.index)
    return train_df, test_df


def validation_data(testing_input):
    # 50% of the dataframe is used for training and 50% for testing
    test_df = testing_input.sample(frac=0.5, random_state=13)
    train_df = testing_input.drop(test_df.index)
    return train_df, test_df


def random_forest(data):
    # Load the data from CSV

    # Prepare the data
    X = data.drop('class', axis=1)
    y = data['class']

    # Perform one-hot encoding on categorical columns
    categorical_columns = X.select_dtypes(include='object').columns
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_columns]))
    X_encoded.columns = encoder.get_feature_names_out(categorical_columns)
    X = pd.concat([X.drop(categorical_columns, axis=1), X_encoded], axis=1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Building the Classifier
    random_forest = RandomForestClassifier(random_state=50)

    # Training
    random_forest.fit(X_train, y_train)

    # Predictions
    y_randomforest_predictions = random_forest.predict(X_test)

    # Evaluation Metrics of the model Accuracy and F1 score
    accuracy = accuracy_score(y_test, y_randomforest_predictions)
    f1 = f1_score(y_test, y_randomforest_predictions, pos_label=' >50K.', average='micro')
    print('Accuracy score:', round(accuracy * 100, 2))
    print('F1 score:', round(f1 * 100, 2))


def run_Code():
    temp = discretize_data(df2, num_cols)
    temp_2 = discretize_data(df_test2, num_cols)

    # save the discretized data to a csv file
    temp.to_csv('discretized_data.csv', index=False)
    temp_2.to_csv('discretized_data_test.csv', index=False)

    temp_forcode = pd.read_csv('discretized_data.csv')

    # Validation data -> find accuracy
    train_df, test_df = validation_data(temp)
    x_train = train_df.drop(columns=['class'])
    y_train = train_df['class']
    x_test = test_df.drop(columns=['class'])
    y_test = test_df['class']
    grow_tree('discretized_data.csv', x_train, y_train, x_test, y_test)

    # now use get_data_set to get the train and test data
    train_df, test_df = get_data_set(temp, temp_2)
    x_train = train_df.drop(columns=['class'])
    y_train = train_df['class']
    x_test = test_df.drop(columns=['class'])
    y_test = test_df['class']
    grow_tree('discretized_data.csv', x_train, y_train, x_test, y_test)

    random_forest(temp)


run_Code()

# grow_tree('discretized_data_test.csv', 'output_discretized')
# df = pd.read_csv('discretized_data_test.csv')


# df3 = pd.read_csv('discretized_data_test.csv', header=0)

# only keep the class column
# df3_onlyclass = df3['class']

# traverse_tree('output/tempfile.gv', df3.iloc[0])

# grow_tree('temp_data.csv', df3, df3_onlyclass)

# print("Done")

# test_df = df3.sample(frac=0.5, random_state=13)
# test_df_x = test_df.drop(columns=['class'])
# test_df_y = test_df['class']
# train_df = df3.drop(test_df.index)
# train_df_x = train_df.drop(columns=['class'])
# train_df_y = train_df['class']
# random_forest(train_df_x, train_df_y, test_df_x, test_df_y)
