# Decision Tree Project

This project focuses on building and pruning decision trees for classification tasks using machine learning techniques. Decision trees are powerful algorithms that recursively split data into subsets based on input features, ultimately providing predictions for new data points.

## Methodology

### Data Processing

1. The project starts by reading the dataset from the file 'data/csv/census_income.csv' and assigning it to the variable `df`.
2. The modified CSV file is then read back into `df`.
3. Missing values in the DataFrame are replaced with the mean value (most frequent value and average value) for each column, effectively handling missing data.
4. The DataFrame with filled missing values is saved to 'discretized_data.csv'.

### Building Decision Trees

1. The decision tree classifier is trained using the function `grow_tree(df_input, X_train, y_train, x_test, y_test)`.
2. The `plot_tree` function is used to visualize the trained decision tree. It takes the trained classifier (DecisionTree), feature names (`feature_names`), and class names (`class_names`) as parameters.
3. The `feature_names` parameter is set to `cat_cols` and `num_cols`, which are lists of column names of the feature matrix. This is necessary to label the nodes in the decision tree with their corresponding feature names.
4. The `class_names` parameter is set to `target`, representing the unique class labels of the target variable (`y`). This is used to label the leaf nodes of the decision tree with the class names.
5. The decision tree is plotted using the matplotlib library.
6. Finally, the decision tree plot is displayed using `plt.show()`, rendering the plot on the screen.

### Pruning Algorithm

The pruning algorithm used is Reduced Error Pruning, which involves training decision trees with different depths and evaluating their performance on separate validation sets to find the optimal depth that balances model complexity and generalization, preventing overfitting. The resulting plot provides insights into the ideal depth.

1. The lists `x_test`, `y_ax_test`, `y_ax_train`, and `y_ax_valid` are initialized to store the values of the maximum depth (x-axis) and the corresponding accuracy scores (y-axis) for the test, train, and validation sets.
2. The for loop iterates from 1 to 20 to evaluate decision trees with different maximum depths.
3. The `delete_nodes` function identifies the node to be deleted and prints the deleted node to the console.
4. Inside the loop, a decision tree classifier is created with the current maximum depth.
5. The decision tree classifier is fitted on the training data (`X_train` and `y_train`).
6. The current maximum depth (`n`) is printed on the console. The number of nodes is appended to `num_nodes`, accuracy of train is appended to `accuracy_train`, and accuracy of test is appended to `accuracy_test`.
7. The accuracy scores for the test, train, and validation sets are computed using the `accuracy` function defined in the code.
8. The accuracy scores for the test, train, and validation sets are appended to the respective lists (`y_ax_test`, `y_ax_train`, and `y_ax_valid`).
9. Finally, a plot is generated using `plt.plot` to visualize the change in accuracy with varying tree size.

The pruning methodology (Reduced Error Pruning) allows for pruning the decision tree by evaluating its performance at different sizes. This is achieved by finding the deepest node and then recursively deleting the nodes. By comparing the accuracy with the size of the tree, the trade-off between the number of nodes and the accuracy of the tree can be analyzed. The plot displays the accuracy vs size of the tree, providing insights into the ideal pruning length.
