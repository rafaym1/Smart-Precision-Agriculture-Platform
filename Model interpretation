# Define a custom scorer for multiclass precision that handles string labels correctly
from sklearn.metrics import precision_score

def multiclass_precision(y_true, y_pred):
    """Custom function to calculate precision for multiclass string labels"""
    return precision_score(y_true, y_pred, average='macro')

# Create a scorer that works with string labels
scorer_dict_tmp = {'ave_precision': make_scorer(multiclass_precision)}

# Set up dataframe to store results
df_tmp = df_dummies.iloc[:, 0:7]
score_diffs = pd.DataFrame(columns=['Feature', 'Drop Impact'])

# Baseline score (from previous calculation)
# If main_test_ave_precision is not calculated yet, uncomment below
# log_model_baseline = LogisticRegression(max_iter=200)
# X_baseline = scaler.fit_transform(df_tmp)
# log_CV_baseline = cross_validate(log_model_baseline, X_baseline, y, cv=10, scoring=scorer_dict_tmp)
# main_test_ave_precision = np.mean(log_CV_baseline['test_ave_precision'])

# Evaluate impact of dropping each feature
for i in range(len(df_tmp.columns)):
    # Drop one feature at a time
    X_tmp = scaler.fit_transform(df_tmp.drop(df_tmp.columns[i], axis=1))
    
    # Train and evaluate model
    log_model_tmp = LogisticRegression(max_iter=200)
    log_CV_tmp = cross_validate(log_model_tmp, X_tmp, y, cv=10, scoring=scorer_dict_tmp)
    
    # Calculate performance drop
    score_tmp = np.mean(log_CV_tmp['test_ave_precision'])
    score_diff = main_test_ave_precision - score_tmp
    
    # Store results
    score_diffs.loc[len(score_diffs)] = [df_tmp.columns[i], score_diff]

# Sort features by impact
score_diffs.sort_values(by='Drop Impact', inplace=True, ascending=True)

# Plot results
plt.figure(figsize=(10, 6))
plt.barh(y=score_diffs['Feature'], width=score_diffs['Drop Impact'], color='green')
plt.title('Feature Importance (Impact of Dropping Each Feature)')
plt.xlabel('Decrease in Precision Score')
plt.tight_layout()
plt.show()

#training the model
log_model_final = LogisticRegression(max_iter = 200)
log_model_final.fit(X_main, y)

# re-fit the scaler to the traning data
scaler.fit(df_dummies.iloc[:,0:7])

# scale demo input with the same scaler used to transform the training data
X_test = scaler.transform(df_test.iloc[:,0:7])
y_test = df_test['label']

# score the model's performance on the test set
preds_test = log_model_final.predict(X_test)
precision_score(y_test, preds_test, average='macro')

fig, ax = plt.subplots(figsize=(18, 3))
fig.suptitle('Model Coefficients Heatmap', y=1, fontsize=16)

sns.heatmap(log_model_final.coef_.transpose(), annot=True, ax=ax, vmin=-6, vmax=6,
            xticklabels=log_model_final.classes_, yticklabels=df_dummies.iloc[:,0:7].columns,
            cmap=sns.diverging_palette(300, 145, s=60, as_cmap=True)) 
plt.show()

