# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import gc
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import metrics

pal = sns.color_palette()
sns.set(rc={'figure.figsize': (6, 6)})
pd.set_option('display.width', 500, 'display.max_rows', 500)

model_sample = pd.read_csv('./input/model_sample.csv', index_col=0)  # , skiprows=range(1,  9000),  nrows=11081 - 9000
# print(model_sample.head())

# drop columns that only contains 1 unique values
orig_columns = model_sample.columns
drop_columns = []
for col in orig_columns:
    unique_values = model_sample[col].dropna().unique()
    if len(unique_values) == 1:
        drop_columns.append(col)
model_sample.drop(drop_columns,  axis=1,  inplace=True)
print('>>drop {} columns: only have 1 unique value'.format([col for col in drop_columns]))

# pre-process missing value

# drop cols
orig_cols_num = model_sample.shape[1]
half_count = len(model_sample) / 2
model_sample.dropna(thresh=half_count,  axis=1,  inplace=True)
print('>>delete {} cols containing more than half na'.format(orig_cols_num - model_sample.shape[1]))

# correlation matrix
# model_sample2 = model_sample.fillna(0)
# corrmat = model_sample2.corr()
# cols = corrmat.nlargest(15, 'y')['y'].index
# cm = np.corrcoef(model_sample2[cols].values.T)
# cm_df = pd.DataFrame(cm)
# cm_df.index = ['i'+col for col in cols]
# cm_df.columns = cols
# print(cm_df)
#
#
# def get_high_corr():
#     high_corr_dict = {}
#     for i in cm_df.columns:
#         for j in cm_df.index:
#             if cm_df[i].loc[j] > 0.8 and str(j)[1:] != str(i):
#                 if i not in high_corr_dict.keys():
#                     high_corr_dict[i] = []
#                     high_corr_dict[i].append(j)
#                 else:
#                     high_corr_dict[i].append(j)
#
#     print(high_corr_dict)


# get_high_corr()

# interpolate
null_total = model_sample.isnull().sum().sort_values(ascending=False)
null_pct = (model_sample.isnull().sum()/model_sample.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([null_total, null_pct], axis=1, keys=['Total', 'Percent'])
print(missing_data)


def drop_and_fill(fill_col, *drop_cols):
    """
    fill the first and drop others (which correlated with the first one)
    """
    dropped = []
    for arg in drop_cols:
        try:
            model_sample.drop(arg, axis=1, inplace=True)
            dropped.append(arg)
        except KeyError:
            print('>>Error: col % s already deleted' % arg)
    try:
        model_sample[fill_col].fillna(0, inplace=True)
    except KeyError:
        print('>>Error: col % s already deleted' % fill_col)
    print('>>dropped % s columns' % str(dropped))


def fill(*kargs, method=None, col=None):
    if not method:
        for arg in kargs:
            try:
                model_sample[arg].fillna(0, inplace=True)
            except KeyError:
                print('>>Error: col % s already deleted' % arg)
    elif method == 'avg':
        model_sample[col].fillna(model_sample[col].mean(), inplace=True)
    elif method == 'mode':
        from scipy.stats import mode
        model_sample[col].fillna(mode(model_sample[col]).mode[0], inplace=True)


def drop(*kargs):
    dropped = []
    for arg in kargs:
        try:
            model_sample.drop(arg, axis=1, inplace=True)
            dropped.append(arg)
        except KeyError and ValueError:
            print('>>Error: col % s already deleted' % arg)
    print('>>dropped % s columns' % str(dropped))


drop_and_fill('x_143', 'x_144', 'x_145', 'x_146')
drop_and_fill('x_185', 'x_186', 'x_187')
drop_and_fill('x_125', 'x_121', 'x_122', 'x_123', 'x_124', 'x_126', 'x_127')
drop_and_fill('x_078', 'x_074', 'x_075', 'x_076', 'x_077', 'x_078', 'x_079', 'x_080')
drop_and_fill('x_059', 'x_055', 'x_056', 'x_057', 'x_058', 'x_059', 'x_060', 'x_061')
drop_and_fill('x_045', 'x_046', 'x_047', 'x_041', 'x_042', 'x_043', 'x_044')
fill('x_148', 'x_147', 'x_131', 'x_143', 'x_138', 'x_158', 'x_171', 'x_184')
fill(col='x_001', method='mode')
fill(col='x_002', method='avg')

# generate feature
# model_sample['30_apply_rate'] = model_sample['x_189']/model_sample['x_188']
model_sample['90_apply_rate'] = model_sample['x_193']/model_sample['x_192']
# model_sample['180_apply_rate'] = model_sample['x_197']/model_sample['x_196']
# model_sample['30_success_rate'] = model_sample['x_191']/model_sample['x_190']
model_sample['90_success_rate'] = model_sample['x_195']/model_sample['x_194']
# model_sample['180_success_rate'] = model_sample['x_199']/model_sample['x_198']

fill('30_apply_rate')
fill('90_apply_rate')
fill('180_apply_rate')
fill('30_success_rate')
fill('90_success_rate')
fill('180_success_rate')

# drop cols
drop_cols = [
    'x_132', 'x_133', 'x_134', 'x_135', 'x_136', 'x_137', 'x_139', 'x_140', 'x_141', 'x_142', 'x_144', 'x_149',
    'x_150', 'x_151', 'x_152', 'x_153', 'x_154', 'x_155', 'x_156', 'x_157', 'x_159', 'x_160', 'x_161', 'x_162',
    'x_163', 'x_164', 'x_165', 'x_166', 'x_167', 'x_168', 'x_169', 'x_170', 'x_172', 'x_173', 'x_174', 'x_175',
    'x_176', 'x_177', 'x_178', 'x_179', 'x_180', 'x_181', 'x_182', 'x_183', 'x_188', 'x_189', 'x_190', 'x_191',
    'x_192', 'x_193', 'x_194', 'x_195', 'x_196', 'x_197', 'x_198', 'x_199'
]
for col in drop_cols:
    drop(col)
print(model_sample.columns)

# correlation
model_sample2 = model_sample.fillna(0)
corrmat = model_sample2.corr()
cols = corrmat.nlargest(15, 'y')['y'].index
cm = np.corrcoef(model_sample2[cols].values.T)
cm_df = pd.DataFrame(cm)
cm_df.index = ['i'+col for col in cols]
cm_df.columns = cols
print(cm_df)

f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()
# y correlation matrix
k = 15  # number of variables for heatmap
cols = corrmat.nlargest(k, 'y')['y'].index
cm = np.corrcoef(model_sample[cols].values.T)
f, ax = plt.subplots(figsize=(12, 9))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                 xticklabels=cols.values)
plt.show()

high_corr_dict = {}


def get_high_corr():
    for i in cm_df.columns:
        for j in cm_df.index:
            if cm_df[i].loc[j] > 0.8 and str(j)[1:] != str(i):
                if i not in high_corr_dict.keys():
                    high_corr_dict[i] = []
                    high_corr_dict[i].append(j)
                else:
                    high_corr_dict[i].append(j)

    print(high_corr_dict)


get_high_corr()

# for lis in high_corr_dict.values():
#     for i in lis:
#         col = i[1:] if i[0] == 'i' else i
#         drop(col)
drop('x_158', 'x_018')
print(model_sample.isnull().sum().max())  # 0 all missing value have been processed
print(model_sample.columns)

# convert dtypes
# print(model_sample.dtypes,  model_sample.info())
# categorical type
categorical_cols = ['y', 'x_001', 'x_003', 'x_004', 'x_005', 'x_006', 'x_007', 'x_008', 'x_009', 'x_010',
                    'x_011', 'x_013', 'x_014', 'x_015', 'x_016', 'x_017', 'x_027', 'x_033']
for col in categorical_cols:
    try:
        model_sample[col] = model_sample[col].astype('category')
    except KeyError:
        continue

print(model_sample[categorical_cols].describe())

# feature engineering

# take a glimpse of y
print(model_sample['y'].describe())
mean = (model_sample.y.values == 1).mean()
print(mean)
plt.figure(figsize=(6, 6))
ax = sns.barplot(['overdue(1)', 'Not overdue(0)'], [mean, 1-mean])
ax.set(ylabel='Proportion', title='overdue vs Not overdue')
for p, uniq in zip(ax.patches, [mean, 1-mean]):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2., height+0.01, '{}%'.format(round(uniq * 100, 2)), ha="center")
plt.show()

# visual unique values: method 1
# plt.style.use('fivethirtyeight')
uniques = [len(model_sample[col].unique()) for col in categorical_cols]
print(uniques)
unique_dict = {'feature': categorical_cols, 'unique_values': uniques}
unique_data = pd.DataFrame(unique_dict).set_index('feature')
print(unique_data)
# unique_data.plot(kind='bar', rot=45)
# plt.xlabel('Feature')
# plt.ylabel('unique value numbers')
# plt.title('Number of unique values per feature')
# plt.show()

# visual unique values: method 2
uniques = [len(model_sample[col].unique()) for col in categorical_cols]
# sns.set(font_scale=1.2)
plt.figure(figsize=(10, 6))
ax = sns.barplot(categorical_cols, uniques, palette=pal, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature')
# add text: method 1
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2., uniq, uniq, ha="center")
# add text: method 2
for col, uniq in zip(categorical_cols, uniques):
    ax.text(categorical_cols.index(col), uniq, uniq, color='black', ha="center")
plt.show()

# feature relation analysis

# categorical feature: mosaic plot


def cate_relation(var):
    from statsmodels.graphics.mosaicplot import mosaic
    data = pd.concat([model_sample['y'], model_sample[var]], axis=1)
    try:
        mosaic(data, [var, 'y'], title='% s vs y' % var)
    except ValueError:
        print('>>error: % s plot failed' % var)
    plt.show()


# for var in categorical_cols[1:]:
#     cate_relation(var)


# drop y
y = model_sample['y']
model_sample.drop(['y'],  axis=1,  inplace=True)

# print(model_sample.shape, model_sample.dtypes)
model_sample = pd.get_dummies(model_sample)
print(model_sample.head())
# print(model_sample.shape, model_sample.dtypes)


# start building model
X_train, X_test, y_train, y_test = train_test_split(model_sample, y, test_size=0.2, random_state=0)
dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)
del X_train, X_test, y_train
gc.collect()
params = {'eta': 0.3,
          'tree_method': "hist",
          'grow_policy': "lossguide",
          'max_leaves': 1400,
          'max_depth': 5,
          'subsample': 0.9,
          'colsample_bytree': 0.7,
          'colsample_bylevel': 0.7,
          'min_child_weight': 0,
          'alpha': 4,
          'objective': 'binary:logistic',
          'scale_pos_weight': float(np.sum(dtrain.get_label() == 0)) / np.sum(dtrain.get_label() == 1),
          'eval_metric': 'auc',
          'nthread': 8,
          'random_state': 99,
          'silent': True}
watchlist = [(dtrain, 'train'), (dtest, 'test')]
model = xgb.train(params, dtrain, 200, watchlist, maximize=True, early_stopping_rounds=25, verbose_eval=5)
xgb.plot_importance(model)
plt.show()
y_prediction = model.predict(dtest, ntree_limit=model.best_ntree_limit)
print(y_prediction)


