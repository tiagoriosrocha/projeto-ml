import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted
from itertools import combinations
from scipy.stats import pointbiserialr
import warnings


class FlagSparseFeatures(BaseEstimator, TransformerMixin):
    """
    A simplified transformer to create binary presence flags for specified or highly sparse numeric features.
    Allows choosing columns and whether to keep original sparse columns.
    Optimized to reduce DataFrame fragmentation.
    """
    def __init__(self, sparsity_threshold=0.75, columns_to_transform=None, keep_original=False):
        """
        Initialize the transformer.

        Parameters:
        sparsity_threshold (float): Proportion of zeros to consider a column sparse.
        columns_to_transform (list of str, optional): Specific columns to consider.
                                                    If None, all numeric columns are considered.
        keep_original (bool): If True, keeps original sparse columns; otherwise, drops them.
        """
        self.sparsity_threshold = sparsity_threshold
        self.columns_to_transform = columns_to_transform
        self.keep_original = keep_original

    def fit(self, X, y=None):
        """
        Identify sparse numeric columns to be transformed.
        """
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X)
        else:
            X_df = X
        
        self.feature_names_in_ = np.array(X_df.columns, dtype=object)
        self.sparse_cols_ = []
        
        candidate_cols = []
        if self.columns_to_transform is None:
            candidate_cols = X_df.select_dtypes(include=np.number).columns.tolist()
        else:
            missing_cols = [col for col in self.columns_to_transform if col not in X_df.columns]
            if missing_cols:
                warnings.warn(f"Warning: Columns {missing_cols} not found in input DataFrame during fit. They will be ignored.")
            candidate_cols = [col for col in self.columns_to_transform if col in X_df.columns]

        for col in candidate_cols:
            if pd.api.types.is_numeric_dtype(X_df[col]):
                try:
                    sparsity = (X_df[col] == 0).mean()
                    if sparsity >= self.sparsity_threshold:
                        self.sparse_cols_.append(col)
                except TypeError:
                    # Silently ignore columns where sparsity calculation fails
                    pass 
            
        return self

    def transform(self, X):
        """
        Transform data by adding presence flags and optionally dropping original columns.
        """
        check_is_fitted(self, 'sparse_cols_') 

        if not isinstance(X, pd.DataFrame):
            X_transformed = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        else:
            X_transformed = X.copy()
        
        new_flags_data = {}

        for col in self.sparse_cols_:
            if col in X_transformed.columns:
                flag_name = f"{col}_presence"
                try:
                    new_flags_data[flag_name] = (X_transformed[col] > 0).astype(int)
                except TypeError:
                    # Silently ignore if a column became non-numeric between fit and transform
                    pass
        
        if new_flags_data:
            flags_df = pd.DataFrame(new_flags_data, index=X_transformed.index)
            X_transformed = pd.concat([X_transformed, flags_df], axis=1)

        if not self.keep_original:
            cols_to_drop = [col for col in self.sparse_cols_ if col in X_transformed.columns]
            if cols_to_drop:
                X_transformed = X_transformed.drop(columns=cols_to_drop)
            
        return X_transformed


class OpcodeAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, opcode_categories, drop_original=True):
        self.opcode_categories = opcode_categories
        self.drop_original = drop_original
        self.opcode_weight_prefix = 'Opcode weight '
        self._feature_names_out = []

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.feature_names_in_ = list(X.columns)

        self._original_opcode_cols_processed = set()
        for cat_name, opcode_suffixes_in_cat in self.opcode_categories.items():
            for suffix in opcode_suffixes_in_cat:
                col_name = f'{self.opcode_weight_prefix}{suffix}'
                if col_name in self.feature_names_in_:
                    self._original_opcode_cols_processed.add(col_name)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X_transformed = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            X_transformed = X.copy()

        self._aggregated_feature_names_created = [] # Track actual new columns

        for cat_name, opcode_suffixes_in_cat in self.opcode_categories.items():
            current_cat_opcode_cols_present = []
            for suffix in opcode_suffixes_in_cat:
                col_name = f'{self.opcode_weight_prefix}{suffix}'
                if col_name in X_transformed.columns:
                    current_cat_opcode_cols_present.append(col_name)
            
            if current_cat_opcode_cols_present:
                new_col_name = f'Agg_Opcode_{cat_name}'
                X_transformed[new_col_name] = X_transformed[current_cat_opcode_cols_present].sum(axis=1)
                self._aggregated_feature_names_created.append(new_col_name)
        
        if self.drop_original:
            cols_to_drop = [
                col for col in self._original_opcode_cols_processed if col in X_transformed.columns
            ]
            X_transformed = X_transformed.drop(columns=cols_to_drop, errors='ignore')
            
        self._feature_names_out = list(X_transformed.columns)
        
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        if hasattr(self, '_feature_names_out') and self._feature_names_out:
            return self._feature_names_out
        
        if not hasattr(self, 'feature_names_in_'):
             raise AttributeError(
                "This OpcodeAggregator instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

        temp_output_features = list(self.feature_names_in_)
        
        if self.drop_original:
            temp_output_features = [
                col for col in temp_output_features if col not in self._original_opcode_cols_processed
            ]
        
        for cat_name in self.opcode_categories.keys():
            has_constituent_opcode = False
            for suffix in self.opcode_categories[cat_name]:
                 if f'{self.opcode_weight_prefix}{suffix}' in self.feature_names_in_:
                     has_constituent_opcode = True
                     break
            if has_constituent_opcode:
                 temp_output_features.append(f'Agg_Opcode_{cat_name}')
        
        return list(dict.fromkeys(temp_output_features))


class TreeFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=15, random_state=42):
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        self.feature_names_in_ = X.columns
        self.tree_ = DecisionTreeClassifier(random_state=self.random_state)
        self.tree_.fit(X, y)
        importances = self.tree_.feature_importances_
        indices = np.argsort(importances)[::-1][:self.k]
        self.selected_features_ = self.feature_names_in_[indices]

        return self

    def transform(self, X):

        return X[self.selected_features_]


class SelectiveInteraction(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.interactions_ = []

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        y = pd.Series(y).copy()

        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("Target `y` must be binary (0/1) for point-biserial correlation.")

        self.feature_corr_ = {
            col: pointbiserialr(X[col], y)[0]
            for col in X.columns
        }
        self.interactions_ = [
            (f1, f2)
            for f1, f2 in combinations(X.columns, 2)
            if abs(self.feature_corr_[f1]) >= self.threshold
            and abs(self.feature_corr_[f2]) >= self.threshold
            and np.sign(self.feature_corr_[f1]) != np.sign(self.feature_corr_[f2])
        ]

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        new_features = {
            f"{f1}_x_{f2}": X[f1] * X[f2]
            for f1, f2 in self.interactions_
        }
        X_new = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)

        return X_new


class BytecodeCharacterAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, categories_map, drop_original=True):
        self.categories_map = categories_map
        self.drop_original = drop_original
        self.prefixes = ['Weight bytecode_character_', 'bytecode_character_']
        self._feature_names_out_cache = None
        self._original_cols_processed_during_fit = {} # To store cols identified in fit

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X)
        else:
            X_df = X
        self.feature_names_in_ = list(X_df.columns)
        self._original_cols_processed_during_fit = {prefix: set() for prefix in self.prefixes}

        for prefix in self.prefixes:
            for cat_name, char_suffixes_in_cat in self.categories_map.items():
                for suffix in char_suffixes_in_cat:
                    col_name = f'{prefix}{suffix}'
                    if col_name in self.feature_names_in_:
                        self._original_cols_processed_during_fit[prefix].add(col_name)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            # Try to use feature_names_in_ if available from fit
            columns = getattr(self, 'feature_names_in_', None)
            X_transformed = pd.DataFrame(X, columns=columns)
        else:
            X_transformed = X.copy()

        self._aggregated_feature_names_created_this_transform = []
        original_cols_to_drop_this_transform = set()

        for prefix in self.prefixes:
            # Determine the output feature name prefix (AggBytecodeWeight or AggBytecodeCount)
            agg_feature_prefix = ''
            if prefix == 'Weight bytecode_character_':
                agg_feature_prefix = 'AggBytecodeWeight'
            elif prefix == 'bytecode_character_':
                agg_feature_prefix = 'AggBytecodeCount'
            else:
                continue

            for cat_name, char_suffixes_in_cat in self.categories_map.items():
                current_cat_char_cols_present = []
                for suffix in char_suffixes_in_cat:
                    col_name = f'{prefix}{suffix}'
                    if col_name in X_transformed.columns:
                        current_cat_char_cols_present.append(col_name)
                        original_cols_to_drop_this_transform.add(col_name)
                
                if current_cat_char_cols_present:
                    new_agg_col_name = f'{agg_feature_prefix}_{cat_name}'
                    X_transformed[new_agg_col_name] = X_transformed[current_cat_char_cols_present].sum(axis=1)
                    self._aggregated_feature_names_created_this_transform.append(new_agg_col_name)
        
        if self.drop_original:
            cols_to_drop = [col for col in original_cols_to_drop_this_transform if col in X_transformed.columns]
            X_transformed = X_transformed.drop(columns=cols_to_drop, errors='ignore')
            
        self._feature_names_out_cache = list(X_transformed.columns)
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        if self._feature_names_out_cache is not None:
            return self._feature_names_out_cache
        
        if input_features is None:
            if not hasattr(self, 'feature_names_in_'):
                 raise AttributeError(
                    f"{self.__class__.__name__} has not been fitted. "
                    "Call 'fit' or provide input_features to get_feature_names_out."
                )
            input_features = self.feature_names_in_
        
        output_features = list(input_features)
        
        # Determine which original character columns would be dropped
        original_cols_that_would_be_dropped = set()
        if self.drop_original:
            for prefix in self.prefixes:
                    for suffix in char_suffixes_in_cat:
                        col_name = f'{prefix}{suffix}'
                        if col_name in input_features:
                            original_cols_that_would_be_dropped.add(col_name)
            
            output_features = [col for col in output_features if col not in original_cols_that_would_be_dropped]
        
        for prefix in self.prefixes:
            agg_feature_prefix = ''
            if prefix == 'Weight bytecode_character_': agg_feature_prefix = 'AggBytecodeWeight'
            elif prefix == 'bytecode_character_': agg_feature_prefix = 'AggBytecodeCount'
            else: continue

            for cat_name in self.categories_map.keys():
                has_constituent_char = False
                for suffix in self.categories_map[cat_name]:
                    if f'{prefix}{suffix}' in input_features:
                        has_constituent_char = True
                        break
                if has_constituent_char:
                    output_features.append(f'{agg_feature_prefix}_{cat_name}')
        
        return sorted(list(set(output_features)))

