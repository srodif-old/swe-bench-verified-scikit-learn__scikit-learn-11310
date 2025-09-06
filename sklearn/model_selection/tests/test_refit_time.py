"""Test for refit_time_ attribute in BaseSearchCV classes"""

import time
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class MockClassifier(BaseEstimator, ClassifierMixin):
    """Mock classifier for testing"""
    
    def __init__(self, param=1):
        self.param = param
        
    def fit(self, X, y):
        # Simulate some work proportional to param
        time.sleep(0.001 * self.param)
        self.fitted_ = True
        self.classes_ = np.unique(y)
        return self
        
    def predict(self, X):
        return np.zeros(X.shape[0], dtype=int)
        
    def score(self, X, y):
        # Return score that varies with param
        return 0.7 + self.param * 0.05


def test_refit_time_grid_search():
    """Test refit_time_ attribute in GridSearchCV"""
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, 20)
    
    # Test with refit=True
    param_grid = {'param': [1, 2, 3]}
    gs = GridSearchCV(MockClassifier(), param_grid, cv=2, refit=True)
    
    start_time = time.time()
    gs.fit(X, y)
    total_time = time.time() - start_time
    
    # Check that refit_time_ exists and is reasonable
    assert hasattr(gs, 'refit_time_'), "refit_time_ should exist when refit=True"
    assert isinstance(gs.refit_time_, float), "refit_time_ should be a float"
    assert gs.refit_time_ > 0, "refit_time_ should be positive"
    assert gs.refit_time_ < total_time, "refit_time_ should be less than total time"
    
    # Test with refit=False
    gs_no_refit = GridSearchCV(MockClassifier(), param_grid, cv=2, refit=False)
    gs_no_refit.fit(X, y)
    
    assert not hasattr(gs_no_refit, 'refit_time_'), "refit_time_ should not exist when refit=False"


def test_refit_time_randomized_search():
    """Test refit_time_ attribute in RandomizedSearchCV"""
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, 20)
    
    # Test with refit=True
    param_dist = {'param': [1, 2, 3, 4]}
    rs = RandomizedSearchCV(MockClassifier(), param_dist, n_iter=3, cv=2, refit=True, random_state=42)
    
    start_time = time.time()
    rs.fit(X, y)
    total_time = time.time() - start_time
    
    # Check that refit_time_ exists and is reasonable
    assert hasattr(rs, 'refit_time_'), "refit_time_ should exist when refit=True"
    assert isinstance(rs.refit_time_, float), "refit_time_ should be a float"
    assert rs.refit_time_ > 0, "refit_time_ should be positive"
    assert rs.refit_time_ < total_time, "refit_time_ should be less than total time"


def test_refit_time_reasonable_values():
    """Test that refit_time_ values are reasonable"""
    X = np.random.rand(30, 3)
    y = np.random.randint(0, 2, 30)
    
    # Test with different param values that take different amounts of time
    param_grid = {'param': [1, 5]}  # param=5 should take longer than param=1
    gs = GridSearchCV(MockClassifier(), param_grid, cv=2, refit=True)
    gs.fit(X, y)
    
    # Refit time should be reasonable compared to individual fit times
    avg_fit_time = np.mean(gs.cv_results_['mean_fit_time'])
    
    # The refit time should be in a reasonable range relative to CV fit times
    assert gs.refit_time_ > 0.3 * avg_fit_time, "refit_time_ should be at least 30% of avg CV fit time"
    assert gs.refit_time_ < 10 * avg_fit_time, "refit_time_ should not be more than 10x avg CV fit time"


def test_refit_time_multimetric():
    """Test refit_time_ with multi-metric scoring"""
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, 20)
    
    # Test with multi-metric scoring
    scoring = ['accuracy', 'precision']
    param_grid = {'param': [1, 2]}
    gs = GridSearchCV(MockClassifier(), param_grid, cv=2, scoring=scoring, refit='accuracy')
    gs.fit(X, y)
    
    # Should still have refit_time_ with multi-metric scoring
    assert hasattr(gs, 'refit_time_'), "refit_time_ should exist with multi-metric scoring"
    assert isinstance(gs.refit_time_, float), "refit_time_ should be a float with multi-metric scoring"
    assert gs.refit_time_ > 0, "refit_time_ should be positive with multi-metric scoring"


if __name__ == "__main__":
    print("Running refit_time_ tests...")
    
    test_refit_time_grid_search()
    print("✓ GridSearchCV refit_time_ test passed")
    
    test_refit_time_randomized_search()
    print("✓ RandomizedSearchCV refit_time_ test passed")
    
    test_refit_time_reasonable_values()
    print("✓ Reasonable values test passed")
    
    test_refit_time_multimetric()
    print("✓ Multi-metric scoring test passed")
    
    print("\n🎉 All refit_time_ tests passed!")