"""
🎯 OPTUNA HYPERPARAMETER TUNING
Otimização automática de hyperparameters
"""
from typing import Dict, Any, Optional
import optuna
from optuna.integration import XGBoostPruningCallback, LightGBMPruningCallback
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from loguru import logger


class OptunaOptimizer:
    """
    Otimização de hyperparameters com Optuna
    """
    
    def __init__(
        self,
        n_trials: int = 100,
        timeout: Optional[int] = 3600,  # 1 hour
        study_name: str = "nexus-ai"
    ):
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name
        
        # Creates or Loads study
        self.study = optuna.create_study(
            study_name=study_name,
            direction='maximize',  # Maximize AUC
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
        )
        
        logger.info(f"🎯 Optuna Optimizer initialized: {study_name}")
    
    def optimize_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """
        Otimiza hyperparameters do XGBoost
        """
        logger.info("🚀 Starting XGBoost optimization...")
        
        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                'random_state': 42,
                'tree_method': 'hist'
            }
            
            model = xgb.XGBClassifier(**params)
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[XGBoostPruningCallback(trial, 'validation_0-auc')],
                verbose=False
            )
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            
            return auc
        
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        logger.success(f"✅ XGBoost optimization complete!")
        logger.success(f"   Best AUC: {best_value:.4f}")
        logger.success(f"   Best params: {best_params}")
        
        return {
            'best_params': best_params,
            'best_auc': best_value,
            'n_trials': len(self.study.trials)
        }
    
    def optimize_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """Otimiza LightGBM"""
        logger.info("🚀 Starting LightGBM optimization...")
        
        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                'random_state': 42,
                'verbose': -1
            }
            
            model = lgb.LGBMClassifier(**params)
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(50),
                    LightGBMPruningCallback(trial, 'auc')
                ]
            )
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            
            return auc
        
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        logger.success(f"✅ LightGBM optimization complete! Best AUC: {best_value:.4f}")
        
        return {
            'best_params': best_params,
            'best_auc': best_value,
            'n_trials': len(self.study.trials)
        }
    
    def optimize_catboost(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """Otimiza CatBoost"""
        logger.info("🚀 Starting CatBoost optimization...")
        
        def objective(trial):
            params = {
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'random_seed': 42,
                'verbose': False
            }
            
            model = cb.CatBoostClassifier(**params)
            
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=False
            )
            
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            
            return auc
        
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        logger.success(f"✅ CatBoost optimization complete! Best AUC: {best_value:.4f}")
        
        return {
            'best_params': best_params,
            'best_auc': best_value,
            'n_trials': len(self.study.trials)
        }
    
    def get_best_trial(self) -> Dict[str, Any]:
        """Returns melhor trial"""
        trial = self.study.best_trial
        
        return {
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
            'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            'duration': (trial.datetime_complete - trial.datetime_start).total_seconds() if trial.datetime_complete and trial.datetime_start else None
        }
    
    def visualize_optimization(self):
        """Generates visualizações do processo of otimização"""
        try:
            # Optimization history
            fig1 = optuna.visualization.plot_optimization_history(self.study)
            fig1.write_html(f"reports/optuna_history_{self.study_name}.html")
            
            # formeter importances
            fig2 = optuna.visualization.plot_param_importances(self.study)
            fig2.write_html(f"reports/optuna_importance_{self.study_name}.html")
            
            # forllel coordinatand plot
            fig3 = optuna.visualization.plot_parallel_coordinate(self.study)
            fig3.write_html(f"reports/optuna_parallel_{self.study_name}.html")
            
            logger.success("✅ Optimization visualizations saved to reports/")
            
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")

