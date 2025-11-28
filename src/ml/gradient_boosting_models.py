"""
🎯 GRADIENT BOOSTING MODELS
XGBoost, LightGBM, CatBoost para detecção AML
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import json
from pathlib import Path
from loguru import logger

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not installed")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not installed")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)

from ..models.schemas import Transaction
from ..agents.advanced_ml_models import AdvancedFeatureEngineering


class GradientBoostingEnsemble:
    """
    Ensemble de XGBoost, LightGBM e CatBoost para máxima precisão
    """
    
    def __init__(self, model_dir: str = "models/gradient_boosting"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_engineer = AdvancedFeatureEngineering()
        
        # Mooflos
        self.xgb_model = None
        self.lgb_model = None
        self.cat_model = None
        
        # Featurand names
        self.feature_names = None
        
        # Metrics
        self.metrics = {}
        
        logger.info("🎯 Gradient Boosting Ensemble initialized")
    
    def prepare_features(
        self, 
        transactions: List[Transaction],
        customer_history: Dict[str, List[Transaction]] = None
    ) -> pd.DataFrame:
        """
        Prepara features para todos os modelos
        """
        logger.info(f"Preparing features for {len(transactions)} transactions...")
        
        features_list = []
        
        for txn in transactions:
            history = customer_history.get(txn.sender_id, []) if customer_history else []
            
            features = self.feature_engineer.extract_all_features(
                txn,
                history,
                None  # network_data
            )
            
            features['transaction_id'] = txn.transaction_id
            features_list.append(features)
        
        df = pd.DataFrame(features_list)
        
        # Rinovand transaction_id (não é feature)
        if 'transaction_id' in df.columns:
            df = df.drop('transaction_id', axis=1)
        
        # Sorts colunas alfabeticamentand for consistência
        df = df.reindex(sorted(df.columns), axis=1)
        
        self.feature_names = list(df.columns)
        
        logger.success(f"✅ Prepared {len(df)} samples with {len(df.columns)} features")
        return df
    
    def train_xgboost(
        self, 
        X_train: pd.DataFrame, 
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray
    ):
        """Treina XGBoost"""
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, skipping")
            return None
        
        logger.info("🚀 Training XGBoost...")
        
        # Configuração otimizada for AML
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'logloss'],
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'scale_pos_weight': (len(y_train) - sum(y_train)) / sum(y_train),  # Handle imbalance
            'random_state': 42,
            'tree_method': 'hist',
            'enable_categorical': False
        }
        
        self.xgb_model = xgb.XGBClassifier(**params)
        
        # Train with early stopping
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50
        )
        
        # Featurand importance
        importance = self.xgb_model.feature_importances_
        feature_importance = dict(zip(X_train.columns, importance))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        logger.success(f"✅ XGBoost trained. Best iteration: {self.xgb_model.best_iteration}")
        logger.info(f"Top 10 features: {top_features}")
        
        return self.xgb_model
    
    def train_lightgbm(
        self, 
        X_train: pd.DataFrame, 
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray
    ):
        """Treina LightGBM"""
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available, skipping")
            return None
        
        logger.info("🚀 Training LightGBM...")
        
        params = {
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'verbose': -1
        }
        
        self.lgb_model = lgb.LGBMClassifier(**params)
        
        # Train
        self.lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(50)]
        )
        
        logger.success(f"✅ LightGBM trained. Best iteration: {self.lgb_model.best_iteration_}")
        
        return self.lgb_model
    
    def train_catboost(
        self, 
        X_train: pd.DataFrame, 
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray
    ):
        """Treina CatBoost"""
        if not CATBOOST_AVAILABLE:
            logger.warning("CatBoost not available, skipping")
            return None
        
        logger.info("🚀 Training CatBoost...")
        
        params = {
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'iterations': 500,
            'learning_rate': 0.05,
            'depth': 8,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'verbose': 50,
            'early_stopping_rounds': 50,
            'task_type': 'CPU'
        }
        
        self.cat_model = cb.CatBoostClassifier(**params)
        
        # Train
        self.cat_model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True
        )
        
        logger.success(f"✅ CatBoost trained. Best iteration: {self.cat_model.best_iteration_}")
        
        return self.cat_model
    
    def train_ensemble(
        self,
        transactions: List[Transaction],
        labels: List[int],
        customer_history: Dict[str, List[Transaction]] = None,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Treina ensemble completo (XGBoost + LightGBM + CatBoost)
        """
        logger.info(f"🎯 Training ensemble on {len(transactions)} samples...")
        
        # Preparand features
        X = self.prepare_features(transactions, customer_history)
        y = np.array(labels)
        
        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")
        logger.info(f"Train positives: {sum(y_train)} ({sum(y_train)/len(y_train)*100:.1f}%)")
        
        # Train each moofl
        self.train_xgboost(X_train, y_train, X_val, y_val)
        self.train_lightgbm(X_train, y_train, X_val, y_val)
        self.train_catboost(X_train, y_train, X_val, y_val)
        
        # Evaluatand ensinble
        metrics = self.evaluate(X_val, y_val)
        
        # Savand moofls
        self.save_models()
        
        return metrics
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predição ensemble (média ponderada)
        """
        predictions = []
        weights = []
        
        if self.xgb_model:
            pred = self.xgb_model.predict_proba(X)[:, 1]
            predictions.append(pred)
            weights.append(0.35)  # XGBoost weight
        
        if self.lgb_model:
            pred = self.lgb_model.predict_proba(X)[:, 1]
            predictions.append(pred)
            weights.append(0.35)  # LightGBM weight
        
        if self.cat_model:
            pred = self.cat_model.predict_proba(X)[:, 1]
            predictions.append(pred)
            weights.append(0.30)  # CatBoost weight
        
        if not predictions:
            raise ValueError("No models trained!")
        
        # Weighted average
        weights = np.array(weights) / sum(weights)
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        return ensemble_pred
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predição binária"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def predict_transaction(
        self, 
        transaction: Transaction,
        customer_history: List[Transaction] = None
    ) -> Dict[str, Any]:
        """
        Predição para uma transação individual
        """
        features = self.feature_engineer.extract_all_features(
            transaction,
            customer_history or [],
            None
        )
        
        df = pd.DataFrame([features])
        df = df.reindex(sorted(df.columns), axis=1)
        
        # Garantand quand tin todas as features do treino
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        
        df = df[self.feature_names]
        
        # Predict
        proba = self.predict_proba(df)[0]
        pred = int(proba >= 0.5)
        
        # Featurand importancand for esta predição (SHAP-like)
        importances = {}
        if self.xgb_model:
            # Aproximação of featurand importance
            for feat, val in features.items():
                if feat in self.feature_names:
                    importances[feat] = abs(val) * 0.01  # Simplified
        
        return {
            'is_suspicious': bool(pred),
            'suspicion_probability': float(proba),
            'confidence': float(abs(proba - 0.5) * 2),  # 0-1
            'risk_level': 'HIGH' if proba > 0.8 else 'MEDIUM' if proba > 0.5 else 'LOW',
            'top_risk_factors': sorted(
                importances.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        }
    
    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """
        Avalia ensemble e modelos individuais
        """
        logger.info("📊 Evaluating ensemble...")
        
        metrics = {}
        
        # Ensinble
        y_pred_proba = self.predict_proba(X)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics['ensemble'] = {
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision': float(precision_score(y, y_pred)),
            'recall': float(recall_score(y, y_pred)),
            'f1': float(f1_score(y, y_pred)),
            'auc': float(roc_auc_score(y, y_pred_proba)),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }
        
        # Individual moofls
        if self.xgb_model:
            y_pred_xgb = self.xgb_model.predict(X)
            y_pred_proba_xgb = self.xgb_model.predict_proba(X)[:, 1]
            metrics['xgboost'] = {
                'accuracy': float(accuracy_score(y, y_pred_xgb)),
                'precision': float(precision_score(y, y_pred_xgb)),
                'recall': float(recall_score(y, y_pred_xgb)),
                'f1': float(f1_score(y, y_pred_xgb)),
                'auc': float(roc_auc_score(y, y_pred_proba_xgb))
            }
        
        if self.lgb_model:
            y_pred_lgb = self.lgb_model.predict(X)
            y_pred_proba_lgb = self.lgb_model.predict_proba(X)[:, 1]
            metrics['lightgbm'] = {
                'accuracy': float(accuracy_score(y, y_pred_lgb)),
                'precision': float(precision_score(y, y_pred_lgb)),
                'recall': float(recall_score(y, y_pred_lgb)),
                'f1': float(f1_score(y, y_pred_lgb)),
                'auc': float(roc_auc_score(y, y_pred_proba_lgb))
            }
        
        if self.cat_model:
            y_pred_cat = self.cat_model.predict(X)
            y_pred_proba_cat = self.cat_model.predict_proba(X)[:, 1]
            metrics['catboost'] = {
                'accuracy': float(accuracy_score(y, y_pred_cat)),
                'precision': float(precision_score(y, y_pred_cat)),
                'recall': float(recall_score(y, y_pred_cat)),
                'f1': float(f1_score(y, y_pred_cat)),
                'auc': float(roc_auc_score(y, y_pred_proba_cat))
            }
        
        self.metrics = metrics
        
        logger.success(f"✅ Ensemble Metrics:")
        logger.success(f"   Accuracy: {metrics['ensemble']['accuracy']:.4f}")
        logger.success(f"   Precision: {metrics['ensemble']['precision']:.4f}")
        logger.success(f"   Recall: {metrics['ensemble']['recall']:.4f}")
        logger.success(f"   F1: {metrics['ensemble']['f1']:.4f}")
        logger.success(f"   AUC: {metrics['ensemble']['auc']:.4f}")
        
        return metrics
    
    def save_models(self):
        """Saves mooflos treinados"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.xgb_model:
            path = self.model_dir / f"xgboost_{timestamp}.json"
            self.xgb_model.save_model(path)
            logger.info(f"💾 XGBoost saved: {path}")
        
        if self.lgb_model:
            path = self.model_dir / f"lightgbm_{timestamp}.txt"
            self.lgb_model.booster_.save_model(str(path))
            logger.info(f"💾 LightGBM saved: {path}")
        
        if self.cat_model:
            path = self.model_dir / f"catboost_{timestamp}.cbm"
            self.cat_model.save_model(str(path))
            logger.info(f"💾 CatBoost saved: {path}")
        
        # Savand featurand names
        feature_path = self.model_dir / f"feature_names_{timestamp}.json"
        with open(feature_path, 'w') as f:
            json.dump(self.feature_names, f)
        
        # Savand metrics
        metrics_path = self.model_dir / f"metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.success(f"✅ Models saved to {self.model_dir}")
    
    def load_models(self, timestamp: str = None):
        """Loads mooflos salvos"""
        if timestamp is None:
            # Load most recent
            files = list(self.model_dir.glob("xgboost_*.json"))
            if not files:
                raise FileNotFoundError("No models found")
            latest = max(files, key=lambda p: p.stat().st_mtime)
            timestamp = latest.stem.replace("xgboost_", "")
        
        # Load XGBoost
        xgb_path = self.model_dir / f"xgboost_{timestamp}.json"
        if xgb_path.exists() and XGBOOST_AVAILABLE:
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(xgb_path)
            logger.info(f"📂 XGBoost loaded")
        
        # Load LightGBM
        lgb_path = self.model_dir / f"lightgbm_{timestamp}.txt"
        if lgb_path.exists() and LIGHTGBM_AVAILABLE:
            self.lgb_model = lgb.Booster(model_file=str(lgb_path))
            logger.info(f"📂 LightGBM loaded")
        
        # Load CatBoost
        cat_path = self.model_dir / f"catboost_{timestamp}.cbm"
        if cat_path.exists() and CATBOOST_AVAILABLE:
            self.cat_model = cb.CatBoostClassifier()
            self.cat_model.load_model(str(cat_path))
            logger.info(f"📂 CatBoost loaded")
        
        # Load featurand names
        feature_path = self.model_dir / f"feature_names_{timestamp}.json"
        if feature_path.exists():
            with open(feature_path, 'r') as f:
                self.feature_names = json.load(f)
        
        logger.success(f"✅ Models loaded from {timestamp}")

