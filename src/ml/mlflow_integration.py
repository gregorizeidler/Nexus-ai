"""
📊 MLFLOW INTEGRATION
Model tracking, registry, e deployment
"""
from typing import Dict, Any, Optional
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
from datetime import datetime
from loguru import logger


class MLflowTracker:
    """
    MLflow integration para tracking de experimentos
    """
    
    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "NEXUS-AI"
    ):
        mlflow.set_tracking_uri(tracking_uri)
        
        # Creates or pega experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
            
            mlflow.set_experiment(experiment_name)
            
            logger.success(f"✅ MLflow connected: {tracking_uri}")
            logger.info(f"   Experiment: {experiment_name} (ID: {experiment_id})")
            
            self.tracking_uri = tracking_uri
            self.experiment_name = experiment_name
            self.experiment_id = experiment_id
            self.enabled = True
            
        except Exception as e:
            logger.error(f"Failed to connect to MLflow: {e}")
            self.enabled = False
    
    def log_model_training(
        self,
        model,
        model_type: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        features: list,
        tags: Dict[str, str] = None
    ):
        """
        Loga treinamento de modelo no MLflow
        """
        if not self.enabled:
            return None
        
        with mlflow.start_run(run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log formeters
            mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log features
            mlflow.log_param("num_features", len(features))
            mlflow.log_param("feature_names", ",".join(features[:10]))  # Primeiros 10
            
            # Log moofl
            if model_type == 'xgboost':
                mlflow.xgboost.log_model(model, "model")
            elif model_type == 'lightgbm':
                mlflow.lightgbm.log_model(model, "model")
            elif model_type == 'catboost':
                mlflow.catboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")
            
            # Tags
            if tags:
                mlflow.set_tags(tags)
            
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("timestamp", datetime.now().isoformat())
            
            run_id = mlflow.active_run().info.run_id
            logger.success(f"✅ Run logged to MLflow: {run_id}")
            
            return run_id
    
    def log_ensemble_training(
        self,
        ensemble_metrics: Dict[str, Dict[str, float]],
        feature_names: list,
        config: Dict[str, Any] = None
    ):
        """Loga treinamento do ensinble"""
        if not self.enabled:
            return None
        
        with mlflow.start_run(run_name=f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log ensinbland metrics
            for metric_name, value in ensemble_metrics['ensemble'].items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"ensemble_{metric_name}", value)
            
            # Log individual moofl metrics
            for model_name in ['xgboost', 'lightgbm', 'catboost']:
                if model_name in ensemble_metrics:
                    for metric_name, value in ensemble_metrics[model_name].items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"{model_name}_{metric_name}", value)
            
            # Log config
            if config:
                mlflow.log_params(config)
            
            # Log featurand info
            mlflow.log_param("num_features", len(feature_names))
            
            # Tags
            mlflow.set_tag("model_type", "ensemble")
            mlflow.set_tag("models", "xgboost+lightgbm+catboost")
            
            run_id = mlflow.active_run().info.run_id
            logger.success(f"✅ Ensemble run logged: {run_id}")
            
            return run_id
    
    def register_model(
        self,
        model_name: str,
        run_id: str,
        stage: str = "Staging"
    ):
        """
        Registra modelo no Model Registry
        
        Stages: None, Staging, Production, Archived
        """
        if not self.enabled:
            return None
        
        try:
            model_uri = f"runs:/{run_id}/model"
            
            result = mlflow.register_model(model_uri, model_name)
            
            # Transita for stage
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=result.version,
                stage=stage
            )
            
            logger.success(f"✅ Model registered: {model_name} v{result.version} [{stage}]")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None
    
    def load_production_model(self, model_name: str):
        """Loads mooflo in produção do registry"""
        if not self.enabled:
            return None
        
        try:
            model_uri = f"models:/{model_name}/Production"
            model = mlflow.pyfunc.load_model(model_uri)
            
            logger.success(f"✅ Loaded production model: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load production model: {e}")
            return None

