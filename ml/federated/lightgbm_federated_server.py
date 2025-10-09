"""
LightGBM Federated Learning Server

Federated learning server specifically designed for LightGBM models
with optimized aggregation strategies for tree-based models.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Any
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, FitRes, EvaluateRes, Scalar
from flwr.server.client_proxy import ClientProxy
import numpy as np
import pickle
import lightgbm as lgb

# Add project paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'models'))
sys.path.append(os.path.join(current_dir, 'utils'))

from models.lightgbm_model import FederatedLightGBM, aggregate_lightgbm_models

class LightGBMFederatedStrategy(FedAvg):
    """
    Custom Flower strategy for LightGBM federated learning.
    
    Handles aggregation of LightGBM models using model selection
    and ensemble techniques since direct parameter averaging
    is not suitable for tree-based models.
    """
    
    def __init__(self, base_model_path: str, **kwargs):
        """
        Initialize strategy with base LightGBM model.
        
        Args:
            base_model_path: Path to pre-trained base LightGBM model
        """
        super().__init__(**kwargs)
        
        self.base_model_path = base_model_path
        self.logger = logging.getLogger("LightGBMStrategy")
        
        # Load base model
        self.base_model = self.load_base_model()
        
        # Track global model state
        self.current_global_model = None
        self.global_round = 0
        
        self.logger.info("LightGBM federated learning strategy initialized")
    
    def load_base_model(self):
        """Load the pre-trained base LightGBM model."""
        try:
            self.logger.info(f"Loading base LightGBM model from: {self.base_model_path}")
            
            if self.base_model_path.endswith('.pkl'):
                # Load pickled model
                with open(self.base_model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                if isinstance(model_data, dict):
                    # Load our FederatedLightGBM format
                    base_model = FederatedLightGBM()
                    base_model.params = model_data.get('params', base_model.params)
                    base_model.feature_names = model_data.get('feature_names', [])
                    base_model.n_features = model_data.get('n_features', 0)
                    
                    # Create LightGBM model
                    if 'model_dump' in model_data:
                        base_model.model = lgb.LGBMClassifier(**base_model.params)
                        booster = lgb.Booster(model_str=model_data['model_dump'])
                        base_model.model._Booster = booster
                        base_model.is_fitted = True
                    
                    return base_model
                else:
                    # Handle sklearn LGBMClassifier directly
                    base_model = FederatedLightGBM()
                    base_model.model = model_data
                    base_model.is_fitted = True
                    
                    # Extract feature information if available
                    if hasattr(model_data, 'feature_name_'):
                        base_model.feature_names = list(model_data.feature_name_)
                        base_model.n_features = len(base_model.feature_names)
                    
                    return base_model
            
            else:
                # Load LightGBM model file
                base_model = FederatedLightGBM()
                base_model.model = lgb.Booster(model_file=self.base_model_path)
                base_model.is_fitted = True
                return base_model
                
        except Exception as e:
            self.logger.error(f"Failed to load base model: {e}")
            # Return empty model
            return FederatedLightGBM()
    
    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """Initialize global model parameters using the base model."""
        self.logger.info("Initializing global parameters with base LightGBM model")
        
        if self.base_model and self.base_model.is_fitted:
            model_params = self.base_model.get_model_params()
            if model_params:
                # Convert to Flower parameters format
                model_dump = model_params.get('model_dump', '')
                if model_dump:
                    model_bytes = model_dump.encode('utf-8')
                    param_array = np.frombuffer(model_bytes, dtype=np.uint8).astype(np.float32)
                    return fl.common.ndarrays_to_parameters([param_array])
        
        # Return empty parameters if no base model
        return fl.common.ndarrays_to_parameters([np.array([0.0], dtype=np.float32)])
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List,
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates from LightGBM clients."""
        
        self.logger.info(f"\n[ROUND] LightGBM FL Round {server_round} - Aggregating results from {len(results)} clients")
        
        if failures:
            self.logger.warning(f"[WARN] Encountered {len(failures)} failures during training")
        
        # Extract FitRes objects and filter out insufficient data responses
        fit_results = []
        insufficient_data_clients = 0
        
        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                client_proxy, fit_res = result
                # Check if client had insufficient data
                if (hasattr(fit_res, 'metrics') and 
                    fit_res.metrics and 
                    fit_res.metrics.get('status') == 'insufficient_data'):
                    insufficient_data_clients += 1
                    self.logger.info(f"[FAIL] Client {client_proxy.cid if hasattr(client_proxy, 'cid') else 'Unknown'}: Insufficient data ({fit_res.metrics.get('data_count', 0)} < 5 transactions)")
                else:
                    fit_results.append(fit_res)
            else:
                fit_results.append(result)
        
        if insufficient_data_clients > 0:
            self.logger.info(f"[BLOCKED] {insufficient_data_clients} client(s) blocked due to insufficient data (< 5 transactions)")
        
        # Handle case where no clients have sufficient data after filtering
        if not fit_results:
            self.logger.info(f"[WAIT] Round {server_round}: No clients with sufficient data (>= 5 transactions)")
            if server_round % 5 == 0:  # Only log every 5 rounds to reduce noise
                self.logger.info("[WAITING] Server waiting for clients to reach data threshold...")
            
            # Return current global parameters
            if self.current_global_model:
                return self.current_global_model, {"status": "waiting_for_sufficient_data", "round": server_round}
            else:
                return self.initialize_parameters(None), {"status": "waiting_for_sufficient_data", "round": server_round}
        
        # Log clients with sufficient data
        self.logger.info(f"[SUCCESS] {len(fit_results)} client(s) have sufficient data - proceeding with FL round")
        
        try:
            # For LightGBM, we use a different aggregation strategy
            # Since we can't average tree parameters, we select the best model
            # based on training data size or use ensemble approaches
            
            # Extract model parameters and weights
            model_params_list = []
            weights = []
            
            self.logger.info(f"DEBUG: Processing {len(fit_results)} fit results for aggregation")
            
            for i, fit_res in enumerate(fit_results):
                self.logger.info(f"DEBUG: Processing client {i}: {fit_res.num_examples} examples")
                if fit_res.num_examples > 0:
                    # Decode model parameters
                    try:
                        param_arrays = fl.common.parameters_to_ndarrays(fit_res.parameters)
                        self.logger.info(f"DEBUG: Client {i} sent {len(param_arrays)} parameter arrays")
                        
                        if len(param_arrays) > 0 and len(param_arrays[0]) > 1:
                            self.logger.info(f"DEBUG: Client {i} parameter array size: {len(param_arrays[0])}")
                            
                            # Convert float32 array back to bytes
                            byte_array = param_arrays[0].astype(np.uint8)
                            model_bytes = byte_array.tobytes()
                            
                            # Decode JSON string
                            import json
                            json_str = model_bytes.decode('utf-8', errors='ignore')
                            
                            self.logger.info(f"DEBUG: Client {i} decoded JSON length: {len(json_str)}")
                            
                            if json_str and len(json_str) > 10:
                                try:
                                    # Parse JSON to get full model state
                                    model_state = json.loads(json_str)
                                    
                                    self.logger.info(f"DEBUG: Client {i} model type: {model_state.get('model_type', 'unknown')}")
                                    
                                    if model_state.get('model_type') == 'lightgbm':
                                        # Add client-specific metadata
                                        model_state['client_samples'] = fit_res.num_examples
                                        model_state['client_id'] = getattr(fit_res, 'cid', 'unknown')
                                        
                                        model_params_list.append(model_state)
                                        weights.append(fit_res.num_examples)
                                        
                                        self.logger.info(f"DEBUG: Successfully added client {i} model to aggregation")
                                    else:
                                        self.logger.warning(f"Received non-LightGBM model parameters from client {i}")
                                        
                                except json.JSONDecodeError as je:
                                    self.logger.warning(f"Failed to parse JSON model state from client {i}: {je}")
                            else:
                                self.logger.warning(f"Client {i} JSON string too short or empty")
                        else:
                            self.logger.warning(f"Client {i} sent empty or invalid parameter array")
                            
                    except Exception as e:
                        self.logger.warning(f"Failed to decode model parameters from client {i}: {e}")
                else:
                    self.logger.warning(f"Client {i} has no training examples")
            
            # Aggregate models (select best or ensemble)
            self.logger.info(f"DEBUG: Attempting aggregation with {len(model_params_list)} model(s), weights: {weights}")
            
            if model_params_list:
                from models.lightgbm_model import aggregate_lightgbm_models
                aggregated_params = aggregate_lightgbm_models(model_params_list, weights)
                self.logger.info(f"DEBUG: Aggregation result: {'Success' if aggregated_params else 'Failed'}")
                
                if aggregated_params:
                    # Convert back to Flower parameters format
                    try:
                        import json
                        # Serialize the full aggregated model state as JSON
                        json_str = json.dumps(aggregated_params, default=str)
                        model_bytes = json_str.encode('utf-8')
                        
                        # Convert to float32 array for Flower
                        byte_array = np.frombuffer(model_bytes, dtype=np.uint8)
                        param_array = byte_array.astype(np.float32)
                        aggregated_parameters = fl.common.ndarrays_to_parameters([param_array])
                        
                        self.logger.debug(f"Encoded aggregated parameters: {len(param_array)} values")
                    except Exception as encode_e:
                        self.logger.warning(f"Failed to encode aggregated parameters: {encode_e}")
                        aggregated_parameters = self.current_global_model
                    
                    # Update current global model
                    self.current_global_model = aggregated_parameters
                    
                    # Save model periodically
                    total_examples = sum(weights)
                    if server_round == 1 or server_round % 3 == 0:
                        try:
                            model_path = os.path.join(os.path.dirname(self.base_model_path), "latest_lightgbm_federated.pkl")
                            self.save_global_model(aggregated_params, model_path, server_round)
                            self.logger.info(f"[SAVED] Model saved: Round {server_round}, {total_examples:,} examples")
                        except Exception as e:
                            self.logger.warning(f"Failed to save model: {e}")
                    else:
                        self.logger.info(f"[COMPLETE] Training completed: Round {server_round}, {total_examples:,} examples")
                    
                    # Calculate aggregated metrics
                    total_examples = sum([res.num_examples for res in fit_results])
                    avg_accuracy = 0.0
                    
                    if fit_results:
                        accuracies = [res.metrics.get("accuracy", 0.0) * res.num_examples 
                                    for res in fit_results if hasattr(res, 'metrics') and res.metrics]
                        if accuracies:
                            avg_accuracy = sum(accuracies) / total_examples
                    
                    aggregated_metrics = {
                        "round": server_round,
                        "total_examples": total_examples,
                        "avg_accuracy": avg_accuracy,
                        "participating_clients": len(fit_results)
                    }
                    
                    self.logger.info(f"Round {server_round} aggregation completed:")
                    self.logger.info(f"  Total training examples: {total_examples:,}")
                    self.logger.info(f"  Average accuracy: {avg_accuracy:.4f}")
                    
                    return aggregated_parameters, aggregated_metrics
            
            # Fallback: return current global model
            self.logger.warning(f"DEBUG: No models to aggregate - model_params_list is empty")
            return self.current_global_model or self.initialize_parameters(None), {"status": "aggregation_failed", "round": server_round}
            
        except Exception as e:
            self.logger.error(f"Aggregation failed: {e}")
            return self.current_global_model or self.initialize_parameters(None), {"status": "aggregation_error", "round": server_round}
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List,
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results from clients."""
        
        self.logger.info(f"Aggregating evaluation results from {len(results)} clients (Round {server_round})")
        
        if failures:
            self.logger.warning(f"Encountered {len(failures)} failures during evaluation")
        
        if not results:
            return None, {}
        
        # Extract EvaluateRes objects from tuples if needed
        eval_results = []
        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                client_proxy, eval_res = result
                eval_results.append(eval_res)
            else:
                eval_results.append(result)
        
        # Calculate aggregated metrics
        try:
            total_examples = sum([res.num_examples for res in eval_results])
            
            if total_examples == 0:
                return 0.0, {}
            
            # Weighted average of losses
            weighted_losses = [res.loss * res.num_examples for res in eval_results]
            aggregated_loss = sum(weighted_losses) / total_examples
            
            # Aggregate other metrics
            aggregated_metrics = {
                "eval_round": server_round,
                "eval_examples": total_examples,
                "eval_loss": aggregated_loss
            }
            
            # Calculate accuracy if available
            if eval_results and hasattr(eval_results[0], 'metrics') and eval_results[0].metrics:
                accuracies = [res.metrics.get("eval_accuracy", 0.0) * res.num_examples 
                            for res in eval_results if hasattr(res, 'metrics') and res.metrics]
                if accuracies:
                    avg_accuracy = sum(accuracies) / total_examples
                    aggregated_metrics["eval_accuracy"] = avg_accuracy
            
            self.logger.info(f"Round {server_round} evaluation completed:")
            self.logger.info(f"  Total evaluation examples: {total_examples:,}")
            self.logger.info(f"  Average evaluation loss: {aggregated_loss:.6f}")
            
            return aggregated_loss, aggregated_metrics
            
        except Exception as e:
            self.logger.warning(f"Evaluation aggregation failed: {e}")
            return 1.0, {"eval_error": str(e)}
    
    def save_global_model(self, model_params: Dict, save_path: str, round_num: int):
        """Save the global model to file."""
        try:
            model_data = {
                'model_dump': model_params.get('model_dump', ''),
                'round': round_num,
                'total_samples': model_params.get('total_samples', 0),
                'last_updated': round_num
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(model_data, f)
                
        except Exception as e:
            self.logger.error(f"Failed to save global model: {e}")

def load_base_lightgbm_model(trained_models_dir: str):
    """Find and load the base LightGBM model."""
    
    # Look for LightGBM model files
    possible_names = [
        "lightgbm_model.pkl",
        "latest_lightgbm_model.pkl", 
        "base_lightgbm_model.pkl",
        "lightgbm.pkl"
    ]
    
    for name in possible_names:
        model_path = os.path.join(trained_models_dir, name)
        if os.path.exists(model_path):
            return model_path
    
    raise FileNotFoundError("No LightGBM base model found. Expected files: " + ", ".join(possible_names))

def start_lightgbm_federated_server(
    server_address: str = "localhost:8080",
    num_rounds: int = 50,
    min_clients: int = 1,
    trained_models_dir: str = None
):
    """
    Start the LightGBM federated learning server.
    
    Args:
        server_address: Server address and port
        num_rounds: Number of federated learning rounds
        min_clients: Minimum number of clients required
        trained_models_dir: Directory containing trained base model
    """
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("LightGBMServer")
    
    print("[STAR] Starting LightGBM Federated Learning Server")
    print("=" * 60)
    
    # Default trained models directory
    if trained_models_dir is None:
        trained_models_dir = os.path.join(os.path.dirname(__file__), "trained_models")
    
    try:
        # Load base model
        model_path = load_base_lightgbm_model(trained_models_dir)
        
        print(f"[AI] Base LightGBM model: {model_path}")
        print(f"[NET] Server address: {server_address}")
        print(f"[MODE] Optimized operation ({num_rounds} rounds)")
        print(f"[CLIENTS] Minimum clients: {min_clients} (server waits for any client)")
        print(f"[THRESHOLD] Clients need >=5 transactions for FL participation")
        print(f"[MODEL] LightGBM Tree-based Fraud Detection")
        
        # Create LightGBM strategy
        strategy = LightGBMFederatedStrategy(
            base_model_path=model_path,
            min_fit_clients=min_clients,
            min_evaluate_clients=0,  # Skip evaluation to avoid stopping server
            min_available_clients=min_clients,
        )
        
        print("\n[LAUNCH] Initializing LightGBM server...")
        
        # Start Flower server
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy
        )
        
        # Server completed
        print("\n[DONE] Server completed all rounds or stopped by user")
        print("[SAVE] Global LightGBM model automatically saved during successful training rounds")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Please ensure you have a trained LightGBM model in the trained_models/ directory.")
        print("Expected files: lightgbm_model.pkl, latest_lightgbm_model.pkl, etc.")
        
    except Exception as e:
        print(f"\n[FAIL] Server failed to start: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main server initialization function."""
    
    # Add config path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))
    
    try:
        from config.federated_config import get_server_config
        server_config = get_server_config()
        SERVER_ADDRESS = server_config["server_address"]
        NUM_ROUNDS = server_config["num_rounds"]
        MIN_CLIENTS = server_config["min_clients"]
        
        print(f"[CONFIG] Loaded configuration: {NUM_ROUNDS} rounds, optimized for LightGBM")
    except ImportError:
        # Fallback to defaults if config not available
        SERVER_ADDRESS = "localhost:8080"
        NUM_ROUNDS = 50
        MIN_CLIENTS = 1
        print("[WARN] Using fallback configuration for LightGBM operation")
    
    # Path to trained models
    trained_models_dir = os.path.join(os.path.dirname(__file__), "trained_models")
    
    print("[BANK] LightGBM Federated Learning for Banking Fraud Detection")
    print("Using LightGBM Tree-based Models with Flower Framework")
    print("[CONTINUOUS] Server runs for optimized rounds")
    print("[STRICT] Clients need >=5 transactions for participation")
    print("[PERSISTENT] Server continues running even with insufficient client data")
    print()
    
    # Check if base model exists
    if not os.path.exists(trained_models_dir):
        print("[ERROR] No trained models directory found!")
        print("Please ensure you have a trained LightGBM model in the trained_models/ directory.")
        return
    
    # Start server
    start_lightgbm_federated_server(
        server_address=SERVER_ADDRESS,
        num_rounds=NUM_ROUNDS,
        min_clients=MIN_CLIENTS,
        trained_models_dir=trained_models_dir
    )

if __name__ == "__main__":
    main()
