import numpy as np
import pandas as pd
import logging
import io
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from config import DB_URL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Create and return a database connection"""
    try:
        engine = create_engine(DB_URL)
        logger.info("Database connection established successfully")
        return engine
    except Exception as e:
        logger.error(f"Failed to establish database connection: {str(e)}")
        raise

def standardize_features(X):
    """Standardize features using StandardScaler"""
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logger.info("Features standardized successfully")
        return X_scaled
    except Exception as e:
        logger.error(f"Failed to standardize features: {str(e)}")
        raise

def find_optimal_parameters(X_scaled, min_samples_range=(2, 10)):
    """
    Find optimal eps and min_samples values for DBSCAN using the elbow method
    Returns optimal eps and min_samples
    """
    try:
        logger.info("Starting parameter optimization...")
        best_eps = None
        best_min_samples = None
        best_score = float('-inf')
        
        for min_samples in range(min_samples_range[0], min_samples_range[1] + 1):
            logger.info(f"Testing min_samples={min_samples}")
            
            # Find optimal eps for current min_samples
            neighbors = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
            distances, _ = neighbors.kneighbors(X_scaled)
            distances = np.sort(distances[:, min_samples-1])
            
            kneedle = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
            if kneedle.elbow is None:
                logger.warning(f"No elbow found for min_samples={min_samples}")
                continue
                
            eps = distances[kneedle.elbow]
            
            # Calculate silhouette score for current parameters
            from sklearn.metrics import silhouette_score
            from sklearn.cluster import DBSCAN
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)
            
            # Skip if all points are noise or only one cluster
            if len(np.unique(labels)) <= 1:
                logger.warning(f"Invalid clustering for eps={eps:.3f}, min_samples={min_samples}")
                continue
                
            score = silhouette_score(X_scaled, labels)
            logger.info(f"Score for eps={eps:.3f}, min_samples={min_samples}: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_eps = eps
                best_min_samples = min_samples
        
        # If no valid parameters found, use defaults
        if best_eps is None:
            logger.warning("No valid parameters found, using defaults")
            best_eps = 0.3
            best_min_samples = 3
        
        logger.info(f"Optimal parameters found: eps={best_eps:.3f}, min_samples={best_min_samples}")
        return best_eps, best_min_samples
        
    except Exception as e:
        logger.error(f"Error in parameter optimization: {str(e)}", exc_info=True)
        raise

def plot_clusters(df, x_col, y_col, cluster_col, title, xlabel, ylabel):
    """Create a scatter plot of clusters and return as base64 image"""
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(df[x_col], df[y_col], c=df[cluster_col], cmap='plasma', s=60)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.colorbar(label='Cluster No')
        
        # Convert plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        logger.info("Cluster plot generated successfully")
        return img_str
    except Exception as e:
        logger.error(f"Error in plotting clusters: {str(e)}")
        raise 