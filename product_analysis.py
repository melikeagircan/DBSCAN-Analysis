import pandas as pd
import logging
from sklearn.cluster import DBSCAN
from utils import get_db_connection, standardize_features, find_optimal_parameters, plot_clusters

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_products():
    """Analyze product performance using DBSCAN clustering with optimized parameters"""
    try:
        logger.info("Starting product analysis...")
        engine = get_db_connection()
        logger.info("Database connection established")
        
        query = """
        SELECT 
            p.product_id,
            p.product_name,
            CAST(AVG(od.unit_price) AS FLOAT) as average_sale_price,
            CAST(SUM(od.quantity) AS INTEGER) as total_quantity_sold,
            CAST(AVG(od.quantity) AS FLOAT) as average_quantity_per_order,
            CAST(COUNT(DISTINCT o.customer_id) AS INTEGER) as unique_customers
        FROM products p
        JOIN order_details od ON p.product_id = od.product_id
        JOIN orders o ON od.order_id = o.order_id
        GROUP BY p.product_id, p.product_name
        ORDER BY p.product_id
        """
        
        logger.info("Executing SQL query...")
        df = pd.read_sql(query, engine)
        logger.info(f"Retrieved {len(df)} products from database")
        
        if df.empty:
            raise ValueError("No product data found in the database")
        
        # Ensure numeric columns are properly typed
        numeric_columns = ["average_sale_price", "total_quantity_sold", "average_quantity_per_order", "unique_customers"]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        if df.empty:
            raise ValueError("No valid numeric data found after cleaning")
        
        # Prepare features
        X = df[numeric_columns]
        logger.info("Standardizing features...")
        X_scaled = standardize_features(X)
        
        # Find optimal parameters
        logger.info("Finding optimal DBSCAN parameters...")
        eps, min_samples = find_optimal_parameters(X_scaled)
        logger.info(f"Optimal parameters found: eps={eps}, min_samples={min_samples}")
        
        # Apply DBSCAN with optimal parameters
        logger.info("Applying DBSCAN clustering...")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        df["cluster"] = dbscan.fit_predict(X_scaled)
        
        # Generate plot as base64
        logger.info("Generating cluster visualization...")
        plot_title = f"Product Segmentation (DBSCAN) - eps={eps:.3f}, min_samples={min_samples}"
        plot_image = plot_clusters(
            df=df,
            x_col="average_sale_price",
            y_col="total_quantity_sold",
            cluster_col="cluster",
            title=plot_title,
            xlabel="Average Sale Price",
            ylabel="Total Quantity Sold"
        )
        
        # Get outliers
        outliers = df[df['cluster'] == -1]
        logger.info(f"Found {len(outliers)} outliers")
        
        # Calculate cluster statistics
        logger.info("Calculating cluster statistics...")
        cluster_stats = []
        for cluster in sorted(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster]
            stats = {
                "cluster": int(cluster),
                "product_count": int(len(cluster_data)),
                "average_price": float(cluster_data['average_sale_price'].mean()),
                "total_quantity": float(cluster_data['total_quantity_sold'].sum())
            }
            cluster_stats.append(stats)
        
        # Convert DataFrame to dict with native Python types
        outliers_dict = outliers[["product_id", "product_name", "average_sale_price", "total_quantity_sold"]].copy()
        outliers_dict['average_sale_price'] = outliers_dict['average_sale_price'].astype(float)
        outliers_dict['total_quantity_sold'] = outliers_dict['total_quantity_sold'].astype(float)
        
        clusters_dict = df[["product_id", "product_name", "cluster"]].copy()
        clusters_dict['cluster'] = clusters_dict['cluster'].astype(int)
        
        result = {
            "total_products": int(len(df)),
            "number_of_clusters": int(len(df['cluster'].unique()) - 1),
            "outliers_count": int(len(outliers)),
            "parameters": {
                "eps": float(eps),
                "min_samples": int(min_samples)
            },
            
            "outliers": outliers_dict.to_dict('records'),
            "cluster_statistics": cluster_stats,
            "clusters": clusters_dict.to_dict('records')
        }
        
        logger.info("Product analysis completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error in product analysis: {str(e)}", exc_info=True)
        raise 