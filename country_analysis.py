import pandas as pd
from sklearn.cluster import DBSCAN
from utils import get_db_connection, standardize_features, find_optimal_parameters, plot_clusters

def analyze_countries():
    """Analyze country-based sales patterns using DBSCAN clustering with optimized parameters"""
    engine = get_db_connection()
    
    query = """
    SELECT 
        c.country,
        COUNT(o.order_id) as total_orders,
        AVG(sub.order_amount) as average_order_amount,
        AVG(sub.product_quantity) as products_per_order
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN (
        SELECT 
            od.order_id,
            SUM(od.unit_price * od.quantity) as order_amount,
            SUM(od.quantity) as product_quantity
        FROM order_details od
        GROUP BY od.order_id
    ) sub ON o.order_id = sub.order_id
    GROUP BY c.country
    HAVING COUNT(o.order_id) > 0
    """
    
    df = pd.read_sql(query, engine)
    
    # Prepare features
    X = df[["total_orders", "average_order_amount", "products_per_order"]]
    X_scaled = standardize_features(X)
    
    # Find optimal parameters
    eps, min_samples = find_optimal_parameters(X_scaled)
    
    # Apply DBSCAN with optimal parameters
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df["cluster"] = dbscan.fit_predict(X_scaled)
    
    # Plot results
    plot_clusters(
        df=df,
        x_col="total_orders",
        y_col="average_order_amount",
        cluster_col="cluster",
        title=f"Country Sales Pattern Analysis (DBSCAN) - eps={eps:.3f}, min_samples={min_samples}",
        xlabel="Total Orders",
        ylabel="Average Order Amount"
    )
    
    # Get outliers
    outliers = df[df['cluster'] == -1]
    
    return {
        "total_countries": len(df),
        "number_of_clusters": len(df['cluster'].unique()) - 1,
        "outliers_count": len(outliers),
        "parameters": {
            "eps": eps,
            "min_samples": min_samples
        },
        "outliers": outliers[["country", "total_orders", "average_order_amount"]].to_dict('records'),
        "clusters": df[["country", "cluster"]].to_dict('records')
    } 