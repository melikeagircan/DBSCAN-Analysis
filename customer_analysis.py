import pandas as pd
from sklearn.cluster import DBSCAN
from utils import get_db_connection, standardize_features, find_optimal_parameters, plot_clusters

def analyze_customers():
    """Analyze customer behavior using DBSCAN clustering with optimized parameters"""
    engine = get_db_connection()
    
    query = """
    SELECT 
        c.customer_id,
        c.company_name, 
        COUNT(o.order_id) as total_orders,
        SUM(od.unit_price * od.quantity * (1 - od.discount)) as total_spends,
        AVG(od.unit_price * od.quantity) as avg_order_value
    FROM customers c
    INNER JOIN orders o ON c.customer_id = o.customer_id
    INNER JOIN order_details od ON o.order_id = od.order_id
    GROUP BY c.customer_id, c.company_name
    HAVING COUNT(o.order_id) > 0
    """
    
    df = pd.read_sql(query, engine)
    
    # Prepare features
    X = df[["total_orders", "total_spends", "avg_order_value"]]
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
        y_col="total_spends",
        cluster_col="cluster",
        title=f"Customer Segmentation (DBSCAN) - eps={eps:.3f}, min_samples={min_samples}",
        xlabel="Total Orders",
        ylabel="Total Spending"
    )
    
    # Get outliers
    outliers = df[df['cluster'] == -1]
    
    return {
        "total_customers": len(df),
        "number_of_clusters": len(df['cluster'].unique()) - 1,
        "outliers_count": len(outliers),
        "parameters": {
            "eps": eps,
            "min_samples": min_samples
        },
        "outliers": outliers[["customer_id", "company_name", "total_orders", "total_spends"]].to_dict('records'),
        "clusters": df[["customer_id", "company_name", "cluster"]].to_dict('records')
    } 