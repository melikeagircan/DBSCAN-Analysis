import pandas as pd
from sklearn.cluster import DBSCAN
from utils import get_db_connection, standardize_features, find_optimal_parameters, plot_clusters

def analyze_suppliers():
    """Analyze supplier performance using DBSCAN clustering with optimized parameters"""
    engine = get_db_connection()
    
    query = """
    SELECT 
        s.supplier_id, 
        COUNT(p.product_id) as supplied_products_count,
        SUM(od.quantity) as total_sales_quantity,
        AVG(od.unit_price) as average_sale_price,
        AVG(sub.customer_count) as average_customer_count
    FROM suppliers s
    INNER JOIN products p ON (p.supplier_id = s.supplier_id)
    INNER JOIN order_details od ON (p.product_id = od.product_id)
    INNER JOIN orders o ON (o.order_id = od.order_id)
    INNER JOIN (
        SELECT
            p.product_id,
            COUNT(o.customer_id) as customer_count
        FROM products p
        INNER JOIN order_details od ON (p.product_id = od.product_id)
        INNER JOIN orders o ON (od.order_id = o.order_id)
        GROUP BY p.product_id
    ) sub ON (p.product_id = sub.product_id)
    GROUP BY s.supplier_id
    HAVING COUNT(p.product_id) > 0
    """
    
    df = pd.read_sql(query, engine)
    
    # Prepare features
    X = df[["supplied_products_count", "total_sales_quantity", "average_sale_price", "average_customer_count"]]
    X_scaled = standardize_features(X)
    
    # Find optimal parameters
    eps, min_samples = find_optimal_parameters(X_scaled)
    
    # Apply DBSCAN with optimal parameters
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df["cluster"] = dbscan.fit_predict(X_scaled)
    
    # Plot results
    plot_clusters(
        df=df,
        x_col="supplied_products_count",
        y_col="total_sales_quantity",
        cluster_col="cluster",
        title=f"Supplier Segmentation (DBSCAN) - eps={eps:.3f}, min_samples={min_samples}",
        xlabel="Supplied Products Count",
        ylabel="Total Sales Quantity"
    )
    
    # Get outliers
    outliers = df[df['cluster'] == -1]
    
    return {
        "total_suppliers": len(df),
        "number_of_clusters": len(df['cluster'].unique()) - 1,
        "outliers_count": len(outliers),
        "parameters": {
            "eps": eps,
            "min_samples": min_samples
        },
        "outliers": outliers[["supplier_id", "supplied_products_count", "total_sales_quantity", "average_sale_price", "average_customer_count"]].to_dict('records'),
        "clusters": df[["supplier_id", "cluster"]].to_dict('records')
    } 