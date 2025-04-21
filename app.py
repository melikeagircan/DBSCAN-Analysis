from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from analyses.customer_analysis import analyze_customers
from analyses.product_analysis import analyze_products
from analyses.supplier_analysis import analyze_suppliers
from analyses.country_analysis import analyze_countries
from typing import Dict, Any

app = FastAPI(
    title="DBSCAN Clustering API",
    description="API for performing DBSCAN clustering analysis on business data",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/customers", response_model=Dict[str, Any])
async def get_customer_analysis():
    """
    Analyze customer behavior using DBSCAN clustering
    """
    try:
        return analyze_customers()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/products", response_model=Dict[str, Any])
async def get_product_analysis():
    """
    Analyze product performance using DBSCAN clustering
    """
    try:
        return analyze_products()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/suppliers", response_model=Dict[str, Any])
async def get_supplier_analysis():
    """
    Analyze supplier performance using DBSCAN clustering
    """
    try:
        return analyze_suppliers()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/countries", response_model=Dict[str, Any])
async def get_country_analysis():
    """
    Analyze country-based sales patterns using DBSCAN clustering
    """
    try:
        return analyze_countries()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000) 